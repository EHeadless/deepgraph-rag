"""
Step 3: Fix Patient Theme Quotes
Map actual Reddit posts to the themes we created, for all diseases.
This replaces the random/irrelevant quotes with actual relevant patient experiences.
"""
import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

driver = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "neurograph2025"))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_diseases_with_themes():
    """Get all diseases that have PatientExperience themes."""
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Disease)-[:HAS_PATIENT_EXPERIENCE]->(pe:PatientExperience)
            RETURN DISTINCT d.name as disease
        """)
        return [r['disease'] for r in result]


def get_reddit_posts_for_disease(disease_name: str):
    """Get all Reddit posts discussing this disease."""
    with driver.session() as session:
        result = session.run("""
            MATCH (r:RedditPost)-[:DISCUSSES]->(d:Disease)
            WHERE d.name = $disease
            AND r.title IS NOT NULL
            RETURN r.title as title, r.url as url, r.subreddit as subreddit
            LIMIT 200
        """, disease=disease_name)
        return [dict(r) for r in result]


def get_themes_for_disease(disease_name: str):
    """Get all themes for this disease."""
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Disease {name: $disease})-[:HAS_PATIENT_EXPERIENCE]->(pe:PatientExperience)
            RETURN pe.name as theme_name, pe.description as description, pe.clinical_insight as insight
            ORDER BY pe.theme_id
        """, disease=disease_name)
        return [dict(r) for r in result]


def categorize_posts_into_themes(posts: list, themes: list, disease_name: str):
    """Use AI to categorize Reddit posts into the established themes."""
    
    if not posts or not themes:
        return {}
    
    # Prepare data for AI
    theme_descriptions = {}
    for theme in themes:
        theme_descriptions[theme['theme_name']] = theme['description']
    
    # Process posts in batches of 20
    all_categorizations = {}
    for theme in themes:
        all_categorizations[theme['theme_name']] = []
    
    batch_size = 20
    for i in range(0, len(posts), batch_size):
        batch = posts[i:i + batch_size]
        
        print(f"  Processing posts {i+1}-{min(i+batch_size, len(posts))} of {len(posts)}...")
        
        post_titles = [p['title'] for p in batch]
        
        prompt = f"""You are categorizing patient Reddit posts about {disease_name} into themes.

THEMES:
{json.dumps(theme_descriptions, indent=2)}

REDDIT POST TITLES:
{json.dumps(post_titles)}

For each post title, determine which theme it belongs to. Some posts might not fit any theme (use "none").

Return JSON mapping each post title to its best theme:
{{
    "post title 1": "theme_name",
    "post title 2": "theme_name",
    "post title 3": "none"
}}

Be generous - if a post could reasonably relate to a theme, include it."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            content = response.choices[0].message.content.strip()
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            
            categorizations = json.loads(content)
            
            # Add posts to their themes
            for post in batch:
                title = post['title']
                theme = categorizations.get(title, 'none')
                if theme != 'none' and theme in all_categorizations:
                    all_categorizations[theme].append(post)
                    
        except Exception as e:
            print(f"    Error processing batch: {e}")
            continue
    
    return all_categorizations


def update_theme_quotes(disease_name: str, categorized_posts: dict):
    """Update PatientExperience nodes with properly categorized quotes."""
    with driver.session() as session:
        for theme_name, posts in categorized_posts.items():
            if not posts:
                continue
            
            # Get top quotes (up to 10)
            quotes = [p['title'] for p in posts[:10]]
            
            session.run("""
                MATCH (d:Disease {name: $disease})-[:HAS_PATIENT_EXPERIENCE]->(pe:PatientExperience {name: $theme})
                SET pe.sample_quotes = $quotes,
                    pe.quote_count = $count
            """, disease=disease_name, theme=theme_name, quotes=quotes, count=len(posts))
            
            print(f"    Updated '{theme_name}': {len(posts)} relevant posts")


def process_disease(disease_name: str):
    """Process one disease completely."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {disease_name}")
    print(f"{'='*60}")
    
    # Get Reddit posts
    print("Getting Reddit posts...")
    posts = get_reddit_posts_for_disease(disease_name)
    print(f"Found {len(posts)} Reddit posts")
    
    if len(posts) < 5:
        print("Not enough posts to categorize")
        return
    
    # Get existing themes
    print("Getting existing themes...")
    themes = get_themes_for_disease(disease_name)
    print(f"Found {len(themes)} themes: {[t['theme_name'] for t in themes]}")
    
    # Categorize posts into themes
    print("Categorizing posts into themes...")
    categorized_posts = categorize_posts_into_themes(posts, themes, disease_name)
    
    # Show results
    total_categorized = sum(len(posts) for posts in categorized_posts.values())
    print(f"\nCategorized {total_categorized} posts:")
    for theme, posts in categorized_posts.items():
        if posts:
            print(f"  {theme}: {len(posts)} posts")
    
    # Update database
    print("\nUpdating database...")
    update_theme_quotes(disease_name, categorized_posts)
    
    print(f"✅ {disease_name} complete!")


def main():
    print("=" * 60)
    print("STEP 3: FIXING PATIENT THEME QUOTES")
    print("=" * 60)
    print("Mapping actual Reddit posts to themes for proper relevance")
    
    # Get diseases with themes
    diseases = get_diseases_with_themes()
    print(f"\nFound diseases with themes: {diseases}")
    
    for disease in diseases:
        try:
            process_disease(disease)
        except Exception as e:
            print(f"ERROR processing {disease}: {e}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Verify the results
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Disease)-[:HAS_PATIENT_EXPERIENCE]->(pe:PatientExperience)
            RETURN d.name as disease, pe.name as theme, pe.quote_count as quotes
            ORDER BY d.name, pe.name
        """)
        
        current_disease = None
        for r in result:
            if r['disease'] != current_disease:
                current_disease = r['disease']
                print(f"\n{current_disease}:")
            print(f"  {r['theme']}: {r['quotes'] or 0} relevant quotes")
    
    print("\n✅ Step 3 complete!")
    print("\nNow the patient themes will show ACTUAL relevant quotes instead of random ones.")


if __name__ == "__main__":
    main()