"""
Extract DETAILED patient-reported symptoms from Reddit posts using GPT-4o-mini
WITH RESUME CAPABILITY
"""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

EXTRACTION_PROMPT = """Analyze this patient/caregiver post about {disease} and extract ALL symptoms with MAXIMUM SPECIFICITY.

Title: {title}
Post: {text}

RULES:
1. Be SPECIFIC - not "tremor" but "resting tremor in right hand when tired"
2. Extract EVERY symptom, even minor ones
3. Include timing, severity, body part when mentioned

Return ONLY valid JSON:
{{
    "physical_symptoms": ["symptom1", "symptom2"],
    "cognitive_symptoms": ["symptom1", "symptom2"],
    "behavioral_symptoms": ["symptom1", "symptom2"],
    "psychiatric_symptoms": ["symptom1", "symptom2"],
    "daily_impacts": ["impact1", "impact2"]
}}
"""

def extract_symptoms(post: dict) -> dict:
    text = post.get('selftext') or post.get('body', '')
    if not text or len(text) < 20:
        return {"physical_symptoms": [], "cognitive_symptoms": [], "behavioral_symptoms": [], "psychiatric_symptoms": [], "daily_impacts": []}
    
    prompt = EXTRACTION_PROMPT.format(
        disease=post['disease'],
        title=post.get('title', 'Comment')[:200],
        text=text[:2000]
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract symptoms with specificity. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        return json.loads(content.strip())
    except Exception as e:
        return {"physical_symptoms": [], "cognitive_symptoms": [], "behavioral_symptoms": [], "psychiatric_symptoms": [], "daily_impacts": []}

def main():
    with open("data/reddit_posts_complete.json", "r") as f:
        posts = json.load(f)
    
    comments = []
    if os.path.exists("data/reddit_comments.json"):
        with open("data/reddit_comments.json", "r") as f:
            comments = json.load(f)
    
    # Resume capability
    extracted = []
    processed_ids = set()
    
    if os.path.exists("data/reddit_symptoms_complete.json"):
        with open("data/reddit_symptoms_complete.json", "r") as f:
            extracted = json.load(f)
        processed_ids = {e.get("post_id") for e in extracted if e.get("post_id")}
        print(f"✅ Resuming from {len(extracted)} already processed items")
    
    posts_to_process = [p for p in posts if p.get("id") not in processed_ids]
    comments_to_process = [c for c in comments if c.get("id") not in processed_ids]
    
    total_remaining = len(posts_to_process) + len(comments_to_process)
    
    print(f"\nTotal posts: {len(posts)}")
    print(f"Total comments: {len(comments)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining to process: {total_remaining}")
    print(f"\n💰 Estimated cost (gpt-4o-mini): ~${total_remaining * 0.001:.0f}-${total_remaining * 0.002:.0f}")
    print(f"⏱️  Estimated time: ~{total_remaining * 0.5 / 60:.0f}-{total_remaining * 1 / 60:.0f} minutes")
    print("=" * 60)
    
    if total_remaining == 0:
        print("Nothing to process!")
        return
    
    input("Press Enter to continue (Ctrl+C to cancel)...")
    
    # Process posts
    if posts_to_process:
        print(f"\nProcessing {len(posts_to_process)} posts...")
        for i, post in enumerate(tqdm(posts_to_process)):
            symptoms = extract_symptoms(post)
            
            all_symptoms = (
                symptoms.get("physical_symptoms", []) +
                symptoms.get("cognitive_symptoms", []) +
                symptoms.get("behavioral_symptoms", []) +
                symptoms.get("psychiatric_symptoms", [])
            )
            
            extracted.append({
                "source": "post",
                "post_id": post.get("id"),
                "disease": post["disease"],
                "title": post.get("title", ""),
                "url": post.get("url", ""),
                "subreddit": post.get("subreddit", ""),
                "symptoms": symptoms,
                "all_symptoms": all_symptoms,
                "daily_impacts": symptoms.get("daily_impacts", [])
            })
            
            time.sleep(0.1)  # Faster with gpt-4o-mini
            
            if (i + 1) % 100 == 0:
                with open("data/reddit_symptoms_complete.json", "w") as f:
                    json.dump(extracted, f)
                total_symptoms = sum(len(e["all_symptoms"]) for e in extracted)
                print(f"\n  💾 Saved: {len(extracted)} items, {total_symptoms} symptoms")
    
    # Process comments
    if comments_to_process:
        print(f"\nProcessing {len(comments_to_process)} comments...")
        for i, comment in enumerate(tqdm(comments_to_process)):
            symptoms = extract_symptoms(comment)
            
            all_symptoms = (
                symptoms.get("physical_symptoms", []) +
                symptoms.get("cognitive_symptoms", []) +
                symptoms.get("behavioral_symptoms", []) +
                symptoms.get("psychiatric_symptoms", [])
            )
            
            extracted.append({
                "source": "comment",
                "post_id": comment.get("id"),
                "disease": comment["disease"],
                "title": comment.get("post_title", ""),
                "symptoms": symptoms,
                "all_symptoms": all_symptoms,
                "daily_impacts": symptoms.get("daily_impacts", [])
            })
            
            time.sleep(0.1)
            
            if (i + 1) % 100 == 0:
                with open("data/reddit_symptoms_complete.json", "w") as f:
                    json.dump(extracted, f)
    
    # Final save
    with open("data/reddit_symptoms_complete.json", "w") as f:
        json.dump(extracted, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    
    physical = []
    cognitive = []
    behavioral = []
    psychiatric = []
    impacts = []
    
    for item in extracted:
        s = item["symptoms"]
        physical.extend(s.get("physical_symptoms", []))
        cognitive.extend(s.get("cognitive_symptoms", []))
        behavioral.extend(s.get("behavioral_symptoms", []))
        psychiatric.extend(s.get("psychiatric_symptoms", []))
        impacts.extend(item.get("daily_impacts", []))
    
    print(f"\nPhysical symptoms: {len(set(physical))} unique")
    print(f"Cognitive symptoms: {len(set(cognitive))} unique")
    print(f"Behavioral symptoms: {len(set(behavioral))} unique")
    print(f"Psychiatric symptoms: {len(set(psychiatric))} unique")
    print(f"Daily life impacts: {len(set(impacts))} unique")
    
    total_unique = len(set(physical + cognitive + behavioral + psychiatric))
    print(f"\nTOTAL UNIQUE SYMPTOMS: {total_unique}")

if __name__ == "__main__":
    main()
