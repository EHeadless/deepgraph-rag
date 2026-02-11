"""
COMPLETE Reddit download - posts AND comments from all relevant sources
"""
import requests
import json
import time
import os
from tqdm import tqdm

# Expanded subreddit list
DISEASE_SUBREDDITS = {
    "Alzheimer's Disease": [
        "alzheimers", "Alzheimers", "dementia", "earlyonsetalzheimers", 
        "CaregiverSupport", "AgingParents", "caregivers", "eldercare"
    ],
    "Parkinson's Disease": [
        "parkinsons", "Parkinsons", "parkinsonsdisease", "caregiversofreddit",
        "movement_disorders"
    ],
    "Multiple Sclerosis": [
        "multiplesclerosis", "MultipleSclerosis", "MS_Multiple_Sclerosis",
        "chronicillness", "autoimmune"
    ],
    "Amyotrophic Lateral Sclerosis": [
        "ALS", "alswarriors", "MND", "motorneurondisease"
    ],
    "Huntington's Disease": [
        "huntingtonsdisease", "Huntingtons", "hdinfo"
    ],
    "Lewy Body Dementia": [
        "dementia", "lewybodydementia", "LewyBodyDementia", "CaregiverSupport"
    ],
    "Creutzfeldt-Jakob Disease": [
        "cjd", "prion", "CJD", "rareconditions", "rarediseases"
    ],
    "Frontotemporal Dementia": [
        "dementia", "FTD", "frontotemporaldementia", "CaregiverSupport"
    ],
    "Vascular Dementia": [
        "dementia", "vasculardementia", "stroke", "strokesurvivors"
    ],
    "Progressive Supranuclear Palsy": [
        "psp", "PSP", "rareconditions", "rarediseases", "CaregiverSupport"
    ],
    "Corticobasal Degeneration": [
        "dementia", "cbd", "corticobasal", "rareconditions", "rarediseases"
    ],
    "Motor Neuron Disease": [
        "MND", "motorneurondisease", "ALS"
    ],
    "Spinocerebellar Ataxia": [
        "ataxia", "SCA", "spinocerebellarataxia", "rareconditions"
    ],
    "Multiple System Atrophy": [
        "MSA", "multiplesystematrophy", "rareconditions", "parkinsons"
    ],
    "Prion Disease": [
        "prion", "cjd", "rarediseases", "neurology"
    ],
    "Essential Tremor": [
        "essentialtremor", "tremor"
    ],
    "Restless Leg Syndrome": [
        "restlesslegs", "RLS"
    ],
    "Myasthenia Gravis": [
        "myastheniagravis", "MG", "autoimmune"
    ],
    "Guillain-Barre Syndrome": [
        "guillainbarre", "GBS", "autoimmune"
    ],
    "Chronic Fatigue Syndrome": [
        "cfs", "chronicfatiguesyndrome", "mecfs"
    ],
}

# MASSIVE symptom search terms - every possible way someone might describe symptoms
SYMPTOM_QUERIES = [
    # === ASKING FOR SHARED EXPERIENCES ===
    "anyone else experience", "does anyone have", "anyone else notice",
    "is this normal", "has anyone had", "do you also have", "am I the only one",
    "common to have", "is it common", "DAE", "does anybody else",
    "is this a symptom", "is this related", "could this be", "new symptom",
    "strange symptom", "unusual symptom", "weird symptom", "odd symptom",
    "first symptom", "early symptom", "initial symptom", "presenting symptom",
    "worst symptom", "most annoying symptom", "most difficult symptom",
    
    # === MOTOR - TREMOR/SHAKING ===
    "tremor", "tremors", "shaking", "shaky", "trembling", "quiver", "quivering",
    "resting tremor", "action tremor", "intention tremor", "postural tremor",
    "internal tremor", "internal shaking", "vibrating feeling", "buzzing sensation",
    "hand tremor", "leg tremor", "head tremor", "voice tremor", "chin tremor",
    "tremor when eating", "tremor when writing", "tremor when tired", "tremor when stressed",
    
    # === MOTOR - STIFFNESS/RIGIDITY ===
    "stiffness", "stiff", "rigidity", "rigid", "tight muscles", "muscle tightness",
    "cogwheel", "lead pipe", "morning stiffness", "neck stiffness", "back stiffness",
    "frozen", "freezing", "freeze up", "stuck", "can't move", "locked up",
    "freezing of gait", "FOG", "feet stuck", "glued to floor",
    
    # === MOTOR - SLOWNESS ===
    "slowness", "slow movement", "bradykinesia", "moving slowly", "takes forever",
    "slow to start", "slow reactions", "delayed movement", "sluggish",
    "everything takes longer", "lost speed", "move like molasses",
    
    # === MOTOR - WALKING/GAIT ===
    "walking difficulty", "trouble walking", "gait problems", "shuffling",
    "shuffle", "short steps", "festination", "small steps", "dragging feet",
    "foot drop", "tripping", "stumbling", "falling", "falls", "fell",
    "balance problems", "balance issues", "unsteady", "wobbly", "off balance",
    "veering", "listing", "leaning", "tilting", "stooped", "bent over",
    "posture problems", "stooped posture", "forward lean", "camptocormia",
    "retropulsion", "falling backward", "propulsion", "falling forward",
    "can't walk straight", "zigzag walking", "wide gait", "narrow gait",
    "difficulty turning", "turning problems", "pivot", "turning in bed",
    
    # === MOTOR - COORDINATION ===
    "coordination", "uncoordinated", "clumsy", "awkward movements", "ataxia",
    "dysmetria", "overshooting", "undershooting", "past pointing",
    "can't judge distance", "bumping into things", "knocking things over",
    "difficulty with fine motor", "fine motor skills", "dexterity",
    
    # === MOTOR - HANDS/ARMS ===
    "hand weakness", "weak hands", "grip strength", "can't grip", "dropping things",
    "buttoning", "can't button", "zippers", "tying shoes", "writing difficulty",
    "handwriting", "small handwriting", "micrographia", "typing difficulty",
    "using utensils", "can't use fork", "can't use knife", "feeding myself",
    "arm weakness", "heavy arms", "arm fatigue", "can't lift arms",
    
    # === MOTOR - LEGS/FEET ===
    "leg weakness", "weak legs", "heavy legs", "leg fatigue", "legs give out",
    "foot weakness", "foot drop", "toe drag", "can't lift foot",
    "leg cramps", "calf cramps", "charley horse", "muscle cramps",
    "restless legs", "RLS", "urge to move legs", "legs won't stay still",
    "leg stiffness", "knee buckling", "knees give out",
    
    # === MOTOR - INVOLUNTARY MOVEMENTS ===
    "involuntary movements", "dyskinesia", "chorea", "writhing", "twisting",
    "dystonia", "muscle spasms", "spasms", "twitching", "twitches",
    "fasciculations", "muscle twitches", "eye twitching", "eyelid twitch",
    "myoclonus", "jerking", "jerks", "startle", "jumping",
    "tics", "repetitive movements",
    
    # === SPEECH ===
    "speech problems", "speaking difficulty", "trouble speaking", "dysarthria",
    "slurred speech", "slurring", "mumbling", "unclear speech", "garbled",
    "quiet voice", "soft voice", "hypophonia", "weak voice", "voice changes",
    "hoarse", "hoarseness", "voice tremor", "shaky voice",
    "stuttering", "stammering", "hesitation", "word finding", "can't find words",
    "tip of tongue", "wrong words", "word substitution", "paraphasia",
    "slow speech", "halting speech", "monotone", "flat voice",
    "nasal speech", "nasality", "hypernasal",
    
    # === SWALLOWING/EATING ===
    "swallowing difficulty", "trouble swallowing", "dysphagia", "choking",
    "food getting stuck", "pills getting stuck", "liquid going down wrong",
    "aspiration", "coughing when eating", "coughing when drinking",
    "drooling", "excess saliva", "sialorrhea", "can't control saliva",
    "chewing difficulty", "jaw weakness", "jaw fatigue", "tired jaw",
    "weight loss", "losing weight", "can't eat enough", "no appetite",
    "taste changes", "food tastes different", "metallic taste",
    
    # === BREATHING ===
    "breathing difficulty", "shortness of breath", "breathless", "dyspnea",
    "can't take deep breath", "shallow breathing", "weak breathing",
    "respiratory", "ventilator", "BiPAP", "CPAP", "oxygen",
    "sleep apnea", "stop breathing", "choking at night",
    
    # === PAIN ===
    "pain", "aching", "aches", "sore", "soreness", "hurts",
    "muscle pain", "myalgia", "joint pain", "arthralgia", "bone pain",
    "nerve pain", "neuropathic pain", "burning pain", "shooting pain",
    "stabbing pain", "electric shock pain", "pins and needles",
    "headache", "head pain", "migraine", "tension headache",
    "neck pain", "back pain", "shoulder pain", "hip pain", "knee pain",
    "facial pain", "jaw pain", "TMJ",
    
    # === SENSORY ===
    "numbness", "numb", "can't feel", "loss of sensation", "hypoesthesia",
    "tingling", "pins and needles", "paresthesia", "prickling",
    "burning sensation", "burning feeling", "hot feeling", "cold feeling",
    "hypersensitivity", "oversensitive", "allodynia", "light touch hurts",
    "vibration sense", "position sense", "proprioception",
    
    # === VISION ===
    "vision problems", "trouble seeing", "blurred vision", "blurry",
    "double vision", "diplopia", "seeing double",
    "eye movement", "can't move eyes", "gaze palsy", "eye tracking",
    "light sensitivity", "photophobia", "bright lights hurt",
    "visual hallucinations", "seeing things", "seeing people", "seeing animals",
    "peripheral vision", "tunnel vision", "blind spots", "scotoma",
    "depth perception", "judging distance", "3D vision",
    "reading difficulty", "can't read", "words jumping", "lines blurring",
    "dry eyes", "eye fatigue", "eye strain",
    
    # === HEARING ===
    "hearing problems", "hearing loss", "can't hear", "hard of hearing",
    "tinnitus", "ringing in ears", "buzzing in ears",
    "auditory hallucinations", "hearing things", "hearing voices",
    "sound sensitivity", "hyperacusis", "sounds too loud",
    
    # === SMELL/TASTE ===
    "loss of smell", "can't smell", "anosmia", "hyposmia", "smell changes",
    "loss of taste", "can't taste", "taste changes", "dysgeusia",
    
    # === BLADDER ===
    "bladder problems", "urinary problems", "incontinence", "accidents",
    "urgency", "urinary urgency", "gotta go now", "can't hold it",
    "frequency", "urinating often", "peeing all the time", "nocturia",
    "hesitancy", "trouble starting", "weak stream", "retention",
    "UTI", "urinary tract infection", "bladder infection",
    
    # === BOWEL ===
    "bowel problems", "constipation", "constipated", "can't go",
    "diarrhea", "loose stools", "bowel incontinence", "fecal incontinence",
    "bloating", "gas", "abdominal pain", "stomach problems",
    
    # === SEXUAL ===
    "sexual problems", "erectile dysfunction", "ED", "impotence",
    "libido", "sex drive", "intimacy issues",
    
    # === SLEEP ===
    "sleep problems", "insomnia", "can't sleep", "trouble sleeping",
    "staying asleep", "waking up", "middle of night", "early waking",
    "excessive sleep", "sleeping too much", "hypersomnia", "always tired",
    "sleep attacks", "falling asleep suddenly", "narcolepsy",
    "REM sleep disorder", "acting out dreams", "RBD", "violent dreams",
    "vivid dreams", "nightmares", "night terrors", "screaming at night",
    "restless sleep", "tossing and turning", "can't get comfortable",
    "sleep apnea", "snoring", "stop breathing sleep",
    "daytime sleepiness", "drowsy", "dozing off", "napping",
    "sleep schedule", "circadian", "sundowning", "worse at night",
    
    # === FATIGUE/ENERGY ===
    "fatigue", "tired", "exhausted", "exhaustion", "no energy",
    "weakness", "weak", "wiped out", "drained", "depleted",
    "chronic fatigue", "always tired", "never rested", "bone tired",
    "exercise intolerance", "can't exercise", "wiped out after",
    "post-exertional", "crash after activity", "payback",
    
    # === COGNITIVE - MEMORY ===
    "memory problems", "memory loss", "forgetful", "forgetting",
    "short term memory", "can't remember recent", "forgot what just said",
    "long term memory", "can't remember past", "childhood memories",
    "forgetting names", "can't remember names", "name recall",
    "forgetting words", "word recall", "vocabulary",
    "forgetting appointments", "forgetting events", "forgot meeting",
    "forgetting to take medication", "forgot pills", "medication management",
    "repeating questions", "asking same thing", "repetitive questions",
    "repeating stories", "telling same story", "already told you",
    "losing things", "misplacing", "can't find keys", "can't find phone",
    "getting lost", "lost driving", "lost in familiar places",
    "not recognizing", "didn't recognize", "recognition problems",
    "faces", "prosopagnosia", "can't recognize faces",
    
    # === COGNITIVE - THINKING ===
    "confusion", "confused", "disoriented", "disorientation",
    "brain fog", "foggy", "fuzzy thinking", "can't think clearly",
    "slow thinking", "processing speed", "takes longer to understand",
    "concentration", "can't concentrate", "focus", "can't focus",
    "attention", "attention problems", "easily distracted", "ADHD-like",
    "decision making", "can't decide", "indecisive", "overwhelmed by choices",
    "problem solving", "can't figure out", "puzzles", "logic",
    "planning", "can't plan", "organizing", "can't organize",
    "sequencing", "order of steps", "following instructions",
    "multitasking", "can't do two things", "one thing at a time",
    "abstract thinking", "metaphors", "jokes", "sarcasm",
    
    # === COGNITIVE - LANGUAGE ===
    "language problems", "aphasia", "can't understand speech",
    "reading comprehension", "can't understand what read",
    "writing problems", "can't write", "agraphia",
    "following conversations", "lost in conversation", "can't keep up",
    
    # === COGNITIVE - SPATIAL ===
    "spatial problems", "visuospatial", "judging distances",
    "getting lost", "navigation", "can't find way",
    "reading maps", "directions", "left and right confusion",
    "parking", "driving spatial", "misjudging space",
    
    # === COGNITIVE - EXECUTIVE ===
    "executive function", "planning", "organizing", "initiating",
    "task completion", "can't finish", "starting tasks", "can't start",
    "time management", "estimating time", "always late",
    "goal-directed behavior", "purposeful action",
    
    # === BEHAVIORAL - PERSONALITY ===
    "personality change", "not themselves", "different person",
    "behavior change", "acting different", "out of character",
    "disinhibition", "inappropriate behavior", "no filter", "saying things",
    "impulsivity", "impulsive", "acting without thinking",
    "compulsive", "compulsions", "can't stop", "repetitive behavior",
    "hoarding", "collecting", "can't throw away",
    "rituals", "routines", "rigid routines", "upset by change",
    "obsessive", "fixated", "preoccupied",
    
    # === BEHAVIORAL - SOCIAL ===
    "social withdrawal", "isolating", "doesn't want to see people",
    "loss of interest", "doesn't care", "nothing interests",
    "loss of empathy", "doesn't care about others", "cold",
    "inappropriate social", "socially inappropriate", "embarrassing behavior",
    
    # === BEHAVIORAL - MOTIVATION ===
    "apathy", "no motivation", "doesn't want to do anything",
    "avolition", "can't initiate", "needs prompting",
    "abulia", "lack of will", "indifference",
    
    # === MOOD ===
    "depression", "depressed", "sad", "hopeless", "worthless",
    "crying", "tearful", "emotional", "crying spells", "weepy",
    "anxiety", "anxious", "worried", "nervous", "panic",
    "panic attacks", "anxiety attacks", "heart racing anxiety",
    "irritability", "irritable", "snapping", "short temper", "angry outbursts",
    "anger", "angry", "rage", "aggression", "aggressive",
    "mood swings", "emotional lability", "up and down", "rapid mood changes",
    "pseudobulbar", "PBA", "laughing crying", "inappropriate emotion",
    "flat affect", "no emotion", "blank expression", "emotionless",
    
    # === PSYCHIATRIC ===
    "hallucinations", "seeing things", "hearing things", "visual hallucinations",
    "seeing people", "seeing shadows", "seeing animals", "seeing bugs",
    "auditory hallucinations", "hearing voices", "hearing music", "hearing sounds",
    "delusions", "paranoia", "paranoid", "suspicious", "thinks being watched",
    "persecution", "thinks people against them", "conspiracy",
    "jealousy delusion", "infidelity", "cheating accusation",
    "misidentification", "Capgras", "thinks impostor", "thinks not real",
    "psychosis", "psychotic", "break from reality",
    "agitation", "agitated", "restless", "can't sit still", "pacing",
    
    # === AUTONOMIC ===
    "blood pressure", "orthostatic", "dizzy standing", "lightheaded standing",
    "fainting", "syncope", "passing out", "near faint",
    "heart rate", "racing heart", "palpitations", "bradycardia",
    "sweating", "excessive sweating", "hyperhidrosis", "night sweats",
    "temperature regulation", "too hot", "too cold", "can't regulate temperature",
    "skin changes", "oily skin", "seborrhea", "dry skin",
    
    # === DAILY ACTIVITIES ===
    "dressing", "can't dress", "buttons", "zippers", "shoes",
    "bathing", "showering", "can't bathe", "needs help bathing",
    "toileting", "bathroom help", "wiping", "hygiene",
    "eating", "feeding", "can't feed self", "needs to be fed",
    "grooming", "brushing teeth", "combing hair", "shaving",
    "cooking", "can't cook", "kitchen safety", "burning food", "leaving stove on",
    "cleaning", "housework", "can't clean", "hoarding mess",
    "laundry", "can't do laundry", "wearing same clothes",
    "managing money", "finances", "bills", "checkbook", "scammed",
    "medication management", "forgetting pills", "wrong dose",
    "driving", "can't drive", "got lost driving", "accidents", "gave up driving",
    "working", "can't work", "lost job", "disability", "had to retire",
    "using phone", "technology", "can't use remote", "can't use computer",
    
    # === PROGRESSION/TIMELINE ===
    "getting worse", "progressing", "declining", "deteriorating",
    "rapid progression", "fast decline", "quickly getting worse",
    "slow progression", "gradual", "slowly getting worse",
    "stable", "plateau", "not getting worse",
    "fluctuating", "good days bad days", "comes and goes",
    "first noticed", "started with", "began with", "initially",
    "looking back", "in hindsight", "before diagnosis",
    "stages", "early stage", "middle stage", "late stage", "advanced",
    
    # === CAREGIVER OBSERVATIONS ===
    "caregiver", "caring for", "my mom", "my dad", "my husband", "my wife",
    "my parent", "loved one", "family member", "spouse",
    "noticed changes", "I've noticed", "started noticing",
    "doesn't realize", "lacks awareness", "anosognosia", "denies problems",
    "safety concerns", "wandering", "getting lost", "leaving house",
    "sundowning", "worse in evening", "evening confusion",
    "24/7 care", "full time care", "nursing home", "memory care",
]

def search_reddit(query: str, subreddit: str = None, limit: int = 25) -> list:
    """Search Reddit for posts."""
    headers = {"User-Agent": "NeuroGraph Research Bot 1.0"}
    
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "restrict_sr": "true", "limit": limit, "sort": "relevance", "t": "all"}
    else:
        url = "https://www.reddit.com/search.json"
        params = {"q": query, "limit": limit, "sort": "relevance", "t": "all"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            posts = []
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                if post.get("selftext") and len(post.get("selftext", "")) > 50:
                    posts.append({
                        "id": post.get("id"),
                        "title": post.get("title"),
                        "selftext": post.get("selftext", "")[:4000],
                        "subreddit": post.get("subreddit"),
                        "url": f"https://reddit.com{post.get('permalink', '')}",
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "created_utc": post.get("created_utc")
                    })
            return posts
        elif response.status_code == 429:
            print(f"  Rate limited, waiting 60s...")
            time.sleep(60)
            return []
        return []
    except Exception as e:
        return []

def get_post_comments(post_id: str, subreddit: str, limit: int = 50) -> list:
    """Get comments for a post."""
    headers = {"User-Agent": "NeuroGraph Research Bot 1.0"}
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    params = {"limit": limit, "sort": "best"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            comments = []
            if len(data) > 1:
                for child in data[1].get("data", {}).get("children", []):
                    comment = child.get("data", {})
                    body = comment.get("body", "")
                    if body and len(body) > 30 and body != "[deleted]" and body != "[removed]":
                        comments.append({
                            "id": comment.get("id"),
                            "body": body[:2000],
                            "score": comment.get("score", 0)
                        })
            return comments
        return []
    except:
        return []

def search_subreddit_posts(subreddit: str, limit: int = 100) -> list:
    """Get posts from a subreddit."""
    headers = {"User-Agent": "NeuroGraph Research Bot 1.0"}
    posts = []
    
    for sort in ["top", "hot", "new"]:
        for time_filter in ["all", "year", "month"]:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {"limit": min(limit, 100), "t": time_filter}
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for child in data.get("data", {}).get("children", []):
                        post = child.get("data", {})
                        if post.get("selftext") and len(post.get("selftext", "")) > 50:
                            posts.append({
                                "id": post.get("id"),
                                "title": post.get("title"),
                                "selftext": post.get("selftext", "")[:4000],
                                "subreddit": post.get("subreddit"),
                                "url": f"https://reddit.com{post.get('permalink', '')}",
                                "score": post.get("score", 0),
                                "num_comments": post.get("num_comments", 0),
                                "created_utc": post.get("created_utc")
                            })
                time.sleep(0.5)
            except:
                continue
    
    return posts

def main():
    os.makedirs("data", exist_ok=True)
    
    all_posts = []
    all_comments = []
    seen_post_ids = set()
    seen_comment_ids = set()
    
    print("=" * 70)
    print("COMPLETE REDDIT DOWNLOAD - POSTS AND COMMENTS")
    print("=" * 70)
    print(f"Diseases: {len(DISEASE_SUBREDDITS)}")
    print(f"Symptom queries: {len(SYMPTOM_QUERIES)}")
    print("=" * 70)
    
    for disease, subreddits in DISEASE_SUBREDDITS.items():
        print(f"\n🔍 {disease}")
        print("-" * 50)
        
        disease_posts = []
        disease_comments = []
        
        # 1. Get all posts from each subreddit
        for subreddit in subreddits:
            print(f"  📂 r/{subreddit}...")
            
            # Get top/hot/new posts
            posts = search_subreddit_posts(subreddit, limit=100)
            for post in posts:
                if post["id"] not in seen_post_ids:
                    post["disease"] = disease
                    disease_posts.append(post)
                    seen_post_ids.add(post["id"])
                    
                    # Get comments for high-engagement posts
                    if post["num_comments"] > 5:
                        comments = get_post_comments(post["id"], subreddit, limit=30)
                        for comment in comments:
                            if comment["id"] not in seen_comment_ids:
                                comment["disease"] = disease
                                comment["post_id"] = post["id"]
                                comment["post_title"] = post["title"]
                                disease_comments.append(comment)
                                seen_comment_ids.add(comment["id"])
                        time.sleep(0.3)
            
            time.sleep(1)
        
        # 2. Search symptom queries
        for subreddit in subreddits[:3]:
            for query in SYMPTOM_QUERIES[:100]:
                posts = search_reddit(query, subreddit, limit=5)
                for post in posts:
                    if post["id"] not in seen_post_ids:
                        post["disease"] = disease
                        disease_posts.append(post)
                        seen_post_ids.add(post["id"])
                time.sleep(0.3)
        
        # 3. Global search with disease name
        for query_base in ["symptoms", "experience", "diagnosed", "first signs", "progression"]:
            query = f"{disease} {query_base}"
            posts = search_reddit(query, None, limit=20)
            for post in posts:
                if post["id"] not in seen_post_ids:
                    post["disease"] = disease
                    disease_posts.append(post)
                    seen_post_ids.add(post["id"])
            time.sleep(0.5)
        
        print(f"  ✅ {len(disease_posts)} posts, {len(disease_comments)} comments")
        all_posts.extend(disease_posts)
        all_comments.extend(disease_comments)
        
        # Save progress
        with open("data/reddit_posts_complete.json", "w") as f:
            json.dump(all_posts, f)
        with open("data/reddit_comments.json", "w") as f:
            json.dump(all_comments, f)
    
    # Final save
    with open("data/reddit_posts_complete.json", "w") as f:
        json.dump(all_posts, f, indent=2)
    with open("data/reddit_comments.json", "w") as f:
        json.dump(all_comments, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Total posts: {len(all_posts)}")
    print(f"Total comments: {len(all_comments)}")
    
    from collections import Counter
    print("\nPosts per disease:")
    for disease, count in Counter(p["disease"] for p in all_posts).most_common():
        print(f"  {disease}: {count}")
    
    print("\nComments per disease:")
    for disease, count in Counter(c["disease"] for c in all_comments).most_common():
        print(f"  {disease}: {count}")

if __name__ == "__main__":
    main()
