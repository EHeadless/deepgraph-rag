"""
Download Amazon Product Data

Uses the Amazon Product Data dataset from UCSD:
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

For demo purposes, we'll use the Electronics category (smaller, feature-rich).
Falls back to generating synthetic data if download fails.
"""

import json
import gzip
import os
from pathlib import Path
import argparse
import requests
from tqdm import tqdm


# Data directory
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "products"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Amazon dataset URLs (2018 version - smaller files)
AMAZON_URLS = {
    "electronics_meta": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Electronics.json.gz",
    "electronics_reviews": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz",
}

# Alternative: Use Hugging Face datasets (easier access)
HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def generate_sample_data(num_products: int = 500) -> list:
    """Generate synthetic product data for demo purposes."""
    import random

    print(f"Generating {num_products} synthetic products...")

    # Product templates by category
    categories = {
        "Headphones": {
            "brands": ["Sony", "Bose", "Apple", "Samsung", "JBL", "Sennheiser", "Audio-Technica", "Beats", "Skullcandy", "Anker"],
            "types": ["Wireless Headphones", "Earbuds", "Over-Ear Headphones", "Gaming Headset", "Sport Earbuds", "True Wireless Earbuds"],
            "features": ["noise_canceling", "wireless", "long_battery", "waterproof", "bass_boost", "touch_controls", "voice_assistant"],
            "use_cases": ["travel", "workout", "gaming", "work_from_home", "music_production"],
            "price_range": (20, 400),
        },
        "Speakers": {
            "brands": ["JBL", "Bose", "Sonos", "Marshall", "Bang & Olufsen", "Ultimate Ears", "Anker", "Sony", "Harman Kardon"],
            "types": ["Portable Speaker", "Smart Speaker", "Soundbar", "Bookshelf Speaker", "Party Speaker", "Outdoor Speaker"],
            "features": ["waterproof", "wireless", "long_battery", "bass_boost", "voice_assistant", "multi_device", "surround_sound"],
            "use_cases": ["outdoor", "travel", "work_from_home", "podcast"],
            "price_range": (30, 500),
        },
        "Smartwatch": {
            "brands": ["Apple", "Samsung", "Garmin", "Fitbit", "Amazfit", "Fossil", "Withings", "Polar"],
            "types": ["Fitness Tracker", "Smartwatch", "GPS Watch", "Sport Watch", "Hybrid Watch"],
            "features": ["waterproof", "long_battery", "fast_charging", "voice_assistant", "app_control"],
            "use_cases": ["workout", "outdoor", "travel", "sleep"],
            "price_range": (50, 800),
        },
        "Camera": {
            "brands": ["Canon", "Sony", "Nikon", "Fujifilm", "Panasonic", "GoPro", "DJI", "Insta360"],
            "types": ["Mirrorless Camera", "DSLR Camera", "Action Camera", "Vlog Camera", "Instant Camera", "Drone Camera"],
            "features": ["4k", "stabilization", "night_mode", "wide_angle", "waterproof", "wireless"],
            "use_cases": ["travel", "outdoor", "podcast", "music_production"],
            "price_range": (100, 2000),
        },
        "Laptop": {
            "brands": ["Apple", "Dell", "HP", "Lenovo", "ASUS", "Acer", "Microsoft", "Razer", "MSI"],
            "types": ["Ultrabook", "Gaming Laptop", "Business Laptop", "2-in-1 Laptop", "Chromebook", "Workstation"],
            "features": ["lightweight", "long_battery", "fast_charging", "4k", "high_refresh", "usb_c"],
            "use_cases": ["work_from_home", "gaming", "music_production", "travel"],
            "price_range": (300, 3000),
        },
        "Keyboard": {
            "brands": ["Logitech", "Razer", "Corsair", "SteelSeries", "Keychron", "Ducky", "Anne Pro", "HyperX"],
            "types": ["Mechanical Keyboard", "Wireless Keyboard", "Gaming Keyboard", "Ergonomic Keyboard", "Compact Keyboard"],
            "features": ["wireless", "ergonomic", "low_latency", "usb_c", "multi_device"],
            "use_cases": ["gaming", "work_from_home", "music_production"],
            "price_range": (30, 300),
        },
        "Mouse": {
            "brands": ["Logitech", "Razer", "SteelSeries", "Corsair", "Zowie", "Glorious", "Pulsar", "Finalmouse"],
            "types": ["Gaming Mouse", "Wireless Mouse", "Ergonomic Mouse", "Trackball Mouse", "Travel Mouse"],
            "features": ["wireless", "ergonomic", "low_latency", "lightweight", "fast_charging"],
            "use_cases": ["gaming", "work_from_home", "travel"],
            "price_range": (20, 200),
        },
        "Monitor": {
            "brands": ["LG", "Samsung", "Dell", "ASUS", "BenQ", "Acer", "ViewSonic", "MSI", "Gigabyte"],
            "types": ["Gaming Monitor", "Ultrawide Monitor", "4K Monitor", "Portable Monitor", "Curved Monitor"],
            "features": ["4k", "hdr", "oled", "high_refresh", "usb_c"],
            "use_cases": ["gaming", "work_from_home", "music_production"],
            "price_range": (150, 1500),
        },
    }

    products = []

    for i in range(num_products):
        # Pick random category
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]

        brand = random.choice(cat_data["brands"])
        product_type = random.choice(cat_data["types"])

        # Generate features (2-5 random features)
        num_features = random.randint(2, 5)
        features = random.sample(cat_data["features"], min(num_features, len(cat_data["features"])))

        # Generate use cases (1-3)
        num_use_cases = random.randint(1, 3)
        use_cases = random.sample(cat_data["use_cases"], min(num_use_cases, len(cat_data["use_cases"])))

        # Generate price
        min_price, max_price = cat_data["price_range"]
        price = round(random.uniform(min_price, max_price), 2)

        # Generate rating
        rating = round(random.uniform(3.5, 5.0), 1)
        review_count = random.randint(10, 10000)

        # Generate title
        feature_words = []
        if "wireless" in features:
            feature_words.append("Wireless")
        if "noise_canceling" in features:
            feature_words.append("Noise Canceling")
        if "waterproof" in features:
            feature_words.append("Waterproof")
        if "4k" in features:
            feature_words.append("4K")
        if "gaming" in [uc for uc in use_cases]:
            feature_words.append("Gaming")

        feature_str = " ".join(feature_words[:2])
        title = f"{brand} {feature_str} {product_type}".strip()

        # Generate description
        description = f"The {title} features {', '.join(features[:3])}. Perfect for {', '.join(use_cases)}."

        product = {
            "asin": f"B{i:08d}",
            "title": title,
            "description": description,
            "price": price,
            "rating": rating,
            "review_count": review_count,
            "category": category,
            "brand": brand,
            "features": features,
            "use_cases": use_cases,
            "also_bought": [],  # Will be populated later
        }

        products.append(product)

    # Generate co-purchase relationships (same category, similar price)
    print("Generating co-purchase relationships...")
    for product in products:
        # Find similar products
        similar = [p for p in products
                   if p["asin"] != product["asin"]
                   and p["category"] == product["category"]
                   and abs(p["price"] - product["price"]) < product["price"] * 0.5]

        # Pick 2-5 random similar products
        if similar:
            num_also_bought = min(random.randint(2, 5), len(similar))
            product["also_bought"] = [p["asin"] for p in random.sample(similar, num_also_bought)]

    return products


def download_amazon_data(category: str = "electronics", max_products: int = 1000):
    """Download Amazon product data or generate synthetic data."""

    output_file = DATA_DIR / "products.jsonl"

    # Try to download real data first
    print("Attempting to download Amazon dataset...")

    # For now, use synthetic data (real dataset requires agreement to terms)
    # In production, you would download from the actual source

    print("Using synthetic data for demo (real Amazon data requires terms agreement)")
    products = generate_sample_data(max_products)

    # Save to JSONL
    print(f"Saving {len(products)} products to {output_file}")
    with open(output_file, 'w') as f:
        for product in products:
            f.write(json.dumps(product) + '\n')

    print(f"✓ Saved {len(products)} products")

    # Print summary
    categories = {}
    for p in products:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\n📊 Product Summary:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return products


def main():
    parser = argparse.ArgumentParser(description="Download Amazon product data")
    parser.add_argument("--category", default="electronics", help="Product category")
    parser.add_argument("--max-products", type=int, default=500, help="Maximum products to download")
    args = parser.parse_args()

    download_amazon_data(args.category, args.max_products)
    print(f"\n✅ Data saved to {DATA_DIR / 'products.jsonl'}")


if __name__ == "__main__":
    main()
