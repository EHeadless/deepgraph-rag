"""
Product Recommendations Graph Schema

Knowledge Graph Structure:
    Product
      ├── HAS_FEATURE ──→ Feature (wireless, waterproof, noise-canceling...)
      ├── FOR_USE_CASE ──→ UseCase (travel, workout, gaming, work...)
      ├── IN_CATEGORY ──→ Category (Electronics, Sports, Home...)
      ├── MADE_BY ──→ Brand
      ├── BOUGHT_WITH ──→ Product (co-purchase patterns)
      └── SIMILAR_TO ──→ Product (feature overlap)

This enables queries like:
    - "Wireless headphones FOR running under $100"
    - "What premium features are now in budget products?"
    - "Build me a home office bundle"
    - "Underserved niches (rare feature + use case combos)"
"""

from deepgraph.core.schema import GraphSchema, NodeSchema, EdgeSchema


# ============== NODE SCHEMAS ==============

PRODUCT_NODE = NodeSchema(
    label="Product",
    id_field="asin",
    properties={
        "asin": str,           # Amazon Standard ID
        "title": str,
        "description": str,
        "price": float,
        "rating": float,
        "review_count": int,
        "category": str,
        "brand": str,
        "embedding": list,     # For vector search
    },
    indexes=["title", "price", "rating"],
    constraints=["asin"],
    vector_config={
        "field": "embedding",
        "dimensions": 1536,
        "similarity": "cosine"
    }
)

FEATURE_NODE = NodeSchema(
    label="Feature",
    id_field="name",
    properties={
        "name": str,           # e.g., "wireless", "waterproof", "noise_canceling"
        "display_name": str,   # e.g., "Wireless", "Waterproof", "Noise Canceling"
        "feature_type": str,   # "connectivity", "durability", "audio", etc.
    },
    indexes=["feature_type"],
    constraints=["name"]
)

USE_CASE_NODE = NodeSchema(
    label="UseCase",
    id_field="name",
    properties={
        "name": str,           # e.g., "travel", "workout", "gaming"
        "display_name": str,   # e.g., "Travel", "Workout", "Gaming"
        "description": str,
    },
    indexes=[],
    constraints=["name"]
)

CATEGORY_NODE = NodeSchema(
    label="Category",
    id_field="name",
    properties={
        "name": str,
        "display_name": str,
        "parent_category": str,
    },
    indexes=["parent_category"],
    constraints=["name"]
)

BRAND_NODE = NodeSchema(
    label="Brand",
    id_field="name",
    properties={
        "name": str,
        "display_name": str,
    },
    indexes=[],
    constraints=["name"]
)


# ============== EDGE SCHEMAS ==============

HAS_FEATURE = EdgeSchema(
    type="HAS_FEATURE",
    from_label="Product",
    to_label="Feature",
    properties={
        "value": str,          # e.g., "20 hours" for battery_life
        "confidence": float,   # extraction confidence
    }
)

FOR_USE_CASE = EdgeSchema(
    type="FOR_USE_CASE",
    from_label="Product",
    to_label="UseCase",
    properties={
        "relevance": float,    # how well suited (0-1)
    }
)

IN_CATEGORY = EdgeSchema(
    type="IN_CATEGORY",
    from_label="Product",
    to_label="Category",
    properties={}
)

MADE_BY = EdgeSchema(
    type="MADE_BY",
    from_label="Product",
    to_label="Brand",
    properties={}
)

BOUGHT_WITH = EdgeSchema(
    type="BOUGHT_WITH",
    from_label="Product",
    to_label="Product",
    properties={
        "frequency": int,      # how often bought together
        "confidence": float,
    }
)

SIMILAR_TO = EdgeSchema(
    type="SIMILAR_TO",
    from_label="Product",
    to_label="Product",
    properties={
        "shared_features": int,
        "similarity_score": float,
    }
)

FEATURE_FOR_USE_CASE = EdgeSchema(
    type="USEFUL_FOR",
    from_label="Feature",
    to_label="UseCase",
    properties={
        "importance": float,   # how important this feature is for this use case
    }
)


# ============== COMPLETE SCHEMA ==============

PRODUCT_SCHEMA = GraphSchema(
    name="products",
    nodes={
        "Product": PRODUCT_NODE,
        "Feature": FEATURE_NODE,
        "UseCase": USE_CASE_NODE,
        "Category": CATEGORY_NODE,
        "Brand": BRAND_NODE,
    },
    edges={
        "HAS_FEATURE": HAS_FEATURE,
        "FOR_USE_CASE": FOR_USE_CASE,
        "IN_CATEGORY": IN_CATEGORY,
        "MADE_BY": MADE_BY,
        "BOUGHT_WITH": BOUGHT_WITH,
        "SIMILAR_TO": SIMILAR_TO,
        "USEFUL_FOR": FEATURE_FOR_USE_CASE,
    }
)


# ============== FEATURE PATTERNS ==============
# Used for extracting features from product titles/descriptions

FEATURE_PATTERNS = {
    # Audio features
    "wireless": ["wireless", "bluetooth", "cordless", "wire-free"],
    "noise_canceling": ["noise canceling", "noise cancelling", "anc", "active noise"],
    "bass_boost": ["bass boost", "extra bass", "deep bass", "powerful bass"],
    "surround_sound": ["surround sound", "7.1", "5.1", "dolby", "spatial audio"],
    "hi_res_audio": ["hi-res", "high resolution", "lossless", "flac", "aptx hd"],

    # Durability features
    "waterproof": ["waterproof", "water resistant", "ipx7", "ipx8", "swim"],
    "dustproof": ["dustproof", "dust resistant", "ip6"],
    "shockproof": ["shockproof", "shock resistant", "rugged", "military grade"],
    "sweatproof": ["sweatproof", "sweat resistant", "sport", "gym"],

    # Battery/Power
    "long_battery": ["long battery", "all-day", "40 hour", "30 hour", "20 hour", "extended battery"],
    "fast_charging": ["fast charging", "quick charge", "rapid charge", "usb-c charging"],
    "wireless_charging": ["wireless charging", "qi charging", "charging case"],

    # Connectivity
    "multi_device": ["multi-device", "multipoint", "switch between", "connect multiple"],
    "low_latency": ["low latency", "gaming mode", "no lag", "instant"],
    "usb_c": ["usb-c", "usb type-c", "type c"],

    # Comfort
    "lightweight": ["lightweight", "light weight", "ultra-light", "featherweight"],
    "foldable": ["foldable", "folding", "collapsible", "portable"],
    "ergonomic": ["ergonomic", "comfortable fit", "memory foam", "soft cushion"],

    # Smart features
    "voice_assistant": ["alexa", "google assistant", "siri", "voice control"],
    "touch_controls": ["touch control", "touch sensor", "tap control"],
    "app_control": ["app control", "companion app", "customizable eq"],
    "transparency_mode": ["transparency mode", "ambient mode", "hear-through", "awareness mode"],

    # Display (for screens)
    "4k": ["4k", "uhd", "ultra hd", "2160p"],
    "hdr": ["hdr", "hdr10", "dolby vision"],
    "oled": ["oled", "amoled"],
    "high_refresh": ["120hz", "144hz", "240hz", "high refresh"],

    # Camera
    "night_mode": ["night mode", "low light", "night vision"],
    "wide_angle": ["wide angle", "ultra wide", "panorama"],
    "stabilization": ["stabilization", "ois", "eis", "gimbal"],
}

# ============== USE CASE PATTERNS ==============

USE_CASE_PATTERNS = {
    "travel": ["travel", "airplane", "flight", "commute", "portable", "on-the-go"],
    "workout": ["workout", "gym", "fitness", "running", "sports", "exercise", "training"],
    "gaming": ["gaming", "game", "esports", "fps", "low latency"],
    "work_from_home": ["work from home", "office", "conference call", "zoom", "meeting", "wfh"],
    "music_production": ["studio", "production", "mixing", "monitoring", "professional"],
    "podcast": ["podcast", "recording", "streaming", "content creation", "youtube"],
    "outdoor": ["outdoor", "hiking", "camping", "adventure", "rugged"],
    "sleep": ["sleep", "sleeping", "bedtime", "relaxation"],
    "kids": ["kids", "children", "child", "safe volume", "parental"],
    "audiophile": ["audiophile", "hi-fi", "lossless", "high fidelity", "reference"],
}

# ============== PRICE TIERS ==============

PRICE_TIERS = {
    "budget": (0, 50),
    "mid_range": (50, 150),
    "premium": (150, 300),
    "luxury": (300, float('inf')),
}


def get_price_tier(price: float) -> str:
    """Get price tier for a given price."""
    for tier, (low, high) in PRICE_TIERS.items():
        if low <= price < high:
            return tier
    return "luxury"


if __name__ == "__main__":
    print("Product Schema")
    print(f"  Nodes: {list(PRODUCT_SCHEMA.nodes.keys())}")
    print(f"  Edges: {list(PRODUCT_SCHEMA.edges.keys())}")
    print(f"\nFeature Types: {len(FEATURE_PATTERNS)}")
    print(f"Use Cases: {len(USE_CASE_PATTERNS)}")
