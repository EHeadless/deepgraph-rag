# Product Navigator

Graph RAG for product recommendations. Find products by Feature x UseCase x Price.

## Run

```bash
streamlit run examples/products/app.py --server.port 8506
```

## The Value

| Query | Vector Search | Graph RAG |
|-------|---------------|-----------|
| "Wireless headphones" | Works | Works |
| "Wireless headphones FOR running under $100" | Fails | Works |
| "What goes with this laptop?" | Impossible | Works |

## Tabs

1. **How It Works** - Parallel to research pattern
2. **Feature x UseCase Filter** - The killer query
3. **Bundle Builder** - Complementary products via co-purchase
4. **Niche Finder** - Rare feature + use case combos
5. **Feature Migration** - Premium features in budget products
6. **Graph Explorer** - Interactive pyvis visualization

## Graph Schema

```
Product
  ├── HAS_FEATURE ──→ Feature (wireless, waterproof, 4k...)
  ├── FOR_USE_CASE ──→ UseCase (travel, workout, gaming...)
  ├── IN_CATEGORY ──→ Category (Headphones, Laptop, Camera...)
  ├── MADE_BY ──→ Brand
  └── BOUGHT_WITH ──→ Product
```

## Data Pipeline

```bash
python examples/products/scripts/build_product_graph.py
```

## Honest Assessment

High differentiation. Clear product-market fit. People actually think in terms of Feature + UseCase + Price when shopping. E-commerce filtering is a real pain point.
