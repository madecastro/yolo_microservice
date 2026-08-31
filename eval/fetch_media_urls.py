"""Pull real catalog Media URLs + CatalogProduct data for the eval.

Reads MONGODB_URI from env (or `.env` in this directory), samples N Media
per brand, joins with CatalogProduct for title/brand/category context,
and writes to data/media_urls.json.

The eval script (run_eval.py) reads that JSON so we don't hit Mongo on
every model run.

Usage:
    python fetch_media_urls.py \\
        --brand-name "Soludos" --brand-name "Pelagic Gear" \\
        --brand-name "U Beauty" --brand-name "GymShark" \\
        --per-brand 10

Only samples Media where refinedProducts is empty — those are the ones
we most want to understand recall on (catalog Media the current
YOLOv8x-COCO couldn't detect).

If --any-media is set instead of the default, samples ALL Media
regardless of refinedProducts state — useful for comparing recall on
already-detected Media.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=False)

def get_brand_id(db, name: str):
    """Case-insensitive brand lookup."""
    doc = db.brands.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}}, {"_id": 1, "name": 1})
    if not doc:
        # Try loose match — some brand names have suffixes like "Gear", "GS", etc.
        doc = db.brands.find_one({"name": {"$regex": name, "$options": "i"}}, {"_id": 1, "name": 1})
    return doc


def sample_media_for_brand(db, brand_doc, per_brand: int, only_empty: bool):
    """Return list of dicts with the fields the eval script needs."""
    brand_id = brand_doc["_id"]
    q = {"brandId": brand_id, "source": "catalog-product"}
    if only_empty:
        q["$or"] = [
            {"refinedProducts": {"$exists": False}},
            {"refinedProducts": {"$size": 0}},
        ]
    media_docs = list(db.media.find(q, {
        "_id": 1, "fileUrl": 1, "width": 1, "height": 1,
        "refinedProducts": 1, "metadata": 1
    }))
    if not media_docs:
        return []
    random.shuffle(media_docs)
    sampled = media_docs[:per_brand]

    # Batch-fetch CatalogProducts for title/brand/category context.
    product_ids = list({
        m.get("metadata", {}).get("catalogProductId")
        for m in sampled
        if m.get("metadata", {}).get("catalogProductId")
    })
    products = {
        p["_id"]: p for p in db.catalogproducts.find(
            {"_id": {"$in": product_ids}},
            {"_id": 1, "title": 1, "brand": 1, "category": 1}
        )
    }

    out = []
    for m in sampled:
        pid = m.get("metadata", {}).get("catalogProductId")
        p = products.get(pid) if pid else None
        out.append({
            "media_id":     str(m["_id"]),
            "brand_id":     str(brand_id),
            "brand_name":   brand_doc["name"],
            "product_id":   str(pid) if pid else None,
            "product_title": (p or {}).get("title"),
            "product_brand": (p or {}).get("brand"),
            "product_category": (p or {}).get("category"),
            "file_url":     m.get("fileUrl"),
            "width":        m.get("width"),
            "height":       m.get("height"),
            "has_refined":  bool(m.get("refinedProducts") and len(m["refinedProducts"]) > 0),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand-name", action="append", default=[],
                    help="Brand name (repeatable). Case-insensitive.")
    ap.add_argument("--per-brand", type=int, default=10,
                    help="Sample size per brand (default 10).")
    ap.add_argument("--any-media", action="store_true",
                    help="Include Media that already have refinedProducts (default: only-empty).")
    ap.add_argument("--out", type=str, default=str(HERE / "data" / "media_urls.json"),
                    help="Output JSON path.")
    args = ap.parse_args()

    if not args.brand_name:
        print("ERROR: pass at least one --brand-name", file=sys.stderr)
        sys.exit(1)

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("ERROR: MONGODB_URI not set (in env or eval/.env)", file=sys.stderr)
        sys.exit(1)

    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    # Mongoose default: db name derived from URI. Let it resolve naturally.
    db = client.get_default_database()
    if db is None:
        # Fallback to the well-known name.
        db = client["liquidRetail"]

    all_rows = []
    for name in args.brand_name:
        brand = get_brand_id(db, name)
        if not brand:
            print(f"⚠️  brand not found: {name}", file=sys.stderr)
            continue
        rows = sample_media_for_brand(db, brand, args.per_brand, only_empty=not args.any_media)
        print(f"📦 {name} → {len(rows)} media (only_empty={not args.any_media})")
        all_rows.extend(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"✓  wrote {len(all_rows)} rows to {args.out}")
    client.close()


if __name__ == "__main__":
    main()
