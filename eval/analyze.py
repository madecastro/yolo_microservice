"""Read data/eval_results.csv and produce a decision table.

Metrics per (model, brand):
  n                 — sample size
  detection_rate    — % of images where the model returned ≥1 confident detection
  mean_top_conf     — mean of top-1 confidence across the sample
  median_bbox_frac  — median of top-1 bbox area / image area
                      (a good product detection covers 30-80% of the frame;
                       tiny detections are usually junk, whole-image is a fallback)
  label_match_rate  — % of images where the top class label fuzzy-matches
                      CatalogProduct.category (best-effort semantic check)
  mean_runtime_ms   — mean per-image inference wall time

Read this table to answer:
  1. Which model has the highest detection_rate per brand?
  2. Is the top class label semantically right (label_match_rate) or is
     the model just detecting SOMETHING generic?
  3. Is the latency delta worth the recall gain?

Usage:
    python analyze.py
    python analyze.py --csv data/eval_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(HERE / "data" / "eval_results.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("no rows in CSV")
        return

    # A "detection" counts when detection_count >= 1 AND top_confidence >= 0.30.
    # Ultralytics's own threshold is 0.2 but consumers of Media.refinedProducts[]
    # expect meaningful boxes — 0.30 is where the noise falls off measurably.
    df["is_detected"] = ((df["detection_count"] >= 1) & (df["top_confidence"] >= 0.30)).astype(int)

    print(f"\nLoaded {len(df)} rows across {df['brand_name'].nunique()} brands × {df['model'].nunique()} models\n")

    # ── Per (model, brand) summary ──
    grouped = df.groupby(["model", "brand_name"], sort=False)
    summary = grouped.agg(
        n=("media_id", "count"),
        detection_rate=("is_detected", lambda x: round(100 * x.mean(), 1)),
        mean_top_conf=("top_confidence", lambda x: round(x.mean(), 3)),
        median_bbox_frac=("top_bbox_area_frac", lambda x: round(x.median(), 3)),
        label_match_rate=("top_class_matches_cat", lambda x: round(100 * x.mean(), 1)),
        mean_runtime_ms=("runtime_ms", lambda x: int(x.mean())),
    ).reset_index()

    # Format nicely.
    print("── PER (MODEL, BRAND) — the primary decision table ──\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print(summary.to_string(index=False))

    # ── Per model overall (aggregated across brands) ──
    print("\n── PER MODEL — averages across all brands ──\n")
    overall = df.groupby("model", sort=False).agg(
        n=("media_id", "count"),
        detection_rate=("is_detected", lambda x: round(100 * x.mean(), 1)),
        mean_top_conf=("top_confidence", lambda x: round(x.mean(), 3)),
        median_bbox_frac=("top_bbox_area_frac", lambda x: round(x.median(), 3)),
        label_match_rate=("top_class_matches_cat", lambda x: round(100 * x.mean(), 1)),
        mean_runtime_ms=("runtime_ms", lambda x: int(x.mean())),
    ).reset_index()
    print(overall.to_string(index=False))

    # ── Union coverage (would combining models help?) ──
    #
    # Pivot: for each media_id, was it detected by each model? Then measure
    # "any model detected it" per brand — the ceiling if we combined all
    # models. This tells us whether adding a second model is redundant
    # (already covered by baseline) or genuinely additive.
    print("\n── UNION COVERAGE (ceiling: what if we ran ALL enabled models?) ──\n")
    pivot = df.pivot_table(
        index=["media_id", "brand_name"],
        columns="model",
        values="is_detected",
        fill_value=0,
    ).reset_index()
    model_cols = [c for c in pivot.columns if c not in ("media_id", "brand_name")]
    pivot["any_model"] = pivot[model_cols].max(axis=1)
    union = pivot.groupby("brand_name")["any_model"].agg(
        n="count", detection_rate=lambda x: round(100 * x.mean(), 1)
    ).reset_index()
    print(union.to_string(index=False))

    # ── Marginal contribution of each model ──
    #
    # For each model, how many Media did IT detect that no OTHER model
    # detected? This is where you see whether Grounding DINO earns its
    # keep or whether Fashionpedia is redundant with COCO on your catalog.
    if len(model_cols) > 1:
        print("\n── MARGINAL COVERAGE (unique detections per model — no other model saw it) ──\n")
        marginal_rows = []
        for m in model_cols:
            others = [c for c in model_cols if c != m]
            unique_hits = ((pivot[m] == 1) & (pivot[others].max(axis=1) == 0)).groupby(pivot["brand_name"]).sum()
            for brand, count in unique_hits.items():
                marginal_rows.append({"model": m, "brand_name": brand, "unique_detections": int(count)})
        marginal = pd.DataFrame(marginal_rows).pivot(
            index="brand_name", columns="model", values="unique_detections"
        ).fillna(0).astype(int)
        print(marginal.to_string())


if __name__ == "__main__":
    main()
