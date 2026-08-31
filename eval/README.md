# Model evaluation harness

Decides which apparel / open-vocabulary detection model to bake into
`yolo_service.py` alongside YOLOv8x-COCO. Runs local inference against
real catalog Media pulled from staging Mongo; produces a comparison
table you can read in ~5 minutes.

Not deployed. Not called from production. Once you pick a winner from
the analysis, set `YOLO_APPAREL_MODEL_URL` on the Render dashboard's
build environment and let the yolo-microservice auto-deploy pick it up.

## What it measures

- **`detection_rate`** — % of images where the model returned ≥1
  detection at confidence ≥ 0.30. This is the primary metric.
- **`mean_top_conf`** — how confident the model is on the images it did
  detect. High = the model is sure; low = it's guessing.
- **`median_bbox_frac`** — top bbox area / image area. Product photos
  usually want ~30–80% (the product fills much of the frame). Tiny
  numbers = spurious junk detections. ≥ 0.95 = whole-image fallback.
- **`label_match_rate`** — % where the top class label fuzzy-matches
  the product's category. Best-effort semantic sanity check — 0 means
  the model detected SOMETHING but the label is wrong.
- **`mean_runtime_ms`** — inference wall time. Grounding DINO is 5–10×
  slower than YOLOv8; this is where you decide whether that's worth it.

Also produces a **union coverage** table (what if we ran ALL models?)
and a **marginal contribution** table (which model uniquely catches
things others miss?) — helpful for deciding whether to layer models
or pick one.

## Setup (one time)

```
cd yolo_microservice/eval
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows PowerShell
python -m pip install -r requirements.txt
```

Create `eval/.env` with the staging MongoDB URI:

```
MONGODB_URI=mongodb+srv://...
```

## Run

Step 1 — sample real Media URLs from Mongo:

```
python fetch_media_urls.py \
    --brand-name "Soludos" \
    --brand-name "Pelagic Gear" \
    --brand-name "U Beauty" \
    --brand-name "GymShark" \
    --per-brand 10
```

Only samples Media where `refinedProducts` is empty by default — the
Media that most need a better detector. Pass `--any-media` to sample
without that filter.

Step 2 — run inference (each model gated separately):

```
python run_eval.py --all                  # COCO + Fashionpedia + Grounding DINO
python run_eval.py --coco --fashionpedia  # skip open-vocab
python run_eval.py --coco --grounding-dino --limit 5   # smoke test
```

First run downloads the models (yolov8x.pt ~130MB, Fashionpedia
checkpoint ~10MB, Grounding DINO ~200MB). Cached after that.

Step 3 — read the decision table:

```
python analyze.py
```

## Swapping candidate models

The Fashionpedia model is env/CLI configurable — swap without editing
the code:

```
python run_eval.py --fashionpedia --fashionpedia-model "kesimeg/yolov8n-fashion"
python run_eval.py --fashionpedia --fashionpedia-model "yainage90/fashion-object-detection"
# or point at a local .pt file:
python run_eval.py --fashionpedia --fashionpedia-model "./checkpoints/my-fashion.pt"
```

Same for Grounding DINO:

```
python run_eval.py --grounding-dino --grounding-dino-model "IDEA-Research/grounding-dino-base"
```

## Interpreting the results

Read left to right through the per-(model, brand) table:

- If **COCO** detection_rate is low on a brand → COCO is missing that
  brand's product category. Fashionpedia or Grounding DINO should be
  meaningfully higher.
- If **Fashionpedia** matches COCO's rate → Fashionpedia isn't earning
  its keep on that brand (probably a beauty or non-apparel category).
- If **Grounding DINO** meaningfully beats both → the specific product
  category isn't covered by either pretrained model, and open-vocab is
  the right answer.
- If **union coverage** ≈ best single model → no gain from stacking;
  pick the single best.
- If **union coverage** meaningfully > best single → stack the two that
  contribute most to marginal coverage.

The winning shape you want:
- **Fashion brands**: Fashionpedia dominates.
- **Beauty brands**: Grounding DINO dominates OR COCO's `bottle`/`cup`
  already covers most of it (in which case adding a model isn't worth
  the deploy).
- **Mixed catalogs**: probably Grounding DINO alone wins on flexibility.

## Housekeeping

`data/*.json` and `data/*.csv` are outputs and should not be committed
if they contain any brand-specific product info. Add them to
`.gitignore` if you plan to share the eval branch publicly.
