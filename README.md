# Onion Quality Assessment & Grading — Backend

Replaces subjective, human-eyeball onion grading with a computer-vision
pipeline that produces the same measurable grade for the same onions,
no matter which procurement center runs it — and keeps a full audit
trail so disagreements can be resolved by re-checking evidence instead
of arguing.

## The problem this solves

> Quality assessment and grading of onions are often subjective and vary
> across procurement centers, resulting in disputes and inconsistencies.

Concretely:
- **Subjectivity** → replaced with quantitative measurements: size (mm),
  shape/roundness, defect area (rot/sprouting/mold), and skin color
  uniformity.
- **Cross-center inconsistency** → every center runs against the *same*,
  versioned grading config (thresholds/weights) pulled from one source
  of truth (`/api/config`), instead of each inspector's personal judgment.
- **Disputes** → every assessment stores the original image, every
  sub-score, and exactly which config version produced the grade, so a
  dispute can be independently re-examined (`/api/disputes`).

## Architecture

```
grading_engine.py
db.py
main.py
uploads/
requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py             # starts on http://0.0.0.0:5000
```

The database (`onion_grading.db`) and an initial default grading config
are created automatically on first run.

## How grading works

1. **Segmentation** — separates onion(s) from the tray/background using
   adaptive HSV thresholding (works across brown/red/yellow/white varieties
   and different lighting, since thresholds adapt per-image).
2. **Size** — equivalent diameter computed from contour area, converted
   from pixels to millimetres using each center's calibrated
   `pixels_per_mm` (set once per camera rig when the center is created).
3. **Shape** — roundness (`4πA/P²`) and eccentricity from an ellipse fit;
   penalizes split, mishapen, or damaged onions.
4. **Defects** — dark-spot ratio (rot/bruising), green-hue ratio
   (sprouting), and localized saturation collapse (mold/dry patches),
   each measured *relative to that onion's own median brightness/
   saturation* so it isn't thrown off by ordinary lighting variation.
5. **Color uniformity** — standard deviation of hue/saturation across the
   onion's own surface; blotchy discoloration lowers the score.
6. **Composite score** — a weighted sum (defect 40%, shape 20%, color 20%,
   size conformity 20% by default — tunable via `/api/config`).
7. **Grade** — composite score mapped to A / B / C / Reject via
   configurable thresholds.

Every threshold lives in one JSON config (see `grading_engine.DEFAULT_CONFIG`)
that can be updated centrally and is versioned in the database, so a
grade can always be traced to the exact rules in force when it was made.

## API reference

### Health
`GET /api/health`

### Procurement centers
- `POST /api/centers` — `{name, location, pixels_per_mm}`
- `GET /api/centers`
- `GET /api/centers/<id>`

### Grading configuration (shared across all centers)
- `GET /api/config` — currently active config
- `POST /api/config` — `{version, config}` — push new standardized thresholds

### Lots (a delivered batch of onions)
- `POST /api/lots` — `{center_id, lot_code?, farmer_name?, variety?, target_size_band?}`
- `GET /api/lots?center_id=`
- `GET /api/lots/<id>`

### Assessment (the core grading action)
- `POST /api/lots/<lot_id>/assess` — multipart form:
  - `image` (file, required)
  - `inspector_name` (optional)
  - Returns full measurement breakdown per onion + lot-level grade.
- `GET /api/assessments?lot_id=&center_id=`
- `GET /api/assessments/<id>` — full stored result, including raw measurements
- `GET /api/assessments/<id>/image` — the original uploaded image

### Disputes
- `POST /api/disputes` — `{assessment_id, raised_by, reason}`
- `GET /api/disputes?status=open|resolved`
- `POST /api/disputes/<id>/resolve` — `{resolution_notes, status}`

### Reporting
- `GET /api/reports/center-comparison` — average score & grade
  distribution per center, to spot centers drifting from the standard.

## Example: assess a lot

```bash
curl -X POST http://localhost:5000/api/lots \
  -H "Content-Type: application/json" \
  -d '{"center_id":1,"farmer_name":"Ram Singh","variety":"Nashik Red","target_size_band":"medium"}'

curl -X POST http://localhost:5000/api/lots/1/assess \
  -F "image=@onions.jpg" \
  -F "inspector_name=Inspector A"
```

## Calibration note

`pixels_per_mm` must be set per procurement center to match its camera
rig (fixed camera height + focal length), OR you can extend
`grading_engine.py` to detect a known reference object (e.g. a coin or
checkerboard marker) in each photo and compute calibration per-image —
recommended if cameras/phones aren't fixed-mounted, since a wrong
calibration silently skews every size-based score.

## Production hardening checklist (not included, by design, to keep this
runnable as-is)

- Swap SQLite for Postgres (`db.py`'s connection function is the only
  place to change)
- Add authentication (JWT) so only registered inspectors can submit
  assessments and only quality-authority roles can change `/api/config`
- Serve behind gunicorn/uwsgi, not Flask's dev server
- Add object storage (S3-compatible) instead of local `uploads/` for images
- Consider a trained defect-classification CNN once labeled data
  accumulates — the current color/texture heuristics are transparent and
  auditable but won't catch every defect type (e.g. internal rot not
  visible externally)
