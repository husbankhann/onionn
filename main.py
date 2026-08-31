"""
Onion Quality Assessment & Grading — Backend API
==================================================
Run with:  python app/main.py
Docs:      see README.md in project root

Solves the stated problem by:
  1. Replacing subjective visual grading with quantitative CV measurements
     (size, shape, defects, color uniformity) -> `grading_engine.py`
  2. Standardizing the SAME config/thresholds across every procurement
     center, versioned so everyone is provably using the same rules
  3. Persisting every image + measurement + config version used, so a
     disputed grade can be independently re-checked instead of argued
     from memory -> `db.py`
  4. Providing a formal dispute workflow (raise / review / resolve)
"""

import os
import io
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image

from grading_engine import OnionGradingEngine, DEFAULT_CONFIG
import db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_from_filestorage(file_storage) -> np.ndarray:
    """Reads an uploaded file into an OpenCV BGR numpy array without
    relying on OpenCV's own (path-based) file IO."""
    pil_img = Image.open(file_storage.stream).convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def ensure_bootstrap():
    """Creates the DB tables and an initial default config on first run."""
    db.init_db()
    if db.get_active_config() is None:
        db.save_config(DEFAULT_CONFIG["config_version"], DEFAULT_CONFIG, activate=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Procurement centers
# ---------------------------------------------------------------------------
@app.route("/api/centers", methods=["POST"])
def create_center():
    data = request.get_json(force=True)
    name = data.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400
    location = data.get("location")
    pixels_per_mm = data.get("pixels_per_mm", 8.0)
    try:
        center_id = db.create_center(name, location, pixels_per_mm)
    except Exception as e:
        return jsonify({"error": f"could not create center: {e}"}), 400
    return jsonify(db.get_center(center_id)), 201


@app.route("/api/centers", methods=["GET"])
def get_centers():
    return jsonify(db.list_centers())


@app.route("/api/centers/<int:center_id>", methods=["GET"])
def get_center(center_id):
    center = db.get_center(center_id)
    if not center:
        return jsonify({"error": "not found"}), 404
    return jsonify(center)


# ---------------------------------------------------------------------------
# Grading configuration (shared/standardized across all centers)
# ---------------------------------------------------------------------------
@app.route("/api/config", methods=["GET"])
def get_config():
    cfg = db.get_active_config()
    return jsonify(cfg)


@app.route("/api/config", methods=["POST"])
def set_config():
    """
    Push a new standardized grading config. This is how a central quality
    authority keeps every procurement center grading against IDENTICAL
    thresholds -- the root fix for cross-center inconsistency.
    """
    data = request.get_json(force=True)
    version = data.get("version")
    config_dict = data.get("config")
    if not version or not config_dict:
        return jsonify({"error": "version and config are required"}), 400
    config_id = db.save_config(version, config_dict, activate=True)
    return jsonify(db.get_config_by_id(config_id)), 201


# ---------------------------------------------------------------------------
# Lots (a lot = one farmer's/truck's batch of onions delivered to a center)
# ---------------------------------------------------------------------------
@app.route("/api/lots", methods=["POST"])
def create_lot():
    data = request.get_json(force=True)
    lot_code = data.get("lot_code") or f"LOT-{uuid.uuid4().hex[:8].upper()}"
    center_id = data.get("center_id")
    if not center_id:
        return jsonify({"error": "center_id is required"}), 400
    if not db.get_center(center_id):
        return jsonify({"error": "center_id does not exist"}), 404
    lot_id = db.create_lot(
        lot_code=lot_code,
        center_id=center_id,
        farmer_name=data.get("farmer_name"),
        variety=data.get("variety"),
        target_size_band=data.get("target_size_band"),
    )
    return jsonify(db.get_lot(lot_id)), 201


@app.route("/api/lots", methods=["GET"])
def get_lots():
    center_id = request.args.get("center_id", type=int)
    return jsonify(db.list_lots(center_id))


@app.route("/api/lots/<int:lot_id>", methods=["GET"])
def get_lot(lot_id):
    lot = db.get_lot(lot_id)
    if not lot:
        return jsonify({"error": "not found"}), 404
    return jsonify(lot)


# ---------------------------------------------------------------------------
# Core grading endpoint
# ---------------------------------------------------------------------------
@app.route("/api/lots/<int:lot_id>/assess", methods=["POST"])
def assess_lot(lot_id):
    """
    Upload an onion image for a given lot. Runs the standardized CV
    grading pipeline and stores a full audit record (image + measurements
    + config version) tied to this lot and procurement center.
    """
    lot = db.get_lot(lot_id)
    if not lot:
        return jsonify({"error": "lot not found"}), 404

    if "image" not in request.files:
        return jsonify({"error": "multipart field 'image' is required"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "invalid or missing image file"}), 400

    center = db.get_center(lot["center_id"])
    active_config = db.get_active_config()
    if not active_config:
        return jsonify({"error": "no active grading config found"}), 500

    try:
        image_bgr = load_image_from_filestorage(file)
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400

    # Persist the raw image for later human review / dispute resolution
    ext = file.filename.rsplit(".", 1)[1].lower()
    stored_filename = f"lot{lot_id}_{uuid.uuid4().hex[:10]}.{ext}"
    save_path = os.path.join(UPLOAD_DIR, stored_filename)
    cv2.imwrite(save_path, image_bgr)

    engine = OnionGradingEngine(config=active_config["config"])
    result = engine.grade_image(
        image_bgr,
        target_size_band=lot.get("target_size_band"),
        pixels_per_mm=center.get("pixels_per_mm") if center else None,
    )

    inspector_name = request.form.get("inspector_name")
    assessment_id = db.save_assessment(
        lot_id=lot_id,
        image_filename=stored_filename,
        config_id=active_config["id"],
        result_dict=result,
        inspector_name=inspector_name,
    )

    response = db.get_assessment(assessment_id)
    return jsonify(response), 201


@app.route("/api/assessments", methods=["GET"])
def get_assessments():
    lot_id = request.args.get("lot_id", type=int)
    center_id = request.args.get("center_id", type=int)
    return jsonify(db.list_assessments(lot_id=lot_id, center_id=center_id))


@app.route("/api/assessments/<int:assessment_id>", methods=["GET"])
def get_assessment(assessment_id):
    result = db.get_assessment(assessment_id)
    if not result:
        return jsonify({"error": "not found"}), 404
    return jsonify(result)


@app.route("/api/assessments/<int:assessment_id>/image", methods=["GET"])
def get_assessment_image(assessment_id):
    assessment = db.get_assessment(assessment_id)
    if not assessment:
        return jsonify({"error": "not found"}), 404
    return send_from_directory(UPLOAD_DIR, assessment["image_filename"])


# ---------------------------------------------------------------------------
# Disputes -- direct fix for "resulting in disputes and inconsistencies"
# ---------------------------------------------------------------------------
@app.route("/api/disputes", methods=["POST"])
def raise_dispute():
    data = request.get_json(force=True)
    assessment_id = data.get("assessment_id")
    reason = data.get("reason")
    raised_by = data.get("raised_by")
    if not assessment_id or not reason:
        return jsonify({"error": "assessment_id and reason are required"}), 400
    if not db.get_assessment(assessment_id):
        return jsonify({"error": "assessment not found"}), 404
    dispute_id = db.create_dispute(assessment_id, raised_by, reason)
    return jsonify(db.get_dispute(dispute_id)), 201


@app.route("/api/disputes", methods=["GET"])
def get_disputes():
    status = request.args.get("status")
    return jsonify(db.list_disputes(status))


@app.route("/api/disputes/<int:dispute_id>/resolve", methods=["POST"])
def resolve_dispute(dispute_id):
    data = request.get_json(force=True)
    resolution_notes = data.get("resolution_notes", "")
    status = data.get("status", "resolved")
    if not db.get_dispute(dispute_id):
        return jsonify({"error": "not found"}), 404
    db.resolve_dispute(dispute_id, resolution_notes, status)
    return jsonify(db.get_dispute(dispute_id))


# ---------------------------------------------------------------------------
# Cross-center consistency report -- lets a quality authority see whether
# centers are grading similarly-scored lots the same way.
# ---------------------------------------------------------------------------
@app.route("/api/reports/center-comparison", methods=["GET"])
def center_comparison():
    centers = db.list_centers()
    report = []
    for c in centers:
        assessments = db.list_assessments(center_id=c["id"])
        if not assessments:
            report.append({"center": c["name"], "assessment_count": 0})
            continue
        scores = [a["average_composite_score"] for a in assessments if a["average_composite_score"] is not None]
        grade_dist = {}
        for a in assessments:
            g = a["lot_grade"]
            grade_dist[g] = grade_dist.get(g, 0) + 1
        report.append({
            "center": c["name"],
            "assessment_count": len(assessments),
            "average_score": round(sum(scores) / len(scores), 2) if scores else None,
            "grade_distribution": grade_dist,
        })
    return jsonify(report)


if __name__ == "__main__":
    ensure_bootstrap()
    app.run(host="0.0.0.0", port=5000, debug=True)
