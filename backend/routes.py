"""HTTP routes: the page, the upload/enhance API, status polling, downloads."""
from flask import (Blueprint, render_template, request, jsonify,
                   send_from_directory, abort, current_app)

from config import Config
from backend import storage, jobs
from backend.pipeline.registry import registry

bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.post("/api/enhance")
def enhance():
    if "image" not in request.files:
        return jsonify(error="No file received. Choose a photo and try again."), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify(error="No file selected."), 400
    if not storage.ext_ok(f.filename):
        return jsonify(error="Unsupported format. Use JPG, PNG, WEBP, BMP or TIFF."), 415

    intent = request.form.get("intent", "auto")
    job_id, path = storage.save_upload(f)
    jobs.submit(job_id, path, intent)
    return jsonify(job_id=job_id), 202


@bp.get("/api/status/<job_id>")
def status(job_id):
    state = jobs.get(job_id)
    if state is None:
        return jsonify(error="Unknown job"), 404
    # Trim research payload from the default response; expose via ?debug=1
    if request.args.get("debug") != "1":
        state = {k: v for k, v in state.items() if k not in {"ranking", "winner_key"}}
    return jsonify(state)


@bp.get("/uploaded/<name>")
def uploaded(name):
    return send_from_directory(Config.UPLOAD_DIR, name)


@bp.get("/result/<name>")
def result(name):
    return send_from_directory(Config.OUTPUT_DIR, name, as_attachment=False)


@bp.get("/download/<name>")
def download(name):
    return send_from_directory(Config.OUTPUT_DIR, name, as_attachment=True,
                               download_name=f"pixelforge-{name}")


@bp.get("/api/models")
def models():
    """Availability of every registered restorer — for ops / research mode."""
    return jsonify(models=registry.catalogue())


@bp.get("/healthz")
def healthz():
    return jsonify(status="ok")


@bp.errorhandler(413)
def too_large(_):
    mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify(error=f"That file is over the {mb} MB limit."), 413
