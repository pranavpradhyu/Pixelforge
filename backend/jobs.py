"""
Lightweight background jobs.

Restoration can take seconds (classical) to minutes (deep, CPU). We run it off
the request thread and let the frontend poll for progress. This uses a stdlib
ThreadPool so it works anywhere with zero infra. For horizontal scaling, swap
`submit` for an RQ/Celery enqueue — the job function signature is unchanged.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

from config import Config
from backend import storage
from backend.pipeline.orchestrator import run

_executor = ThreadPoolExecutor(max_workers=Config.JOB_WORKERS)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set(job_id, **fields):
    with _lock:
        _jobs.setdefault(job_id, {}).update(fields)


def get(job_id):
    with _lock:
        return dict(_jobs.get(job_id, {})) or None


def submit(job_id: str, input_path, intent: str):
    _set(job_id, id=job_id, status="queued", progress=0, message="Queued")
    _executor.submit(_work, job_id, input_path, intent)
    return job_id


def _work(job_id, input_path, intent):
    def progress(pct, msg):
        _set(job_id, status="running", progress=pct, message=msg)

    try:
        rgb = storage.read_image(input_path)
        result = run(rgb, intent=intent, progress=progress)
        out_path = storage.write_image(result["output"], job_id, "png")
        _set(
            job_id,
            status="done",
            progress=100,
            message="Done",
            output_url=f"/result/{out_path.name}",
            input_url=f"/uploaded/{input_path.name}",
            elapsed_s=result["elapsed_s"],
            analysis=result["analysis"],
            # research-mode payload; the default UI ignores it
            ranking=result["ranking"],
            winner_key=result["winner_key"],
        )
    except Exception as exc:
        _set(job_id, status="error", progress=100,
             message="We couldn't process that image. Try another file.",
             detail=str(exc))
