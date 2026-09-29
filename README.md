# PixelForge - Intelligent Image Restoration Studio

Upload a damaged photo (too dark, blurry, grainy, faded or low-resolution)
and get back the best repaired version.

### Live demo: https://pixelforge-tfz2.onrender.com

> The demo runs on Render's free tier, so the first visit after a period of
> inactivity can take 30-60 seconds to wake up.

**Stack:** Python, Flask, OpenCV, scikit-image, NumPy, vanilla JavaScript
**Deployed on:** Render (free tier, classical OpenCV methods)

PixelForge inspects the photo, runs **many** restoration algorithms in the
backend, scores every result, and returns only the single best version. The
user never sees a model, a metric, or a comparison - just a before/after slider
and a download button.

> Built as an end-to-end computer-vision application: heavy on CV breadth,
> real-world utility, and an automatic *model-selection* mechanism rather than a
> single fixed model.

---

## Why this design

Most "AI photo enhancer" demos pick one model and hope it fits every photo. It
never does — a super-resolver wrecks a noisy scan, a low-light net blows out a
daylight product shot. PixelForge instead treats restoration as a **routing +
selection** problem:

```
        ┌───────────┐   routes    ┌──────────────────────┐
 image →│  analyzer │────────────▶│  candidate restorers │
        └───────────┘             │  (classical + deep)  │
              │                   └──────────┬───────────┘
              │ measurements                 │ N outputs
              ▼                              ▼
        (priors only)              ┌──────────────────────┐
                                   │  NR-IQA scoring +    │
                                   │  fidelity guard      │
                                   └──────────┬───────────┘
                                              │ best that still
                                              ▼ looks like the input
                                        final image → user
```

1. **Analyze** — classical CV measures light, focus, grain, resolution, colour,
   and faces (`backend/pipeline/analyzer.py`). This *routes* compute so only
   relevant repairs run.
2. **Restore** — each applicable branch produces a candidate. Branches include
   classical methods (CLAHE, gray-world WB, non-local-means, Richardson-Lucy
   deblur, unsharp, Lanczos) and pluggable deep models (Real-ESRGAN, GFPGAN,
   Zero-DCE, DnCNN, rembg/U²-Net).
3. **Select** — every candidate is graded by a no-reference image-quality metric
   and checked against the input with an SSIM **fidelity guard** so the pipeline
   can never ship a hallucinated image that no longer resembles the upload
   (`backend/iqa/`, `backend/pipeline/selector.py`).

The comparison is real and it is entirely server-side. The winning model is
logged for research/ops but stripped from the user-facing API response.

## Computer-vision techniques used

| Stage | Classical (always on) | Deep (optional, flag-gated) |
|------|------------------------|------------------------------|
| Analysis | Laplacian blur, Immerkaer noise σ, histogram exposure, Hasler–Süsstrunk colourfulness, Haar face detect | — |
| Low light | CLAHE + gamma, tone S-curve | Zero-DCE |
| Denoise | Non-local means | DnCNN |
| Deblur/sharpen | Richardson–Lucy, unsharp mask | (NAFNet-ready) |
| Super-resolution | Lanczos | Real-ESRGAN x4 |
| Colour | Gray-world white balance | DeOldify-ready |
| Face | — | GFPGAN / CodeFormer-ready |
| Matting | — | rembg (U²-Net / IS-Net) |
| Quality (selection) | Composite NR-IQA | CLIP-IQA / BRISQUE via `pyiqa` |

## Real-world uses

- Reviving old family photographs (fade, grain, scratches, low res).
- E-commerce / marketplace listing cleanup and background cutout.
- Real-estate and rental photo enhancement (dim interiors).
- Archival and library digitisation.
- Insurance / claims and forensic image clarification — where the **fidelity
  guard** matters most.

## Further research scope

- **Learned routing**: replace the hand-tuned `_route()` with a small network
  that predicts the best restorer family from degradation features → a
  restoration mixture-of-experts.
- **Better selection metric**: the composite NR-IQA is a stand-in; swapping in a
  learned perceptual metric (CLIP-IQA, MANIQA, TOPIQ) and validating its
  correlation with human preference is a project in itself.
- **Perception–fidelity trade-off**: tune the guard per domain (aggressive for
  consumer photos, strict for forensic/medical) and study the frontier.
- **Region-wise routing**: run different models on face vs background vs text
  regions and recompose.
- **Edge deployment**: distil the winning ensemble into one fast model.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # base: CPU, no weights needed (OpenCV 4.x)
cp .env.example .env                      # optional, tweak flags
python run.py                             # http://localhost:5000
```

The base install runs the full pipeline with classical restorers and the
composite quality metric — no GPU, no downloads.

### Turn on deep models

```bash
pip install -r requirements-heavy.txt
# download weights into ./weights (see requirements-heavy.txt notes), then:
export ENABLE_REMBG=1 ENABLE_REALESRGAN=1 ENABLE_GFPGAN=1 ENABLE_PYIQA=1
python run.py
```

Each flag is independent and fails safe: if a package or weight is missing, that
branch is simply skipped and the rest of the pipeline still competes.

### Production

```bash
gunicorn 'run:app' --workers 1 --threads 4 --timeout 300
```

Inference runs off the request thread via a thread-pool job queue
(`backend/jobs.py`); the frontend polls `/api/status/<id>`. To scale out, swap
`jobs.submit` for an RQ/Celery enqueue — the job function is unchanged.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/enhance` | multipart `image` + `intent` → `{job_id}` |
| GET | `/api/status/<id>` | progress + result (`?debug=1` adds backend ranking) |
| GET | `/download/<name>` | download the final image |
| GET | `/api/models` | registered restorers + availability |
| GET | `/healthz` | liveness |

## Layout

```
backend/
  pipeline/
    analyzer.py      degradation analysis + routing
    registry.py      maps route keys → restorers
    orchestrator.py  analyze → run candidates → select
    selector.py      NR-IQA ranking + fidelity guard
    models/          classical.py, deep.py, base.py
  iqa/metrics.py     composite + optional learned NR-IQA
  jobs.py            threaded job queue with progress
  storage.py         upload/output IO + retention janitor
  routes.py          Flask API + pages
frontend/            template + darkroom-themed CSS/JS UI
```

## Deployment

The live demo is deployed from this repo to Render using `render.yaml`
(1 gunicorn worker + 4 threads, because job progress is kept in memory).

## Notes on the free tier

Render's 512 MB free instance is tight for deep models and CPU inference is
slow. The app is built to run there on classical restorers alone; enable deep
models on a box with more RAM (and ideally a GPU) or run them as an async
worker.
