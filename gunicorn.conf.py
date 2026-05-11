# Gunicorn config — all tunables read from env so Render dashboard
# tweaks don't require a Docker rebuild.
#
# Env vars (defaults match what's safe on Render Standard 2GB):
#   GUNICORN_WORKERS               1     YOLOv8x holds ~500MB; multiple workers OOM
#   GUNICORN_THREADS               1     PyTorch isn't reliably thread-safe
#   GUNICORN_TIMEOUT               300   long enough for video frame extraction
#   GUNICORN_MAX_REQUESTS          0     0 = disabled; set to ~100 to recycle workers
#                                        and combat PyTorch memory creep
#   GUNICORN_MAX_REQUESTS_JITTER   0     randomize restart timing (pair with MAX_REQUESTS)
#   GUNICORN_PRELOAD_APP           false load app + model BEFORE forking workers — faster
#                                        cold start, shared model across workers (COW),
#                                        but reloads of the model in workers can race.

import os

def _int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')

bind                = '0.0.0.0:5000'
workers             = _int('GUNICORN_WORKERS', 1)
threads             = _int('GUNICORN_THREADS', 1)
timeout             = _int('GUNICORN_TIMEOUT', 300)
max_requests        = _int('GUNICORN_MAX_REQUESTS', 0)
max_requests_jitter = _int('GUNICORN_MAX_REQUESTS_JITTER', 0)
preload_app         = _bool('GUNICORN_PRELOAD_APP', False)

# Route gunicorn's own logs to stdout so Render's log stream picks them up
# alongside the app's print() output.
accesslog = '-'
errorlog  = '-'
loglevel  = 'info'
