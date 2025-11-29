# Gunicorn configuration optimized for memory efficiency
import multiprocessing

# Worker configuration - Use threads instead of multiple processes
workers = 1  # Single worker to avoid loading models multiple times
worker_class = "sync"  # Sync is better for CPU-bound model inference
threads = 1  # Single thread per worker
timeout = 300  # 5 minutes for model loading
keepalive = 5

# Binding
bind = "0.0.0.0:8000"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "chest-xray-api"

# Preload app to save memory
preload_app = False  # Set to False for lazy loading
max_requests = 100  # Restart worker after 100 requests to clear memory
max_requests_jitter = 10

# Worker boot timeout
worker_boot_timeout = 300

