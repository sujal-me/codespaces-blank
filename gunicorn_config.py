# Gunicorn configuration optimized for memory efficiency
import multiprocessing

# Worker configuration
workers = 1  # Single worker to avoid loading models multiple times
worker_class = "sync"
threads = 2
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
max_requests = 1000
max_requests_jitter = 50

# Worker boot timeout
worker_boot_timeout = 300
