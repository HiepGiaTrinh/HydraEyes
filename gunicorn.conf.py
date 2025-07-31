# Gunicorn configuration file for H2 Factory Camera Monitoring System

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# Restart workers after this many requests, to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging - Use local directory instead of system directories
accesslog = "./logs/gunicorn_access.log"
errorlog = "./logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "h2_factory_camera_monitoring"

# Daemon mode
daemon = False
pidfile = "./logs/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment and configure if needed)
# keyfile = "/path/to/private.key"
# certfile = "/path/to/certificate.crt"

# Preload app for better memory usage
preload_app = True

# Worker timeout
graceful_timeout = 30
