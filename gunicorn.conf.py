# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 4
timeout = 120
loglevel = "info"
accesslog = "-"
errorlog = "-"
capture_output = True

# IMPROVED: worker type for better performance
worker_class = "sync"  # Or "gevent" if async needed