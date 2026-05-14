# app/auth.py – Custom Basic Authentication (no external dependencies)

from functools import wraps
from flask import request, Response
from config import Config

def check_auth(username, password):
    """Verify username and password against config."""
    return username == Config.DASHBOARD_USERNAME and password == Config.DASHBOARD_PASSWORD

def authenticate():
    """Send a 401 response that enables basic auth."""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials',
        401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def login_required(f):
    """Decorator to protect routes with HTTP Basic Auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated