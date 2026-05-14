from flask_httpauth import HTTPBasicAuth
from config import Config

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    return username == Config.DASHBOARD_USERNAME and password == Config.DASHBOARD_PASSWORD