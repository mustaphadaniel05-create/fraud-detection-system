# app/routes/page_routes.py
from flask import Blueprint, render_template

page_bp = Blueprint("page_bp", __name__)


@page_bp.route("/")
def home():
    """
    System homepage (new landing page)
    """
    return render_template("index.html")


@page_bp.route("/verify")
def verify_page():
    """
    Main verification page
    """
    return render_template("verify.html")


@page_bp.route("/enroll")
def enroll_page():
    """
    User enrollment page
    """
    return render_template("enroll.html")


@page_bp.route("/challenge")
def challenge_page():
    """
    Active liveness challenge page
    """
    return render_template("active_challenge.html")


@page_bp.route("/dashboard")
def dashboard_page():
    """
    System monitoring dashboard
    """
    return render_template("dashboard.html")