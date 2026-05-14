# app/routes/page_routes.py
from flask import Blueprint, render_template
from app.auth import login_required   # changed

page_bp = Blueprint("page_bp", __name__)

@page_bp.route("/")
def home():
    return render_template("index.html")

@page_bp.route("/verify")
def verify_page():
    return render_template("verify.html")

@page_bp.route("/enroll")
def enroll_page():
    return render_template("enroll.html")

@page_bp.route("/challenge")
def challenge_page():
    return render_template("active_challenge.html")

@page_bp.route("/dashboard")
@login_required                       # changed
def dashboard_page():
    return render_template("dashboard.html")