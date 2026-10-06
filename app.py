from flask import Flask, render_template_string, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_model.joblib")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.csv")

# Predefined demo profiles for quick testing
DEMO_STUDENTS = {
    "topper": {
        "student_name": "Aisha Khan",
        "age": 19,
        "gender": "Female",
        "study_hours_weekly": 24,
        "attendance_rate": 96,
        "previous_marks": 90,
        "assignment_marks": 92,
        "midterm_marks": 91,
        "absences": 1,
        "failures": 0,
        "parental_education": "Bachelor's",
        "tutoring": "Yes",
        "extracurricular": "Yes",
        "motivation_level": "High",
        "sleep_hours": 7.5,
        "internet_access": "Yes"
    },
    "average": {
        "student_name": "Lucas Silva",
        "age": 18,
        "gender": "Male",
        "study_hours_weekly": 14,
        "attendance_rate": 82,
        "previous_marks": 70,
        "assignment_marks": 74,
        "midterm_marks": 72,
        "absences": 5,
        "failures": 0,
        "parental_education": "Some College",
        "tutoring": "No",
        "extracurricular": "Yes",
        "motivation_level": "Medium",
        "sleep_hours": 7,
        "internet_access": "Yes"
    },
    "at_risk": {
        "student_name": "Sophia Miller",
        "age": 19,
        "gender": "Female",
        "study_hours_weekly": 5,
        "attendance_rate": 58,
        "previous_marks": 45,
        "assignment_marks": 48,
        "midterm_marks": 44,
        "absences": 14,
        "failures": 2,
        "parental_education": "High School",
        "tutoring": "No",
        "extracurricular": "No",
        "motivation_level": "Low",
        "sleep_hours": 5.0,
        "internet_access": "Yes"
    }
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Prediction - College ML Project</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-1: #f0f7ff;
            --bg-2: #fff7ed;
            --card-bg: rgba(255, 255, 255, 0.92);
            --primary: #4f46e5;
            --primary-2: #7c3aed;
            --secondary: #06b6d4;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --deep: #0f172a;
            --text: #1f2937;
            --muted: #64748b;
            --border: rgba(148, 163, 184, 0.35);
            --shadow: 0 20px 50px rgba(79, 70, 229, 0.18);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(79, 70, 229, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(6, 182, 212, 0.18), transparent 28%),
                linear-gradient(135deg, var(--bg-1) 0%, #f7faff 32%, var(--bg-2) 100%);
        }

        .hero-banner {
            background: linear-gradient(135deg, #101827 0%, #312e81 28%, #7c3aed 58%, #0ea5e9 100%);
            color: white;
            padding: 2.1rem 1rem 1.9rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 18px 40px rgba(79, 70, 229, 0.28);
        }

        .hero-banner::before,
        .hero-banner::after {
            content: "";
            position: absolute;
            border-radius: 50%;
            background: rgba(255,255,255,0.09);
        }

        .hero-banner::before {
            width: 260px; height: 260px; top: -100px; right: -30px;
        }

        .hero-banner::after {
            width: 220px; height: 220px; bottom: -120px; left: -30px;
        }

        .hero-banner .container {
            position: relative;
            z-index: 1;
        }

        .hero-banner h1 {
            margin: 0 0 0.7rem;
            font-size: clamp(2.1rem, 4vw, 3.2rem);
            font-weight: 800;
            letter-spacing: -0.05em;
        }

        .hero-banner .badge {
            padding: 0.8rem 1.2rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            color: white;
            border: 1px solid rgba(255,255,255,0.18);
            backdrop-filter: blur(8px);
            font-size: 0.8rem;
            letter-spacing: 0.04em;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.18);
        }

        .container {
            max-width: 1200px;
        }

        main.container {
            padding-top: 0.5rem;
        }

        .card-custom {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }

        .card-custom:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 56px rgba(79, 70, 229, 0.2);
        }

        .form-control,
        .form-select {
            min-height: 48px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            background: #ffffff;
            padding: 0.7rem 0.9rem;
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 500;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
            box-shadow: none;
        }

        #predictForm .row.g-3 > [class*="col-"]:not(.col-12) {
            padding: 0.8rem;
            border: 1px solid #e5eaf1;
            border-radius: 14px;
            background: #ffffff;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }

        #predictForm .row.g-3 > [class*="col-"]:not(.col-12):hover {
            border-color: #cbd5e1;
        }

        #predictForm .row.g-3 > [class*="col-"]:not(.col-12):focus-within {
            border-color: rgba(79, 70, 229, 0.45);
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.06);
        }

        .form-control:focus,
        .form-select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
            background: #ffffff;
            outline: none;
        }

        .form-label {
            display: block;
            font-size: 0.84rem;
            font-weight: 700;
            color: #334155;
            letter-spacing: 0;
            text-transform: none;
            margin: 0 0 0.5rem 0.1rem;
        }

        .btn-group .btn {
            border-radius: 10px !important;
            margin: 0;
            font-weight: 700;
            padding: 0.6rem 0.8rem;
        }

        .parameter-header {
            display: block;
        }

        .profile-buttons {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            width: 100%;
        }

        .profile-buttons .btn {
            width: 100%;
            min-height: 46px;
            margin: 0;
            padding: 0.6rem 0.7rem;
            line-height: 1.25;
            white-space: normal;
            color: #ffffff;
            border: 1px solid transparent;
            box-shadow: 0 5px 12px rgba(15, 118, 110, 0.14);
            transition: filter 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }

        .profile-buttons .btn:nth-child(1) {
            background: linear-gradient(135deg, #15803d 0%, #22a06b 100%);
        }

        .profile-buttons .btn:nth-child(2) {
            background: linear-gradient(135deg, #0f766e 0%, #14a38b 100%);
        }

        .profile-buttons .btn:nth-child(3) {
            background: linear-gradient(135deg, #0e7490 0%, #14b8a6 100%);
        }

        .profile-buttons .btn:hover,
        .profile-buttons .btn:focus-visible {
            color: #ffffff;
            filter: brightness(1.06);
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.24);
            transform: translateY(-2px);
        }

        .profile-buttons .btn:focus-visible {
            outline: 3px solid rgba(20, 163, 139, 0.32);
            outline-offset: 3px;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
            border: none;
            border-radius: 12px;
            min-height: 52px;
            font-weight: 800;
            letter-spacing: 0.02em;
            box-shadow: 0 8px 18px rgba(124, 58, 237, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .btn-primary:hover,
        .btn-primary:focus-visible {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(124, 58, 237, 0.26);
        }

        .btn-primary:focus-visible {
            outline: 3px solid rgba(6, 182, 212, 0.45);
            outline-offset: 3px;
        }

        .btn-outline-secondary {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
            border-radius: 10px;
            font-weight: 700;
            transition: color 0.18s ease, background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }

        .btn-group .btn-outline-secondary:hover,
        .btn-group .btn-outline-secondary:focus-visible,
        .btn-group .btn-outline-secondary:active {
            color: #ffffff;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #0891b2 100%);
            border-color: #4f46e5;
            box-shadow: 0 8px 18px rgba(79, 70, 229, 0.28);
            transform: translateY(-2px);
        }

        .btn-group .btn-outline-secondary:focus-visible {
            outline: 3px solid rgba(6, 182, 212, 0.45);
            outline-offset: 2px;
        }

        .score-pill {
            font-size: clamp(2.2rem, 4vw, 3.2rem);
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: -0.06em;
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .score-progress {
            height: 10px;
            border-radius: 999px;
            background: #e2e8f0;
            overflow: hidden;
        }

        .score-progress .progress-bar {
            background: linear-gradient(90deg, #06b6d4, #4f46e5, #7c3aed);
            border-radius: inherit;
        }

        .snapshot-list {
            display: grid;
            gap: 0.65rem;
        }

        .snapshot-item {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.85rem 1rem;
            border: 1px solid rgba(248, 113, 113, 0.22);
            border-radius: 14px;
            background: linear-gradient(100deg, #fff7ed, #fff1f2);
            text-align: left;
        }

        .snapshot-item strong {
            display: block;
            color: #9f1239;
        }

        .snapshot-item p {
            margin: 0.2rem 0 0;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .snapshot-value {
            flex: 0 0 auto;
            color: #9f1239;
            font-weight: 850;
            white-space: nowrap;
        }

        .snapshot-positive {
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 14px;
            background: linear-gradient(100deg, #f0fdf4, #ecfeff);
            color: #166534;
            line-height: 1.55;
        }

        .performance-insight {
            border-left: 4px solid #7c3aed;
            border-radius: 0 12px 12px 0;
            background: linear-gradient(100deg, #f5f3ff, #f0f9ff);
            color: #475569;
            line-height: 1.55;
        }

        .badge-excellent { background: linear-gradient(135deg, #22c55e, #16a34a); color: white; }
        .badge-good { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
        .badge-average { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: white; }
        .badge-needs { background: linear-gradient(135deg, #f87171, #ef4444); color: white; }

        .list-group-item {
            border-color: rgba(148, 163, 184, 0.32);
            background: #ffffff;
            padding-left: 0;
            padding-right: 0;
            color: var(--text);
        }

        .recommendation-list {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.85rem;
        }

        .recommendation-list .list-group-item {
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem !important;
            line-height: 1.6;
        }

        .recommendation-list strong {
            color: #312e81;
        }

        .small.text-muted, .text-secondary {
            color: var(--muted) !important;
        }

        code {
            color: #312e81;
            background: linear-gradient(135deg, #eef2ff, #ecfeff);
            border-radius: 8px;
            padding: 0.2rem 0.5rem;
            font-weight: 700;
        }

        .card-custom.p-4.bg-white {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.96));
        }

        @media (max-width: 991.98px) {
            .hero-banner {
                padding-top: 1.8rem;
                padding-bottom: 1.6rem;
            }

            .card-custom {
                border-radius: 20px;
            }
        }

        @media (max-width: 576px) {
            .btn-group {
                width: 100%;
                display: grid;
                grid-template-columns: 1fr;
                gap: 0.5rem;
            }

            .btn-group .btn {
                width: 100%;
                margin: 0;
            }

            .card-custom {
                padding: 1rem !important;
            }

            .recommendation-list {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header class="hero-banner mb-4">
        <div class="container text-center">
            <h1 class="fw-bold">Student Performance Prediction System</h1>
            <span class="badge bg-light text-dark">Student Success Forecasting • ML Insights • Academic Planning</span>
        </div>
    </header>

    <main class="container pb-5">
        <div class="row">
            <!-- Form Column -->
            <div class="col-12 mb-4">
                <div class="card card-custom p-4 bg-white">
                    <div class="parameter-header mb-3">
                        <h4 class="fw-bold mb-3">Student Parameters</h4>
                        <div class="btn-group btn-group-sm profile-buttons">
                            <button type="button" class="btn btn-outline-secondary" onclick="fillDemo('topper')">High Achiever</button>
                            <button type="button" class="btn btn-outline-secondary" onclick="fillDemo('average')">Typical Student</button>
                            <button type="button" class="btn btn-outline-secondary" onclick="fillDemo('at_risk')">Academic Support</button>
                        </div>
                    </div>

                    <form id="predictForm" method="POST" action="/predict">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Student Name</label>
                                <input type="text" class="form-control" name="student_name" id="student_name" value="Alex Smith" required>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Age</label>
                                <input type="number" class="form-control" name="age" id="age" value="19" min="15" max="30" required>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Gender</label>
                                <select class="form-select" name="gender" id="gender">
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Weekly Study Hours (hrs)</label>
                                <input type="number" class="form-control" name="study_hours_weekly" id="study_hours_weekly" value="16" min="1" max="40" step="0.5" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Attendance Rate (%)</label>
                                <input type="number" class="form-control" name="attendance_rate" id="attendance_rate" value="88" min="40" max="100" required>
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">Previous Marks (0-100)</label>
                                <input type="number" class="form-control" name="previous_marks" id="previous_marks" value="78" min="0" max="100" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Assignment Marks (0-100)</label>
                                <input type="number" class="form-control" name="assignment_marks" id="assignment_marks" value="82" min="0" max="100" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Midterm Marks (0-100)</label>
                                <input type="number" class="form-control" name="midterm_marks" id="midterm_marks" value="80" min="0" max="100" required>
                            </div>

                            <div class="col-md-4">
                                <label class="form-label">Absences (Days)</label>
                                <input type="number" class="form-control" name="absences" id="absences" value="3" min="0" max="40" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Past Failures</label>
                                <input type="number" class="form-control" name="failures" id="failures" value="0" min="0" max="5" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Sleep Hours (Daily)</label>
                                <input type="number" class="form-control" name="sleep_hours" id="sleep_hours" value="7.5" min="4" max="12" step="0.5" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Parental Education</label>
                                <select class="form-select" name="parental_education" id="parental_education">
                                    <option value="None">None</option>
                                    <option value="High School">High School</option>
                                    <option value="Some College">Some College</option>
                                    <option value="Bachelor's" selected>Bachelor's</option>
                                    <option value="Master's/PhD">Master's/PhD</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Extra Tutoring</label>
                                <select class="form-select" name="tutoring" id="tutoring">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Motivation</label>
                                <select class="form-select" name="motivation_level" id="motivation_level">
                                    <option value="Low">Low</option>
                                    <option value="Medium" selected>Medium</option>
                                    <option value="High">High</option>
                                </select>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Extracurricular Activities</label>
                                <select class="form-select" name="extracurricular" id="extracurricular">
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Home Internet Access</label>
                                <select class="form-select" name="internet_access" id="internet_access">
                                    <option value="Yes" selected>Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>

                            <div class="col-12 mt-4">
                                <button type="submit" class="btn btn-primary w-100 py-2 fw-semibold">
                                    Predict Student Final Score
                                </button>
                                <p class="small text-muted text-center mt-2 mb-0">
                                    Select a performance profile to view its prediction, or enter student data and predict the final score.
                                </p>
                            </div>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Prediction Result -->
            <div class="col-12 mb-4">
                <div class="card card-custom p-4 bg-white mb-4 text-center">
                    <h5 class="text-muted fw-normal mb-3">Model Prediction Output</h5>
                    {% if result %}
                        <div class="score-pill mb-2 text-primary">{{ result.score }} / 100</div>
                        <div class="mb-3">
                            <span class="badge px-3 py-2 fs-6 
                                {% if result.category == 'Excellent' %}badge-excellent
                                {% elif result.category == 'Good' %}badge-good
                                {% elif result.category == 'Average' %}badge-average
                                {% else %}badge-needs{% endif %}">
                                {{ result.category }}
                            </span>
                        </div>
                        <div class="progress score-progress mb-3"
                             role="progressbar"
                             aria-label="Predicted final score"
                             aria-valuenow="{{ result.score }}"
                             aria-valuemin="0"
                             aria-valuemax="100">
                            <div class="progress-bar" style="width: {{ result.score }}%"></div>
                        </div>
                        <p class="text-secondary small">{{ result.overview }}</p>
                    {% else %}
                        <div class="py-3 text-muted">
                            <p class="mb-1">Enter student information and select <strong>"Predict Student Final Score"</strong>, or select a performance profile to generate its prediction and academic recommendations.</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>

        {% if result %}
        <div class="row g-4 mb-4">
            <div class="col-lg-6">
                <section class="card card-custom p-4 bg-white h-100" aria-labelledby="attentionHeading">
                    <h5 id="attentionHeading" class="fw-bold mb-3">Areas Needing Attention</h5>
                    {% if result.low_parameters %}
                    <div class="snapshot-list">
                        {% for item in result.low_parameters %}
                        <div class="snapshot-item">
                            <div>
                                <strong>{{ item.name }}</strong>
                                <p>{{ item.guidance }}</p>
                            </div>
                            <span class="snapshot-value">{{ item.value }}</span>
                        </div>
                        {% endfor %}
                    </div>
                    {% else %}
                    <div class="snapshot-positive p-3 small">
                        No low academic indicators were identified from the information provided. Keep maintaining these positive habits.
                    </div>
                    {% endif %}
                    <div class="performance-insight p-3 small mt-3">
                        <strong class="d-block text-dark mb-1">Performance Insight</strong>
                        {% if result.category == 'Excellent' %}
                            Strong academic indicators are reflected in this prediction. Continue the consistent study and attendance habits.
                        {% elif result.category == 'Good' %}
                            The prediction is positive. Keep building on current habits and focus on improving the lowest assessment area.
                        {% elif result.category == 'Average' %}
                            Focused progress in attendance, study routine, and assessment preparation can help strengthen future performance.
                        {% else %}
                            The indicators suggest the student may benefit from a structured study plan and early support from instructors.
                        {% endif %}
                    </div>
                </section>
            </div>
            <div class="col-lg-6">
                <section class="card card-custom p-4 bg-white h-100" aria-labelledby="recommendationsHeading">
                    <h5 id="recommendationsHeading" class="fw-bold mb-3">Academic Recommendations</h5>
                    <ul class="list-group list-group-flush small recommendation-list">
                        {% for item in result.recommendations %}
                        <li class="list-group-item">
                            <strong>{{ item.title }}:</strong> {{ item.desc }}
                        </li>
                        {% endfor %}
                    </ul>
                </section>
            </div>
        </div>
        {% endif %}

        <section class="row mb-4" aria-labelledby="historyHeading">
            <div class="col-12">
                <div class="card card-custom p-3 bg-white">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 id="historyHeading" class="fw-bold mb-0">Prediction History</h6>
                        <a href="/download-history" class="btn btn-sm btn-outline-success">Export CSV</a>
                    </div>
                    <p class="small text-muted mb-0">Predictions are stored automatically into <code>prediction_history.csv</code>.</p>
                </div>
            </div>
        </section>
    </main>

    <script>
        const demoData = {{ demo_json|safe }};
        const submittedData = {{ form_data|tojson }};

        for (const [field, value] of Object.entries(submittedData)) {
            const element = document.getElementById(field);
            if (element) element.value = value;
        }

        function fillDemo(key) {
            const data = demoData[key];
            if (!data) return;
            for (const [k, value] of Object.entries(data)) {
                const el = document.getElementById(k);
                if (el) el.value = value;
            }
            document.getElementById("predictForm").requestSubmit();
        }
    </script>
</body>
</html>
"""

def generate_recommendations(data, score, category):
    items = []
    att = float(data.get('attendance_rate', 0))
    absences = int(data.get('absences', 0))
    study = float(data.get('study_hours_weekly', 0))
    sleep = float(data.get('sleep_hours', 0))

    if score < 50:
        items.append({
            "title": "Contact Your Instructor",
            "desc": "Arrange a meeting this week to review difficult topics, missed work, and a recovery plan. Check progress together every week."
        })
        items.append({
            "title": "Build a Weekly Study Plan",
            "desc": "Schedule focused 30-45 minute study sessions on at least five days each week. Review class notes after each lesson and practice with questions."
        })
    elif score < 70:
        items.append({
            "title": "Strengthen Core Subjects",
            "desc": "Choose the topics you find hardest, practice them in short daily sessions, and ask your instructor to check your understanding each week."
        })

    if att < 75 or absences > 6:
        items.append({"title": "Improve Attendance", "desc": f"Attendance is {att}% with {absences} absences. Attend every class where possible, and ask teachers for notes and make-up work for missed lessons."})
    if study < 10:
        items.append({"title": "Increase Study Time",         "desc": f"Current study time is {study} hours per week. Add 2-3 focused hours each week until reaching at least 10-15 hours, with regular breaks."})

    for subject in ("previous_marks", "assignment_marks", "midterm_marks"):
        mark = float(data.get(subject, 100))
        if mark < 60:
            subject_name = {
                "previous_marks": "Previous assessments",
                "assignment_marks": "Assignments",
                "midterm_marks": "Midterm topics"
            }[subject]
            items.append({
                "title": f"Review {subject_name}",
                "desc": f"The current mark is {mark:g}/100. Review mistakes, redo missed questions, and ask your instructor for targeted practice."
            })

    if int(data.get("failures", 0)) > 0:
        items.append({
            "title": "Recover Failed Courses",
            "desc": "Ask your academic advisor which courses or assessments can be recovered, note the deadlines, and prioritize those requirements in your study plan."
        })
    if data.get("tutoring", "No") == "No" and score < 70:
        items.append({
            "title": "Get Extra Academic Support",
            "desc": "Consider instructor office hours, peer study groups, or campus tutoring to get help with challenging topics."
        })
    if data.get("internet_access", "Yes") == "No" and score < 70:
        items.append({
            "title": "Find Study Resources",
            "desc": "Ask your school or library about printed materials, computer access, and offline resources so you can study outside class."
        })
    if sleep < 6:
        items.append({"title": "Protect Your Rest", "desc": "Aim for a consistent sleep schedule and enough rest to support focus and learning."})
    if not items:
        items.append({"title": "Maintain Your Progress", "desc": "Keep attending classes, review notes regularly, and continue the study habits that are working."})

    overview = f"Predicted score: {score}/100. Category: {category}."
    return overview, items


def estimate_score(data):
    marks = (
        0.20 * float(data.get("previous_marks", 75))
        + 0.35 * float(data.get("assignment_marks", 75))
        + 0.45 * float(data.get("midterm_marks", 75))
    )
    attendance_adjustment = (float(data.get("attendance_rate", 85)) - 80) * 0.04
    study_adjustment = (float(data.get("study_hours_weekly", 15)) - 12) * 0.08
    absence_adjustment = max(0, int(data.get("absences", 0)) - 3) * 0.25
    failure_adjustment = int(data.get("failures", 0)) * 2
    motivation_adjustment = {
        "Low": -0.5,
        "Medium": 0,
        "High": 0.5
    }.get(data.get("motivation_level", "Medium"), 0)
    sleep_adjustment = -0.5 if float(data.get("sleep_hours", 7)) < 6 else 0

    score = (
        marks + attendance_adjustment + study_adjustment
        - absence_adjustment - failure_adjustment
        + motivation_adjustment + sleep_adjustment
    )
    return round(max(0.0, min(100.0, score)), 1)


def identify_low_parameters(data):
    low_parameters = []
    attendance = float(data.get("attendance_rate", 0))
    study_hours = float(data.get("study_hours_weekly", 0))
    absences = int(data.get("absences", 0))
    sleep_hours = float(data.get("sleep_hours", 0))

    if attendance < 75:
        low_parameters.append({
            "name": "Attendance",
            "value": f"{attendance:g}%",
            "guidance": "Attendance is below the 75% target. Attend classes consistently and catch up on missed lessons."
        })
    if study_hours < 10:
        low_parameters.append({
            "name": "Weekly Study Hours",
            "value": f"{study_hours:g} hrs",
            "guidance": "Study time is below 10 hours per week. Add regular focused study sessions to the weekly routine."
        })

    for key, name in (
        ("previous_marks", "Previous Marks"),
        ("assignment_marks", "Assignment Marks"),
        ("midterm_marks", "Midterm Marks")
    ):
        mark = float(data.get(key, 100))
        if mark < 60:
            low_parameters.append({
                "name": name,
                "value": f"{mark:g}/100",
                "guidance": "This mark is below 60. Review mistakes and practice the topics that caused difficulty."
            })

    if absences > 6:
        low_parameters.append({
            "name": "Absences",
            "value": f"{absences} days",
            "guidance": "More than 6 absences can cause missed learning. Reduce absences and arrange to catch up on classwork."
        })
    if int(data.get("failures", 0)) > 0:
        failures = int(data.get("failures", 0))
        low_parameters.append({
            "name": "Past Course Failures",
            "value": str(failures),
            "guidance": "Prioritize recovery options with an instructor or academic advisor."
        })
    if sleep_hours < 6:
        low_parameters.append({
            "name": "Sleep Hours",
            "value": f"{sleep_hours:g} hrs",
            "guidance": "Sleep is below 6 hours per night. Aim for a more consistent and sufficient sleep schedule."
        })
    if data.get("motivation_level") == "Low":
        low_parameters.append({
            "name": "Motivation",
            "value": "Low",
            "guidance": "Set small weekly learning goals and ask an instructor or advisor for support."
        })

    return low_parameters


@app.route("/", methods=["GET"])
def index():
    import json
    return render_template_string(
        HTML_TEMPLATE,
        result=None,
        demo_json=json.dumps(DEMO_STUDENTS),
        form_data={}
    )

@app.route("/predict", methods=["POST"])
def predict():
    import json
    data = request.form.to_dict()
    
    # Load model if exists
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        gender_map = {"Female": 0, "Male": 1, "Other": 2}
        parent_map = {"None": 0, "High School": 1, "Some College": 2, "Bachelor's": 3, "Master's/PhD": 4}
        bin_map = {"No": 0, "Yes": 1}
        mot_map = {"Low": 0, "Medium": 1, "High": 2}

        features = [
            float(data.get("age", 18)),
            gender_map.get(data.get("gender", "Male"), 1),
            float(data.get("study_hours_weekly", 15)),
            float(data.get("attendance_rate", 85)),
            float(data.get("previous_marks", 75)),
            float(data.get("assignment_marks", 75)),
            float(data.get("midterm_marks", 75)),
            float(data.get("absences", 2)),
            float(data.get("failures", 0)),
            parent_map.get(data.get("parental_education", "High School"), 1),
            bin_map.get(data.get("tutoring", "No"), 0),
            bin_map.get(data.get("extracurricular", "No"), 0),
            mot_map.get(data.get("motivation_level", "Medium"), 1),
            float(data.get("sleep_hours", 7)),
            bin_map.get(data.get("internet_access", "Yes"), 1)
        ]

        pred = model.predict(np.array([features]))[0]
        score = round(max(0.0, min(100.0, float(pred))), 1)
    else:
        score = estimate_score(data)

    if score >= 85:
        category = "Excellent"
    elif score >= 70:
        category = "Good"
    elif score >= 50:
        category = "Average"
    else:
        category = "Needs Improvement"

    overview, recs = generate_recommendations(data, score, category)
    if not os.path.exists(MODEL_PATH):
        overview = f"Estimated from the entered academic indicators because the trained model is not available. {overview}"
    result = {
        "score": score,
        "category": category,
        "overview": overview,
        "low_parameters": identify_low_parameters(data),
        "recommendations": recs
    }

    # Save to history CSV
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "student_name", "score", "category"])
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), data.get("student_name", "Anonymous"), score, category])

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        demo_json=json.dumps(DEMO_STUDENTS),
        form_data=data
    )

@app.route("/download-history")
def download_history():
    if os.path.exists(HISTORY_FILE):
        return send_file(HISTORY_FILE, as_attachment=True)
    return "No history recorded yet", 404

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
