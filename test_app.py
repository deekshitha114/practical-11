"""
test_app.py — CI smoke tests for the Student Result Prediction API
==================================================================
Used by GitHub Actions to validate the container before pushing to GHCR.
Run with:  pytest test_app.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app (model.pkl must exist in the working directory)
from app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    body = response.json()
    assert body["status"] == "healthy"


def test_health_contains_version():
    response = client.get("/health")
    assert "version" in response.json()


# ---------------------------------------------------------------------------
# Successful prediction
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "attendance":      85.0,
    "study_hours":     5.0,
    "previous_scores": 78.0,
    "assignments":     9,
    "extracurricular": 1,
}


def test_predict_valid_request_returns_200():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text


def test_predict_returns_pass_or_fail():
    response = client.post("/predict", json=VALID_PAYLOAD)
    prediction = response.json()["prediction"]
    assert prediction in ("Pass", "Fail"), f"Unexpected prediction: {prediction}"


def test_predict_response_contains_request_id():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "request_id" in response.json()


def test_predict_response_contains_version():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "version" in response.json()


# ---------------------------------------------------------------------------
# Validation failures  (HTTP 422)
# ---------------------------------------------------------------------------
def test_predict_missing_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "attendance"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    bad_payload = {**VALID_PAYLOAD, "attendance": "not-a-number"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_out_of_range_attendance_returns_422():
    bad_payload = {**VALID_PAYLOAD, "attendance": 150.0}  # > 100 → ge/le violation
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_out_of_range_assignments_returns_422():
    bad_payload = {**VALID_PAYLOAD, "assignments": 15}  # > 10
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 404 for unknown routes
# ---------------------------------------------------------------------------
def test_unknown_route_returns_404():
    response = client.get("/not-a-real-route")
    assert response.status_code == 404


def test_another_unknown_route_returns_404():
    response = client.post("/nonexistent-endpoint", json={})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Request-ID header propagation
# ---------------------------------------------------------------------------
def test_x_request_id_header_is_echoed():
    custom_id = "test-request-abc-123"
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
        headers={"X-Request-ID": custom_id},
    )
    assert response.status_code == 200
    # Middleware echoes back the X-Request-ID in the response header
    assert response.headers.get("X-Request-ID") == custom_id


# ---------------------------------------------------------------------------
# Low-attendance → likely Fail (sanity check, not a hard assertion)
# ---------------------------------------------------------------------------
def test_predict_low_attendance_student():
    poor_student = {
        "attendance":      30.0,
        "study_hours":     0.5,
        "previous_scores": 35.0,
        "assignments":     1,
        "extracurricular": 0,
    }
    response = client.post("/predict", json=poor_student)
    assert response.status_code == 200
    # Model should predict Fail for a clearly poor-performing student
    assert response.json()["prediction"] == "Fail"
