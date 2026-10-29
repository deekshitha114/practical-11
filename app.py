"""
app.py  —  Student Result Prediction API  (Practical-11 edition)
================================================================
Practical-11 additions over the baseline FastAPI service:
  • Structured Python logging to stdout (container-friendly).
  • Request/response middleware: method, path, status, duration, request_id.
  • RequestValidationError handler: logs field locations + error count.
  • Safe prediction logging: feature names only; no student PII.
  • Model-load success/failure events.
  • logger.exception() for unexpected runtime failures.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import logging
import sys
import time
import uuid

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ===========================================================================
# STEP 4 — Logger configuration
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

# ===========================================================================
# Model constants
# ===========================================================================
MODEL_FILE = "model.pkl"
FEATURES   = ["attendance", "study_hours", "previous_scores", "assignments", "extracurricular"]
API_VERSION = "1.0.0"

# ===========================================================================
# STEP 5 — Model loading with logging
# ===========================================================================
try:
    model = joblib.load(MODEL_FILE)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise

# ===========================================================================
# FastAPI application
# ===========================================================================
app = FastAPI(
    title="Student Result Prediction API",
    description=(
        "Predicts whether a student will Pass or Fail based on "
        "attendance, study hours, previous scores, assignments submitted, "
        "and extracurricular participation.  "
        "Practical-11: logging and error tracking enabled."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow browser / Swagger requests from any origin in a lab setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# ===========================================================================
# STEP 6 — Request / Response logging middleware
# ===========================================================================
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Assigns a request_id (from X-Request-ID header or auto-generated),
    measures wall-clock response time, and logs every HTTP transaction.
    Responses with status >= 400 are logged at WARNING level.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


# ===========================================================================
# STEP 7 — FastAPI validation-error handler
# ===========================================================================
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Logs field-level validation failures (field locations + count) before
    returning FastAPI's standard 422 response.  The full invalid payload is
    NOT logged to avoid capturing potentially sensitive data.
    """
    fields = [
        ".".join(str(part) for part in error.get("loc", []))
        for error in exc.errors()
    ]

    logger.warning(
        "event=validation_error path=%s field_locations=%s error_count=%s",
        request.url.path,
        fields,
        len(exc.errors()),
    )

    return await request_validation_exception_handler(request, exc)


# ===========================================================================
# Pydantic input schema
# ===========================================================================
class StudentResultInput(BaseModel):
    """
    Input payload for the /predict endpoint.

    Field descriptions reflect the features used during model training.
    No student-identifying information (name, roll number, email, etc.)
    is part of this schema.
    """
    attendance: float = Field(
        ...,
        ge=0,
        le=100,
        description="Attendance percentage (0–100).",
        examples=[82.5],
    )
    study_hours: float = Field(
        ...,
        ge=0,
        le=24,
        description="Average daily study hours (0–24).",
        examples=[4.5],
    )
    previous_scores: float = Field(
        ...,
        ge=0,
        le=100,
        description="Previous examination score percentage (0–100).",
        examples=[73.0],
    )
    assignments: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of assignments submitted out of 10.",
        examples=[8],
    )
    extracurricular: int = Field(
        ...,
        ge=0,
        le=1,
        description="Extracurricular participation flag: 1 = yes, 0 = no.",
        examples=[1],
    )

    @field_validator("attendance", "study_hours", "previous_scores")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Value must be ≥ 0")
        return v


class PredictionResponse(BaseModel):
    prediction: str
    request_id: str
    version: str


# ===========================================================================
# STEP 8 — Safe prediction logging  (SAFE_INPUT_FIELDS)
# ===========================================================================
# Add only non-sensitive feature names here when classroom value-logging is
# required.  Empty set = log feature *names* only (default safe behaviour).
SAFE_INPUT_FIELDS: set = set()
# Example (classroom only — no PII): SAFE_INPUT_FIELDS = {"attendance", "study_hours"}


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/health", summary="Health / readiness check", tags=["Operations"])
def health():
    """
    Returns a simple 200 OK response that confirms the service is running.
    The request middleware automatically logs method, path, status, and duration.
    """
    return {
        "status":  "healthy",
        "version": API_VERSION,
        "model":   MODEL_FILE,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict student result (Pass / Fail)",
    tags=["Prediction"],
)
def predict(data: StudentResultInput, request: Request):
    """
    Accepts anonymised academic-performance features and returns a Pass/Fail
    prediction.

    - **attendance** — percentage of classes attended
    - **study_hours** — average daily study hours
    - **previous_scores** — previous exam score (%)
    - **assignments** — assignments submitted out of 10
    - **extracurricular** — extracurricular participation (0 or 1)
    """
    # Retrieve the request ID echoed from the middleware header (if present)
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    try:
        payload = data.model_dump()  # use data.dict() for Pydantic v1

        # Build the safe-input dict from approved fields only
        safe_input = {
            key: value
            for key, value in payload.items()
            if key in SAFE_INPUT_FIELDS
        }

        logger.info(
            "event=prediction_request request_id=%s feature_names=%s safe_input=%s",
            request_id,
            sorted(payload.keys()),
            safe_input,
        )

        # ------------------------------------------------------------------
        # Inference — preserve original feature order expected by the model
        # ------------------------------------------------------------------
        input_df    = pd.DataFrame([payload])[FEATURES]
        prediction  = model.predict(input_df)[0]

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            request_id,
            str(prediction),
        )

        return PredictionResponse(
            prediction=str(prediction),
            request_id=request_id,
            version=API_VERSION,
        )

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions without wrapping them
        raise

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            request_id,
        )
        raise HTTPException(status_code=500, detail="Prediction failed due to a server error.")


@app.get("/", summary="API root redirect", include_in_schema=False)
def root():
    """Redirect landing page to interactive API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
