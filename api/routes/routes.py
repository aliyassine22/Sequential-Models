# routes/routes.py
from fastapi import APIRouter, Query, status
from models.schemas import PredictIn, ClassificationOut

router = APIRouter()

@router.get(
    "/predict",
    response_model=ClassificationOut,
    status_code=status.HTTP_200_OK,
    summary="Quick predict (GET)",
    description="Pass text & model_type as query params."
)
def predict_get(
    text: str = Query(..., min_length=1),
    model_type: str = Query("lstm")
):
    # Example deterministic stub that matches your output schema:
    label = "Depression" if any(w in text.lower() for w in ["sad", "down"]) else "Normal"
    score = 0.82 if label == "Depression" else 0.71
    return {"label": label, "score": score}   # <-- dict with keys label/score


@router.post(
    "/predict",
    response_model=ClassificationOut,
    summary="Predict class of a text"
)
def predict(payload: PredictIn):
    text = payload.text
    model_type = payload.model_type
    return {"text":text, "model_type":model_type}   # <-- dict with keys label/score
