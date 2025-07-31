# api/models/schemas.py
from typing import Optional, Literal, Dict
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

# --------- shared "enums" for better docs and validation ---------

ModelType = Literal["LSTM", "GRU"]

# NOTE: exact strings match what your UI expects to display
Label = Literal["Normal", "Depression", "Suicidal", "Some Other Disorder"]

# --------- request models ---------

class PredictIn(BaseModel):
    """
    Request body for POST /predict
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw user text to classify.",
        examples=["I feel down and tired lately."]
    )
    model_type: ModelType = Field(
        "LSTM",
        description="Which model to use for inference.",
        examples=["LSTM"]
    )

# --------- response / error models ---------

class PredictOut(BaseModel):
    """
    Response body for POST /predict.
    We expose the field as `class` in JSON because your frontend reads response.data.class.
    In Python we call it `label` and map it via alias for safety.
    """
    # Make sure aliases are honored when serializing to JSON
    model_config = ConfigDict(populate_by_name=True)

    # alias -> JSON field name will be "class"
    label: Label = Field(
        ...,
        alias="class",
        description="Predicted category."
    )    

class ErrorOut(BaseModel):
    code: str = Field(examples=["MODEL_NOT_READY", "VALIDATION_ERROR"])
    message: str = Field(examples=["Model weights not loaded."])
