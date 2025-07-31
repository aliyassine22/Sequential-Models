from pydantic import BaseModel, Field
from typing import Literal

ModelType = Literal["gru", "lstm"]

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["I feel down today but hopeful."])
    model_type: ModelType = Field("lstm", description="Which model to use")

# need to make this compatible with the post function
class ClassificationOut(BaseModel):
    model_type: str = Field(examples=["Depression", "Normal", "Suicidal", "Other"])