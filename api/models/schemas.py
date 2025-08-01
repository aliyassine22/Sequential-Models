# api/models/schemas.py
from typing import  Literal # used to restrict strings
from pydantic import BaseModel, Field 
# BaseModel: pydantic base class used for defining request/response schemas.
# Field: adds metadata like default values, constraints, descriptions

from pydantic.config import ConfigDict # used to allow alias names 

# shared enums for better docs and validation 
ModelType = Literal["LSTM", "GRU"]

# exact strings match what your UI expects to display, if any thing is different, you will get an error
Label = Literal["Normal", "Depression", "Suicidal", "Some Other Disorder"]


"""
Serialization means converting a Python object (like a Pydantic model or a dictionary) into a format that can be easily sent over the network or stored (JSON format)
Python object ➜ Serialization ➜ JSON string for API response.  Used in the PredictOut method
JSON string ➜ Deserialization ➜ Python object on the receiving end. Used in the PredictIn method 
"""

# request models 
class PredictIn(BaseModel):
    # request body for pOST /predict
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

# response models 
"""
Response body for POST /predict.
We expose the field as class in JSON because the frontend reads response.data.class
"""
class PredictOut(BaseModel):
    # Make sure aliases are honored when serializing to JSON
    model_config = ConfigDict(populate_by_name=True) # ensures the alias (class) is respected during serialization
    label: Label = Field(
        ..., # field must be present, thats why we use 3 dots, however it is empty
        alias="class", # serialization, to json format, we use class. label is used instead of class in deserialization to python compatible formats
        description="Predicted category."
    )    

class ErrorOut(BaseModel):
    code: str = Field(examples=["MODEL_NOT_READY", "VALIDATION_ERROR"])
    message: str = Field(examples=["Model weights not loaded."])