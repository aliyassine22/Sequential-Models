from fastapi import APIRouter, Query, status
from models.schemas import PredictIn, PredictOut, ErrorOut
from controller.classification_controller import load_model, evaluate_text
from controller.preprocessing_controller import perprocess_text
from torch.utils.data import DataLoader


router = APIRouter()
@router.post(
    "/predict", # end point
    response_model=PredictOut, # output must be strucuted as dictated in PredictOut response model
    status_code=status.HTTP_200_OK, # default status code when things work properly
    summary="Run text classification", # for swagger UI
    responses={
        200: {"description": "OK"},
        422: {"model": ErrorOut, "description": "Validation error"},
        501: {"model": ErrorOut, "description": "Model not yet wired"},
    }, # custom descriptions and models for possible responses
)

def predict(payload: PredictIn) -> PredictOut:
    tensor=perprocess_text(payload.text)
    feed_data=DataLoader(tensor, batch_size=1, shuffle=False) # must be written 
    model=load_model(payload.model_type)
    result=evaluate_text(model, feed_data)
    # Ensure the label is one of the allowed strings:
    label=result
    return PredictOut(
        label=label,  # label is the python keyword mentioned earlier            
    )
    
# problem i was facing:
# our tensor shape without dataloader is (1, 125, 300)
# our tensor shpape with dataloader is (125, 300)
