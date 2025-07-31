import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.routes import router as api_router

app = FastAPI(
    title="Mental Health NLP API",
    version="0.1.0",
    description="""
API for preprocessing and classification of text.
Use **/docs** to try endpoints and track progress.
""",
    docs_url="/docs",   # Swagger UI
    redoc_url="/redoc", # ReDoc
)

# Allow your Vite dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)