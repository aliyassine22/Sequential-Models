import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.routes import router 

app = FastAPI(
    title="Mental Health NLP API",
    version="0.1.0",
    description="""
API for preprocessing and classification of text. use /docs to try endpoints and track progress.
""",
    docs_url="/docs",   # Swagger UI
    redoc_url="/redoc", # ReDoc
)

# Allow your vite dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ], # allow_origins must be a list, links the fastAPI app to our frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register routes
app.include_router(router) # we are connecting our fastAI application with the API routes defined in a seperate file via a router, making them accessible

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # our backend is running at port 8000 of our local host