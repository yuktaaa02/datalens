from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router

app = FastAPI(
    title="DataLens AI API",
    description="Intelligent and Explainable Data Preprocessing Recommendation System",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite defaults to port 5173 / 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "DataLens AI Backend"}