from fastapi import FastAPI
from app.routes import router

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Smart Reply API is running"}

app.include_router(router)
