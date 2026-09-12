from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from app.routes import router

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Smart Reply API is running"}


app.include_router(router)
