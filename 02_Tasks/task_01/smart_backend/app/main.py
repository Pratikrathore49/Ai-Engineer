# app/main.py

from fastapi import FastAPI
from dotenv import load_dotenv

from app.logger import setup_logging
from app.routes import router
from contextlib import asynccontextmanager

# Set up structured logging before anything else
setup_logging()

import logging
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Smart Reply API started and readyto accept requests")
    yield
app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    logger.debug("Health check endpoint called")
    return {"message": "Smart Reply API is running"}

app.include_router(router)
