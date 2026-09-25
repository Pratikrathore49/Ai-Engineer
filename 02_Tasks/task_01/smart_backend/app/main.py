from dotenv import load_dotenv
load_dotenv()

from app.logger import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException

from app.health import health_check
from app.routes import router
from app.exception_handlers import(http_exception_handler,validation_exception_handler,
unhandled_exception_handler)

@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("Smart Reply API started and ready to accept requests")
    yield

app = FastAPI(lifespan=lifespan)

#regitster exception handlers
app.add_exception_handler(HTTPException,http_exception_handler)
app.add_exception_handler(RequestValidationError,validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

@app.get("/")
def root():
    logger.debug("Health check endpoint called")
    return {
        "message":"Smart Replay API is running"
    }

@app.get("/health")
def health():
    return health_check()
    
app.include_router(router)