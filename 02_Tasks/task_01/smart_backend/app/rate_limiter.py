# app/rate_limiter.py
import time
import logging
from collections import defaultdict

from app.auth import verify_api_key
from fastapi import HTTPException, Request,Depends

logger = logging.getLogger(__name__)

MAX_REQUESTS = 20
WINDOW_SECONDS = 60

request_counts = defaultdict(list)

def rate_limit(request:Request, api_key:str = Depends(verify_api_key)):
    api_key = request.state.api_key

    current_time = time.time()

    #Keep only requests from the current 60-second window
    request_counts[api_key]=[
        timestamp
        for timestamp in request_counts[api_key]
        if current_time - timestamp < WINDOW_SECONDS
    ]
    # Reject if the client has already made 20 requests
    if len(request_counts[api_key]) >= MAX_REQUESTS:
       safe_key_hint = api_key[:min(4, len(api_key))] + "..."
       logger.warning("Rate limit exceeded | key_hint=%s | requests=%d",safe_key_hint,len(request_counts[api_key]))
       raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After":"60"}
        )
    # Recode this request
    request_counts[api_key].append(current_time)

