# app/rate_limiter.py
import time
import logging
from collections import defaultdict

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

MAX_REQUESTS = 20
WINDOW_SECONDS = 60

request_counts = defaultdict(list)

def rate_limit(request:Request):
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
        logger.warning("Rate limit exceeded | api_key=%s | requests=%d", api_key[:4] + "...",len(request_counts[api_key]))
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After":"60"}
        )
    # Recode this request
    request_counts[api_key].append(current_time)

