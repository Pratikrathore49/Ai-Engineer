import os 

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

VALID_API_KEYS =[
    key.strip()
    for key in os.getenv("SECURITY_API_KEYS","").split(",")
    if key.strip()
]

def verify_api_key(
    api_key:str | None = Header(
        default=None,
        alias="X-API_Key"
    )
):
    if api_key is None:
        raise HTTPException(status_code=401,details="X-API-Key header is required")
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key