import os, hmac
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import model

API_KEY = os.environ.get("DCSM_API_KEY", "")
import threading
app = FastAPI(title="DCSM student outcome service", docs_url=None, redoc_url=None)


@app.on_event("startup")
def warm():
    threading.Thread(target=model.load_or_baseline, daemon=True).start()


def check(key: Optional[str]):
    if not API_KEY:
        raise HTTPException(503, "Service has no DCSM_API_KEY configured.")
    if not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(401, "Bad API key.")


class Rows(BaseModel):
    rows: List[Dict[str, Any]]


@app.get("/")
def root():
    return {"service": "dcsm-ml", "ok": True}


@app.get("/health")
def health(x_api_key: Optional[str] = Header(None)):
    check(x_api_key)
    return {"ok": True, "model": model.summary(), "min_real_rows": model.MIN_REAL}


@app.post("/train")
def train(body: Rows, x_api_key: Optional[str] = Header(None)):
    check(x_api_key)
    if len(body.rows) > 20000:
        raise HTTPException(413, "Too many rows.")
    try:
        return {"ok": True, "model": model.train(body.rows)}
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/predict")
def predict(body: Rows, x_api_key: Optional[str] = Header(None)):
    check(x_api_key)
    if len(body.rows) > 2000:
        raise HTTPException(413, "Too many rows.")
    return {"ok": True, "model": model.summary(), "predictions": model.predict(body.rows)}
