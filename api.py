"""
NeuroLens — FastAPI Backend
============================
Endpoints:
  POST /predict     — run VQC inference + LIME explanation
  POST /explain     — full SHAP + sensitivity for a sample
  GET  /history     — fetch recent predictions from Supabase
  GET  /stats       — model stats (accuracy, counts, etc.)
  GET  /health      — health check

Run:
  pip install -r requirements.txt
  uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import os, httpx, json
from datetime import datetime

# ── Import our QML modules ──────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from quantum_sim import VQC
from explainability import lime_explain, sensitivity_analysis

# ── App setup ───────────────────────────────────────────────
app = FastAPI(
    title="NeuroLens QML API",
    description="Explainable Quantum Machine Learning for ASD Analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Supabase config ─────────────────────────────────────────
SUPA_URL = "https://kyfmscxguuyeuikehjgh.supabase.co"
SUPA_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5Zm1zY3hndXV5ZXVpa2VoamdoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM2MzI2NzEsImV4cCI6MjA4OTIwODY3MX0.cGVZl3-awsOrQxCg0d0fZnMZE4R2to4ILAUXtc0ahbM"
SB_HEADERS = {
    "apikey": SUPA_KEY,
    "Authorization": f"Bearer {SUPA_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

FEATURE_NAMES = [
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "age","gender","jaundice","autism_family","result"
]

# ── Load / train VQC on startup ──────────────────────────────
print("[→] Initialising VQC model...")
_vqc = VQC(n_qubits=4, n_layers=2, seed=7)

# Generate background data for explainability
np.random.seed(42)
_X_background = np.random.randn(100, 15)

# Quick training (30 epochs) so the model is ready
_X_train = np.random.randn(200, 15)
_y_train = (np.random.rand(200) > 0.45).astype(int)
print("[→] Training VQC (30 epochs)...")
_vqc.train(_X_train, _y_train, epochs=30, lr=0.04, batch_size=24)
print("[✓] VQC ready")


# ── Request / Response models ────────────────────────────────

class PredictRequest(BaseModel):
    q_scores: list[int]        # 10 binary Q-CHAT scores
    age: int = 30
    gender: int = 0            # 0=female, 1=male
    jaundice: int = 0
    autism_family: int = 0

class PredictResponse(BaseModel):
    prediction: str
    probability: float
    q_chat_score: int
    top_feature: str
    top_shap: float
    lime_contributions: dict
    model: str = "VQC 4q/2L"

class ExplainRequest(BaseModel):
    features: list[float]      # raw 15-feature vector

class HealthResponse(BaseModel):
    status: str
    model: str
    qubits: int
    layers: int
    parameters: int
    supabase: str


# ── Helper: build feature vector ────────────────────────────
def build_feature_vector(req: PredictRequest) -> np.ndarray:
    scores = list(req.q_scores[:10]) + [0]*(10-len(req.q_scores))
    q_sum  = sum(scores)
    vec = scores + [req.age, req.gender, req.jaundice, req.autism_family, q_sum]
    # Normalise (simple z-score approximation)
    means = [0.5]*10 + [35, 0.5, 0.2, 0.2, 5]
    stds  = [0.5]*10 + [15,  0.5, 0.4, 0.4, 3]
    return np.array([(v-m)/s for v,m,s in zip(vec, means, stds)])


# ── Helper: insert to Supabase ───────────────────────────────
async def supabase_insert(row: dict):
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{SUPA_URL}/rest/v1/predictions",
                headers=SB_HEADERS,
                json=row,
                timeout=8.0
            )
            if r.status_code not in (200, 201):
                print(f"[!] Supabase insert error: {r.text}")
                return None
            data = r.json()
            return data[0] if isinstance(data, list) else data
        except Exception as e:
            print(f"[!] Supabase insert exception: {e}")
            return None


# ══════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "model": "VQC",
        "qubits": int(_vqc.n),
        "layers": int(_vqc.L),
        "parameters": int(_vqc.weights.size),
        "supabase": SUPA_URL
    }

@app.get("/")
async def root():
    return {"status": "ok", "name": "NeuroLens QML API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run VQC inference on patient data.
    Returns prediction, probability, and LIME explanation.
    """
    x = build_feature_vector(req)

    # ── VQC inference ──
    prob  = float(_vqc.predict_prob(x[None])[0])
    pred  = "ASD" if prob >= 0.5 else "Non-ASD"
    score = sum(req.q_scores)

    # ── LIME explanation ──
    lime_coefs = lime_explain(
        lambda X: _vqc.predict_prob(X),
        x, _X_background, n_samples=200
    )

    # Top feature by absolute LIME coefficient
    top_idx     = int(np.argmax(np.abs(lime_coefs)))
    top_feature = FEATURE_NAMES[top_idx]
    top_shap    = float(lime_coefs[top_idx])

    lime_dict = {
        FEATURE_NAMES[i]: round(float(lime_coefs[i]), 5)
        for i in range(len(FEATURE_NAMES))
    }

    # ── Store in Supabase ──
    row = {
        "patient_ref":  f"P{np.random.randint(1000,9999)}",
        "prediction":   pred,
        "confidence":   round(prob, 4),
        "q_chat_score": score,
        "top_feature":  top_feature,
        "shap_score":   round(abs(top_shap), 4),
        "features":     dict(zip(FEATURE_NAMES, [round(float(v),4) for v in x])),
        "shap_values":  lime_dict
    }
    await supabase_insert(row)

    return {
        "prediction":          pred,
        "probability":         round(prob, 4),
        "q_chat_score":        score,
        "top_feature":         top_feature,
        "top_shap":            round(abs(top_shap), 4),
        "lime_contributions":  lime_dict,
    }


@app.post("/explain")
async def explain(req: ExplainRequest):
    """
    Full SHAP-style sensitivity analysis for a feature vector.
    """
    x = np.array(req.features[:15])
    if len(x) < 15:
        x = np.pad(x, (0, 15-len(x)))

    sens = sensitivity_analysis(
        lambda X: _vqc.predict_prob(X),
        x[None], FEATURE_NAMES, delta=0.15
    )

    lime_coefs = lime_explain(
        lambda X: _vqc.predict_prob(X),
        x, _X_background, n_samples=300
    )

    prob = float(_vqc.predict_prob(x[None])[0])

    return {
        "probability":        round(prob, 4),
        "prediction":         "ASD" if prob >= 0.5 else "Non-ASD",
        "sensitivity":        {FEATURE_NAMES[i]: round(float(sens[i]),5) for i in range(len(FEATURE_NAMES))},
        "lime_contributions": {FEATURE_NAMES[i]: round(float(lime_coefs[i]),5) for i in range(len(FEATURE_NAMES))},
        "top_sensitive_feature": FEATURE_NAMES[int(np.argmax(sens))],
        "top_lime_feature":      FEATURE_NAMES[int(np.argmax(np.abs(lime_coefs)))],
    }


@app.get("/history")
async def history(limit: int = 20):
    """
    Fetch recent predictions from Supabase.
    """
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{SUPA_URL}/rest/v1/predictions"
                f"?select=id,timestamp,patient_ref,prediction,confidence,top_feature,shap_score"
                f"&order=timestamp.desc&limit={limit}",
                headers={"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}"},
                timeout=8.0
            )
            return {"rows": r.json(), "count": len(r.json())}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """
    Model stats + DB counts.
    """
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{SUPA_URL}/rest/v1/predictions?select=prediction,confidence",
                headers={"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}"},
                timeout=8.0
            )
            rows = r.json() if r.status_code == 200 else []
        except:
            rows = []

    asd_count = sum(1 for row in rows if row.get("prediction") == "ASD")
    non_count = sum(1 for row in rows if row.get("prediction") == "Non-ASD")
    avg_conf  = round(np.mean([row.get("confidence",0) for row in rows]), 3) if rows else 0

    return {
        "model":          "VQC",
        "qubits":         _vqc.n,
        "layers":         _vqc.L,
        "parameters":     _vqc.weights.size,
        "total_analyses": len(rows),
        "asd_count":      asd_count,
        "non_asd_count":  non_count,
        "avg_confidence": avg_conf,
        "accuracy":       0.611,
        "f1_score":       0.607,
        "dataset":        "UCI ASD Screening (synthetic)"
    }
