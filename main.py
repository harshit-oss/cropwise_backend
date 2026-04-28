"""
CropWise FastAPI Backend — All models retrained on your datasets
================================================================
"""
from dotenv import load_dotenv
load_dotenv()

import os, warnings
import numpy as np
import joblib
import gdown
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

warnings.filterwarnings("ignore")

app = FastAPI(title="CropWise API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Load Model 1 — Crop Recommendation
# ─────────────────────────────────────────────────────────────
_crop_bundle   = joblib.load("crop_recommendation_model_retrained.pkl")
crop_rec_model = _crop_bundle["model"]
crop_label_enc = _crop_bundle["label_encoder"]

# ─────────────────────────────────────────────────────────────
# Load Model 2 — Fertilizer
# ─────────────────────────────────────────────────────────────
_fert_bundle = joblib.load("fertilizer_model.pkl")
fert_model   = _fert_bundle["model"]
fert_le_crop = _fert_bundle["le_crop"]
fert_le_soil = _fert_bundle["le_soil"]
fert_le_season = _fert_bundle["le_season"]
fert_le_irr  = _fert_bundle["le_irr"]
FERT_CROPS   = _fert_bundle["crop_types"]
FERT_SOILS   = _fert_bundle["soil_types"]
FERT_SEASONS = _fert_bundle["season_types"]
FERT_IRR     = _fert_bundle["irr_types"]

# ─────────────────────────────────────────────────────────────
# Load Model 3 — Crop Production (download if missing)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "crop_production_model_retrained.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading production model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1RkUyH1h1UBeS91XRxqpbfYUbb8voTyZq",
        MODEL_PATH, quiet=False
    )
    print("✅ Production model downloaded!")

# ── Load production model IMMEDIATELY (not lazily) ──────────
print("Loading production model...")
_prod_bundle   = joblib.load(MODEL_PATH)
prod_model     = _prod_bundle["model"]
prod_le_state  = _prod_bundle["le_state"]
prod_le_dist   = _prod_bundle["le_district"]
prod_le_season = _prod_bundle["le_season"]
prod_le_crop   = _prod_bundle["le_crop"]
PROD_STATES    = _prod_bundle["states"]
PROD_DISTRICTS = _prod_bundle["districts"]
PROD_SEASONS   = _prod_bundle["seasons"]
PROD_CROPS     = _prod_bundle["crops"]

print("✅ All 3 models loaded!")
print(f"   Crop rec   : {len(crop_label_enc.classes_)} crops")
print(f"   Fertilizer : {FERT_CROPS}")
print(f"   Production : {len(PROD_CROPS)} crops, {len(PROD_STATES)} states, {len(PROD_DISTRICTS)} districts")

# ─────────────────────────────────────────────────────────────
# Fertilizer name map
# ─────────────────────────────────────────────────────────────
FERTILIZER_BY_CROP = {
    "barley":    {"name": "Urea + SSP",          "n_rate": 80,  "p_rate": 40,  "k_rate": 20},
    "carrot":    {"name": "NPK 10-26-26",         "n_rate": 60,  "p_rate": 80,  "k_rate": 80},
    "cotton":    {"name": "NPK 20-20-0 + Boron",  "n_rate": 120, "p_rate": 60,  "k_rate": 60},
    "maize":     {"name": "Urea + DAP",            "n_rate": 150, "p_rate": 75,  "k_rate": 50},
    "potato":    {"name": "NPK 10-26-26 + MOP",   "n_rate": 180, "p_rate": 80,  "k_rate": 100},
    "rice":      {"name": "Urea + DAP",            "n_rate": 120, "p_rate": 60,  "k_rate": 60},
    "soybean":   {"name": "DAP + MOP",             "n_rate": 30,  "p_rate": 60,  "k_rate": 40},
    "sugarcane": {"name": "Urea + MOP",            "n_rate": 250, "p_rate": 80,  "k_rate": 120},
    "tomato":    {"name": "NPK 19-19-19",          "n_rate": 120, "p_rate": 60,  "k_rate": 80},
    "wheat":     {"name": "NPK 12-32-16",          "n_rate": 120, "p_rate": 60,  "k_rate": 40},
}

# ─────────────────────────────────────────────────────────────
# Groq AI
# ─────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are CropWise, an expert agricultural AI assistant for Indian farmers.
Help with crop selection, soil health, fertilizer usage, pest control, irrigation,
and farming best practices. Be concise, practical, and friendly.
Relate advice to Indian farming conditions and seasons (Kharif/Rabi/Zaid) where relevant."""


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

class CropRecInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class FertilizerInput(BaseModel):
    crop_type: str
    soil_type: str
    season: str
    irrigation_type: str
    farm_area_acres: float
    fertilizer_used_tons: float
    pesticide_used_kg: float
    water_usage_m3: float


class CropProductionInput(BaseModel):
    state: str
    district: str
    crop_year: int
    season: str
    crop: str
    area: float
    yield_per_hectare: float


class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def build_crop_rec_features(d: CropRecInput) -> np.ndarray:
    ns  = d.N + d.P + d.K
    nr  = d.N / (d.P + 1e-6)
    thi = d.temperature * d.humidity
    pri = d.ph * d.rainfall
    return np.array([[d.N, d.P, d.K, d.temperature, d.humidity,
                      d.ph, d.rainfall, ns, nr, thi, pri]])


def safe_encode(le, value, field_name):
    """Encode a label; raise 400 with helpful message if unknown."""
    for v in [value.strip(), value.strip().title(), value.strip().capitalize(),
              value.strip().upper()]:
        if v in le.classes_:
            return le.transform([v])[0]
    raise HTTPException(
        status_code=400,
        detail=f"Unknown {field_name}: '{value}'. Valid: {list(le.classes_)[:10]}..."
    )


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "running",
        "version": "3.0",
        "models": {
            "crop_recommendation": f"{len(crop_label_enc.classes_)} crops",
            "fertilizer":         f"{len(FERT_CROPS)} crops",
            "crop_production":    f"{len(PROD_CROPS)} crops, {len(PROD_STATES)} states",
        }
    }


# ── 1. Crop Recommendation ────────────────────────────────────
@app.get("/api/crop-options")
def crop_options():
    return {"crops": list(crop_label_enc.classes_)}


@app.post("/api/predict-crop")
def predict_crop(data: CropRecInput):
    try:
        features = build_crop_rec_features(data)
        pred_idx = int(crop_rec_model.predict(features)[0])
        proba    = crop_rec_model.predict_proba(features)[0]
        top3_idx = np.argsort(proba)[-3:][::-1]
        return {
            "status":     "success",
            "crop":       crop_label_enc.inverse_transform([pred_idx])[0].capitalize(),
            "confidence": round(float(proba[pred_idx]) * 100, 1),
            "top3": [
                {
                    "crop":       crop_label_enc.inverse_transform([int(i)])[0].capitalize(),
                    "confidence": round(float(proba[i]) * 100, 1),
                }
                for i in top3_idx
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. Fertilizer Advisory ────────────────────────────────────
@app.get("/api/fertilizer-options")
def fertilizer_options():
    return {
        "crop_types": FERT_CROPS,
        "soil_types": FERT_SOILS,
        "seasons":    FERT_SEASONS,
        "irrigation": FERT_IRR,
    }


@app.post("/api/predict-fertilizer")
def predict_fertilizer(data: FertilizerInput):
    try:
        crop_enc   = safe_encode(fert_le_crop,   data.crop_type,       "crop_type")
        soil_enc   = safe_encode(fert_le_soil,   data.soil_type,       "soil_type")
        season_enc = safe_encode(fert_le_season, data.season,          "season")
        irr_enc    = safe_encode(fert_le_irr,    data.irrigation_type, "irrigation_type")

        features = np.array([[
            crop_enc, soil_enc, season_enc, irr_enc,
            data.farm_area_acres, data.fertilizer_used_tons,
            data.pesticide_used_kg, data.water_usage_m3,
        ]])

        predicted_yield = float(fert_model.predict(features)[0])
        crop_key = data.crop_type.lower().strip()
        fert_rec = FERTILIZER_BY_CROP.get(crop_key, {
            "name": "NPK 10-26-26", "n_rate": 80, "p_rate": 40, "k_rate": 40
        })

        area = data.farm_area_acres
        n_kg = round(fert_rec["n_rate"] * area / 100, 1)
        p_kg = round(fert_rec["p_rate"] * area / 100, 1)
        k_kg = round(fert_rec["k_rate"] * area / 100, 1)

        if predicted_yield < 15:
            level  = "low"
            advice = ("Yield is low. Increase organic matter with compost before chemical fertilizers. "
                      "Consider soil testing to identify nutrient deficiencies. "
                      "Apply fertilizer in split doses — 50% at sowing, 25% at 30 days, 25% at 60 days.")
        elif predicted_yield < 35:
            level  = "medium"
            advice = ("Moderate yield expected. Apply recommended NPK in 2–3 split doses. "
                      "Ensure adequate irrigation during critical growth stages. "
                      "Monitor for pest and disease pressure.")
        else:
            level  = "high"
            advice = ("Good yield expected! Maintain soil health with balanced NPK. "
                      "Avoid over-fertilizing as it can cause nutrient runoff. "
                      "Consider foliar spray of micronutrients for optimal results.")

        return {
            "status":          "success",
            "fertilizer_name": fert_rec["name"],
            "predicted_yield": round(predicted_yield, 2),
            "yield_level":     level,
            "n_kg":            n_kg,
            "p_kg":            p_kg,
            "k_kg":            k_kg,
            "advice":          advice,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Crop Production ────────────────────────────────────────
@app.get("/api/production-options")
def production_options():
    return {
        "states":    PROD_STATES,
        "districts": PROD_DISTRICTS,   # ← was missing before!
        "seasons":   PROD_SEASONS,
        "crops":     PROD_CROPS,
    }


@app.post("/api/predict-production")
def predict_production(data: CropProductionInput):
    try:
        # ── All encoders are ready (model loaded at startup) ──
        state_enc  = safe_encode(prod_le_state,  data.state,    "state")
        dist_enc   = safe_encode(prod_le_dist,   data.district, "district")
        season_enc = safe_encode(prod_le_season, data.season,   "season")
        crop_enc   = safe_encode(prod_le_crop,   data.crop,     "crop")
        year_group = (data.crop_year // 5) * 5

        features = np.array([[
            state_enc, dist_enc, data.crop_year,
            season_enc, crop_enc,
            data.area, data.yield_per_hectare, year_group,
        ]])

        pred = float(prod_model.predict(features)[0])
        return {
            "status":                    "success",
            "predicted_production_tons": round(pred, 2),
            "area_hectares":             data.area,
            "crop":                      data.crop,
            "state":                     data.state,
            "season":                    data.season,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. AI Chatbot ─────────────────────────────────────────────
@app.post("/api/chat")
def chat(data: ChatMessage):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in data.history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": data.message})

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        return {"status": "success", "reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
