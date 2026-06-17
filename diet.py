"""
diet.py — Diet-based acne risk prediction API (PRODUCTION + DEBUG FIXED)
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import pandas as pd
import joblib

from auth import get_current_user
from engine import AcneGuardEngine

router = APIRouter(prefix="/diet", tags=["Diet Analysis"])

# ─────────────────────────────────────────────
# ✅ FIXED PATH (IMPORTANT)
# ─────────────────────────────────────────────
# diet.py is at: backend/diet.py

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "skin_health_backend" / "ml"

MODEL_PATH = ML_DIR / "model.joblib"
ENCODERS_PATH = ML_DIR / "encoders.joblib"

print("\n[DietML] ================= PATH DEBUG =================")
print("[DietML] BASE_DIR =", BASE_DIR)
print("[DietML] ML_DIR =", ML_DIR)
print("[DietML] MODEL_PATH =", MODEL_PATH)
print("[DietML] ENCODERS_PATH =", ENCODERS_PATH)
print("[DietML] MODEL EXISTS =", MODEL_PATH.exists())
print("[DietML] ENCODERS EXISTS =", ENCODERS_PATH.exists())
print("[DietML] =================================================\n")

# ─────────────────────────────────────────────
# MODEL HOLDERS
# ─────────────────────────────────────────────
_model = None
_preprocessor = None
_target_enc = None

# ─────────────────────────────────────────────
# SAFE MODEL LOADING + DEBUG
# ─────────────────────────────────────────────
try:
    if MODEL_PATH.exists() and ENCODERS_PATH.exists():

        _model = joblib.load(MODEL_PATH)
        artifacts = joblib.load(ENCODERS_PATH)

        # 🔥 DEBUG: inspect encoders file
        print("\n[DietML] ========== ARTIFACT DEBUG ==========")
        print("[DietML] ARTIFACT TYPE =", type(artifacts))

        try:
            print("[DietML] ARTIFACT KEYS =", artifacts.keys())
        except Exception:
            print("[DietML] ARTIFACT HAS NO KEYS (NOT A DICT)")

        print("[DietML] FULL ARTIFACT =", artifacts)
        print("[DietML] ======================================\n")

        # extract objects
        _preprocessor = artifacts.get("preprocessor")
        _target_enc = artifacts.get("target_encoder")

        print("[DietML] PREPROCESSOR =", _preprocessor)
        print("[DietML] TARGET_ENCODER =", _target_enc)

        print("\n[DietML] ✅ Model loaded successfully\n")

    else:
        print("\n[DietML ERROR] ❌ Model or encoders missing\n")

except Exception as e:
    print("\n[DietML ERROR]", str(e), "\n")


# ─────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────
class DietPredictRequest(BaseModel):
    diet_type: str
    specific_food_item: str
    sugar_intake: str
    dairy_intake: str
    processed_food_freq: str
    spicy_food_freq: str
    oily_food_level: str
    water_intake_liters: float
    sleep_hours: int
    stress_level: int = Field(..., ge=1, le=5)
    exercise: str
    skin_type: str


class DietPredictResponse(BaseModel):
    risk_level: str
    confidence_score: float
    main_causes: List[str]
    recommendations: List[str]


# ─────────────────────────────────────────────
# FOOD MAP
# ─────────────────────────────────────────────
FOOD_MAP = {
    "Red Meat": "High_Inflammation",
    "Processed Meat": "High_Inflammation",
    "Starchy/Potato": "High_Glycemic",
    "Leafy Greens": "Antioxidant_Rich",
    "Fish": "Omega3_Rich",
    "Nuts/Seeds": "Omega3_Rich",
    "Paneer/Dairy Rich": "Hormonal_Trigger",
    "Eggs": "Hormonal_Trigger",
    "Chicken": "Balanced",
}


def get_category(food: str):
    return FOOD_MAP.get(food, "Balanced")


# ─────────────────────────────────────────────
# API ENDPOINT
# ─────────────────────────────────────────────
@router.post("/predict", response_model=DietPredictResponse)
async def predict_diet(
    body: DietPredictRequest,
    current_user: dict = Depends(get_current_user)
):

    # ❌ HARD STOP IF MODEL NOT READY
    if _model is None or _preprocessor is None or _target_enc is None:
        print("\n[DietML ERROR] ❌ Runtime model objects missing")
        print("[DietML] _model =", _model)
        print("[DietML] _preprocessor =", _preprocessor)
        print("[DietML] _target_enc =", _target_enc)

        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check encoders.joblib structure or model path."
        )

    try:
        food_cat = get_category(body.specific_food_item)

        row = body.dict()
        row["food_category"] = food_cat

        df = pd.DataFrame([row])

        X = _preprocessor.transform(df)

        pred_idx = _model.predict(X)[0]
        risk = _target_enc.inverse_transform([pred_idx])[0]

        probs = _model.predict_proba(X)[0]
        confidence = float(max(probs))

        # ── SAFE ENGINE CALL ──
        try:
            engine = AcneGuardEngine()
            explanation = await engine.generate_diet_explanation(body.dict(), risk)
            causes = explanation.main_causes
            recs = explanation.recommendations

        except Exception as e:
            print("[DietML ENGINE ERROR]", e)
            causes = ["Diet-based ML prediction"]
            recs = ["Maintain balanced diet, hydration, and sleep"]

        return DietPredictResponse(
            risk_level=risk,
            confidence_score=round(confidence, 4),
            main_causes=causes,
            recommendations=recs,
        )

    except Exception as e:
        print("[DietML API ERROR]", e)
        raise HTTPException(status_code=500, detail=str(e))# 
# 
# 















# """
# # diet.py — Diet-based acne risk prediction API (PRODUCTION FIXED)
# # """

# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel, Field
# from typing import List
# from pathlib import Path
# import pandas as pd
# import joblib

# from auth import get_current_user
# from engine import AcneGuardEngine

# router = APIRouter(prefix="/diet", tags=["Diet Analysis"])

# # ─────────────────────────────────────────────
# # ✅ FIXED PATH (IMPORTANT CORRECTION)
# # ─────────────────────────────────────────────

# # diet.py is located at:
# # backend/diet.py

# BASE_DIR = Path(__file__).resolve().parent

# ML_DIR = BASE_DIR / "skin_health_backend" / "ml"

# MODEL_PATH = ML_DIR / "model.joblib"
# ENCODERS_PATH = ML_DIR / "encoders.joblib"

# print("[DietML] BASE_DIR =", BASE_DIR)
# print("[DietML] MODEL_PATH =", MODEL_PATH)
# print("[DietML] EXISTS =", MODEL_PATH.exists())

# # ─────────────────────────────────────────────
# # MODEL HOLDERS
# # ─────────────────────────────────────────────
# _model = None
# _preprocessor = None
# _target_enc = None

# # ─────────────────────────────────────────────
# # SAFE MODEL LOADING
# # ─────────────────────────────────────────────
# try:
#     if MODEL_PATH.exists() and ENCODERS_PATH.exists():

#         _model = joblib.load(MODEL_PATH)
#         artifacts = joblib.load(ENCODERS_PATH)

#         _preprocessor = artifacts.get("preprocessor")
#         _target_enc = artifacts.get("target_encoder")

#         print("[DietML] ✅ Model loaded successfully")

#     else:
#         print("[DietML ERROR] ❌ Model or encoders missing")

# except Exception as e:
#     print("[DietML ERROR]", e)


# # ─────────────────────────────────────────────
# # REQUEST SCHEMA
# # ─────────────────────────────────────────────
# class DietPredictRequest(BaseModel):
#     diet_type: str
#     specific_food_item: str
#     sugar_intake: str
#     dairy_intake: str
#     processed_food_freq: str
#     spicy_food_freq: str
#     oily_food_level: str
#     water_intake_liters: float
#     sleep_hours: int
#     stress_level: int = Field(..., ge=1, le=5)
#     exercise: str
#     skin_type: str


# class DietPredictResponse(BaseModel):
#     risk_level: str
#     confidence_score: float
#     main_causes: List[str]
#     recommendations: List[str]


# # ─────────────────────────────────────────────
# # FOOD MAP
# # ─────────────────────────────────────────────
# FOOD_MAP = {
#     "Red Meat": "High_Inflammation",
#     "Processed Meat": "High_Inflammation",
#     "Starchy/Potato": "High_Glycemic",
#     "Leafy Greens": "Antioxidant_Rich",
#     "Fish": "Omega3_Rich",
#     "Nuts/Seeds": "Omega3_Rich",
#     "Paneer/Dairy Rich": "Hormonal_Trigger",
#     "Eggs": "Hormonal_Trigger",
#     "Chicken": "Balanced",
# }


# def get_category(food: str):
#     return FOOD_MAP.get(food, "Balanced")


# # ─────────────────────────────────────────────
# # API ENDPOINT
# # ─────────────────────────────────────────────
# @router.post("/predict", response_model=DietPredictResponse)
# async def predict_diet(
#     body: DietPredictRequest,
#     current_user: dict = Depends(get_current_user)
# ):

#     # ❌ HARD STOP ONLY IF MODEL MISSING
#     if _model is None or _preprocessor is None or _target_enc is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Model not loaded. Please ensure model.joblib and encoders.joblib exist in backend/skin_health_backend/ml"
#         )

#     try:
#         food_cat = get_category(body.specific_food_item)

#         row = body.dict()
#         row["food_category"] = food_cat

#         df = pd.DataFrame([row])

#         X = _preprocessor.transform(df)

#         pred_idx = _model.predict(X)[0]
#         risk = _target_enc.inverse_transform([pred_idx])[0]

#         probs = _model.predict_proba(X)[0]
#         confidence = float(max(probs))

#         # ── SAFE ENGINE CALL (NO CRASH) ──
#         try:
#             engine = AcneGuardEngine()
#             explanation = await engine.generate_diet_explanation(body.dict(), risk)
#             causes = explanation.main_causes
#             recs = explanation.recommendations
#         except Exception:
#             causes = ["Diet-based ML prediction"]
#             recs = ["Maintain balanced diet, hydration, and sleep"]

#         return DietPredictResponse(
#             risk_level=risk,
#             confidence_score=round(confidence, 4),
#             main_causes=causes,
#             recommendations=recs,
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))







# """
# diet.py — Diet-based acne risk prediction router (FIXED)
# """

# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel, Field
# from typing import List
# from pathlib import Path
# import pandas as pd
# import joblib

# from auth import get_current_user
# from engine import AcneGuardEngine

# router = APIRouter(prefix="/diet", tags=["Diet Analysis"])

# # ─────────────────────────────────────────────
# # ✅ FIXED PATH (ONLY ONE SOURCE OF TRUTH)
# # ─────────────────────────────────────────────
# BASE_DIR = Path(__file__).resolve().parent

# ML_DIR = BASE_DIR / "skin_health_backend" / "ml"

# MODEL_PATH = ML_DIR / "model.joblib"
# ENCODERS_PATH = ML_DIR / "encoders.joblib"

# print("[DietML] MODEL_PATH =", MODEL_PATH)
# print("[DietML] MODEL_EXISTS =", MODEL_PATH.exists())
# print("[DietML] ENCODERS_EXISTS =", ENCODERS_PATH.exists())

# _model = None
# _preprocessor = None
# _target_enc = None

# try:
#     if MODEL_PATH.exists() and ENCODERS_PATH.exists():

#         _model = joblib.load(MODEL_PATH)
#         artifacts = joblib.load(ENCODERS_PATH)

#         _preprocessor = artifacts["preprocessor"]
#         _target_enc = artifacts["target_encoder"]

#         print("[DietML] ✅ Model loaded successfully")

#     else:
#         print("[DietML ERROR] ❌ Model files missing in ML folder")

# except Exception as e:
#     print("[DietML ERROR]", e)
#     _model = None


# # ─────────────────────────────────────────────
# # Schemas
# # ─────────────────────────────────────────────
# class DietPredictRequest(BaseModel):
#     diet_type: str
#     specific_food_item: str
#     sugar_intake: str
#     dairy_intake: str
#     processed_food_freq: str
#     spicy_food_freq: str
#     oily_food_level: str
#     water_intake_liters: float
#     sleep_hours: int
#     stress_level: int = Field(..., ge=1, le=5)
#     exercise: str
#     skin_type: str


# class DietPredictResponse(BaseModel):
#     risk_level: str
#     confidence_score: float
#     main_causes: List[str]
#     recommendations: List[str]


# # ─────────────────────────────────────────────
# # Food Mapping
# # ─────────────────────────────────────────────
# FOOD_CAT_MAP = {
#     "Red Meat": "High_Inflammation",
#     "Processed Meat": "High_Inflammation",
#     "Starchy/Potato": "High_Glycemic",
#     "Leafy Greens": "Antioxidant_Rich",
#     "Fish": "Omega3_Rich",
#     "Nuts/Seeds": "Omega3_Rich",
#     "Paneer/Dairy Rich": "Hormonal_Trigger",
#     "Eggs": "Hormonal_Trigger",
#     "Chicken": "Balanced",
# }

# def derive_category(food: str) -> str:
#     return FOOD_CAT_MAP.get(food, "Balanced")


# # ─────────────────────────────────────────────
# # API ENDPOINT
# # ─────────────────────────────────────────────
# @router.post("/predict", response_model=DietPredictResponse)
# async def predict_diet_risk(
#     body: DietPredictRequest,
#     current_user: dict = Depends(get_current_user),
# ):

#     # ❗ STOP 503 at root cause level
#     if _model is None or _preprocessor is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Diet model not loaded. Ensure model.joblib + encoders.joblib are deployed."
#         )

#     try:
#         food_cat = derive_category(body.specific_food_item)

#         row = body.dict()
#         row["food_category"] = food_cat

#         df = pd.DataFrame([row])

#         X = _preprocessor.transform(df)

#         pred_idx = _model.predict(X)[0]
#         risk = _target_enc.inverse_transform([pred_idx])[0]

#         probs = _model.predict_proba(X)[0]
#         confidence = float(max(probs))

#         engine = AcneGuardEngine()
#         explanation = await engine.generate_diet_explanation(body.dict(), risk)

#         return {
#             "risk_level": risk,
#             "confidence_score": round(confidence, 4),
#             "main_causes": explanation.main_causes,
#             "recommendations": explanation.recommendations,
#         }

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))




# """
# diet.py — Diet-based acne risk prediction router.
# Uses trained Random Forest model (skin_health_backend/ml/).
# Endpoint: POST /diet/predict (JWT required)
# """

# from pathlib import Path
# import joblib
# import pandas as pd
# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel, Field
# from typing import List

# from auth import get_current_user

# router = APIRouter(prefix="/diet", tags=["Diet Analysis"])

# # ─────────────────────────────────────────────
# # MODEL LOADING (CLEAN + SAFE)
# # ─────────────────────────────────────────────

# BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = BASE_DIR / "skin_health_backend" / "skin_health_backend" / "ml" / "model.joblib"
# ENCODERS_PATH = BASE_DIR / "skin_health_backend" / "skin_health_backend" / "ml" / "encoders.joblib"

# print("[DEBUG] MODEL_PATH =", MODEL_PATH)
# print("[DEBUG] EXISTS =", MODEL_PATH.exists())

# try:
#     _model = joblib.load(MODEL_PATH)
#     _artifacts = joblib.load(ENCODERS_PATH)

#     _preprocessor = _artifacts["preprocessor"]
#     _target_enc = _artifacts["target_encoder"]

#     print("[DietML] Model loaded successfully.")
# except Exception as e:
#     print("[DietML ERROR]", e)
#     _model = None
#     _preprocessor = None
#     _target_enc = None


# # ─────────────────────────────────────────────
# # REQUEST / RESPONSE SCHEMAS
# # ─────────────────────────────────────────────

# class DietPredictRequest(BaseModel):
#     diet_type: str
#     specific_food_item: str
#     sugar_intake: str
#     dairy_intake: str
#     processed_food_freq: str
#     spicy_food_freq: str
#     oily_food_level: str
#     water_intake_liters: float
#     sleep_hours: int
#     stress_level: int = Field(..., ge=1, le=5)
#     exercise: str
#     skin_type: str


# class DietPredictResponse(BaseModel):
#     risk_level: str
#     confidence_score: float
#     main_causes: List[str]
#     recommendations: List[str]


# # ─────────────────────────────────────────────
# # FOOD MAPPING
# # ─────────────────────────────────────────────

# _FOOD_CAT_MAP = {
#     "Red Meat": "High_Inflammation",
#     "Processed Meat": "High_Inflammation",
#     "Starchy/Potato": "High_Glycemic",
#     "Balanced Veg": "Balanced",
#     "Leafy Greens": "Antioxidant_Rich",
#     "Legumes/Pulses": "Antioxidant_Rich",
#     "Fish": "Omega3_Rich",
#     "Nuts/Seeds": "Omega3_Rich",
#     "Paneer/Dairy Rich": "Hormonal_Trigger",
#     "Eggs": "Hormonal_Trigger",
#     "Chicken": "Balanced",
# }


# def _derive_category(food: str) -> str:
#     return _FOOD_CAT_MAP.get(food, "Balanced")


# def _build_causes(data: DietPredictRequest, food_cat: str):
#     causes = []

#     if data.sugar_intake == "High":
#         causes.append("High Sugar Intake")
#     if data.processed_food_freq == "High":
#         causes.append("Processed Food")
#     if data.dairy_intake == "Daily":
#         causes.append("High Dairy Intake")
#     if data.oily_food_level == "High":
#         causes.append("Oily Food")
#     if data.stress_level >= 4:
#         causes.append("High Stress")
#     if data.sleep_hours < 6:
#         causes.append("Sleep Deprivation")
#     if data.water_intake_liters < 1.5:
#         causes.append("Low Water Intake")
#     if data.exercise == "No":
#         causes.append("Lack of Exercise")

#     if food_cat == "High_Inflammation":
#         causes.append("Inflammatory Diet")
#     if food_cat == "Hormonal_Trigger":
#         causes.append("Hormonal Triggers")
#     if food_cat == "High_Glycemic":
#         causes.append("High Glycemic Food")

#     return causes[:5]


# def _build_recommendations(data: DietPredictRequest, risk: str, food_cat: str):
#     recs = []

#     if food_cat == "High_Inflammation":
#         recs.append("Reduce red/processed meat.")
#     if food_cat == "Hormonal_Trigger":
#         recs.append("Reduce dairy intake.")
#     if food_cat == "High_Glycemic":
#         recs.append("Switch to low glycemic foods.")
#     if data.sugar_intake == "High":
#         recs.append("Reduce sugar intake.")
#     if data.water_intake_liters < 2:
#         recs.append("Increase water intake to 2.5L/day.")
#     if data.sleep_hours < 7:
#         recs.append("Improve sleep (7–8 hours).")
#     if data.exercise == "No":
#         recs.append("Start regular exercise.")

#     if risk == "Low" and not recs:
#         recs.append("Your lifestyle is healthy. Keep it up!")

#     return recs


# # ─────────────────────────────────────────────
# # ENDPOINT
# # ─────────────────────────────────────────────

# @router.post("/predict", response_model=DietPredictResponse)
# async def predict_diet_risk(
#     body: DietPredictRequest,
#     current_user: dict = Depends(get_current_user),
# ):
#     # SAFE MODE (NO CRASH)
#     if _model is None or _preprocessor is None:
#         return {
#             "risk_level": "unknown",
#             "confidence_score": 0.0,
#             "main_causes": ["Model not loaded"],
#             "recommendations": ["Backend model missing or failed to load"]
#         }

#     try:
#         food_cat = _derive_category(body.specific_food_item)

#         row = body.dict()
#         row["food_category"] = food_cat

#         df = pd.DataFrame([row])

#         X = _preprocessor.transform(df)

#         pred_idx = _model.predict(X)[0]
#         risk = _target_enc.inverse_transform([pred_idx])[0]

#         probs = _model.predict_proba(X)[0]
#         confidence = float(probs[pred_idx])

#         # SAFE LOCAL LOGIC (NO ENGINE DEPENDENCY)
#         causes = _build_causes(body, food_cat)
#         recs = _build_recommendations(body, risk, food_cat)

#         return DietPredictResponse(
#             risk_level=risk,
#             confidence_score=round(confidence, 4),
#             main_causes=causes,
#             recommendations=recs,
#         )

#     except Exception as e:
#         print("[DIET ERROR]", e)
#         raise HTTPException(status_code=500, detail="Diet prediction failed")

















# """
# diet.py — Diet-based acne risk prediction router.
# Uses the trained Random Forest model (skin_health_backend/ml/).
# Endpoint: POST /diet/predict (JWT required)
# """
# import os
# import joblib
# import pandas as pd
# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel, Field
# from typing import Optional, List
# from auth import get_current_user
# from engine import AcneGuardEngine

# router = APIRouter(prefix="/diet", tags=["Diet Analysis"])

# # ── Load model once at import time ───────────────────────────────────────────
# import os
# from pathlib import Path
# import joblib

# BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = BASE_DIR / "skin_health_backend" / "skin_health_backend" / "ml" / "model.joblib"
# ENCODERS_PATH = BASE_DIR / "skin_health_backend" / "skin_health_backend" / "ml" / "encoders.joblib"

# print("[DEBUG] MODEL_PATH =", MODEL_PATH)
# print("[DEBUG] EXISTS =", MODEL_PATH.exists())

# try:
#     _model = joblib.load(MODEL_PATH)
#     _artifacts = joblib.load(ENCODERS_PATH)

#     _preprocessor = _artifacts["preprocessor"]
#     _target_enc = _artifacts["target_encoder"]

#     print("[DietML] Model loaded successfully.")
# except Exception as e:
#     print(f"[DietML ERROR] {e}")
#     _model = None
# # BASE_DIR = os.path.abspath(
# #     os.path.join(os.path.dirname(__file__), "skin_health_backend")
# # )

# # MODEL_PATH = os.path.join(BASE_DIR, "skin_health_backend", "ml", "model.joblib")
# # ENCODERS_PATH = os.path.join(BASE_DIR, "skin_health_backend", "ml", "encoders.joblib")
# # # _BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
# # # _MODEL_PATH    = os.path.join(_BASE_DIR, "skin_health_backend", "ml", "model.joblib")
# # # _ENCODERS_PATH = os.path.join(_BASE_DIR, "skin_health_backend", "ml", "encoders.joblib")

# # try:
# #     _model         = joblib.load(_MODEL_PATH)
# #     _artifacts     = joblib.load(_ENCODERS_PATH)
# #     _preprocessor  = _artifacts["preprocessor"]
# #     _target_enc    = _artifacts["target_encoder"]
# #     _feat_names    = _artifacts["feature_names"]
# #     print("[DietML] Model loaded successfully.")
# # except Exception as e:
# #     print(f"[DietML] WARNING: Could not load model — {e}")
# #     _model = None


# # ── Schemas ──────────────────────────────────────────────────────────────────
# class DietPredictRequest(BaseModel):
#     diet_type:           str = Field(..., example="Veg")
#     specific_food_item:  str = Field(..., example="Leafy Greens")
#     sugar_intake:        str = Field(..., example="Medium")
#     dairy_intake:        str = Field(..., example="Occasional")
#     processed_food_freq: str = Field(..., example="Low")
#     spicy_food_freq:     str = Field(..., example="Low")
#     oily_food_level:     str = Field(..., example="Low")
#     water_intake_liters: float = Field(..., example=2.5)
#     sleep_hours:         int   = Field(..., example=7)
#     stress_level:        int   = Field(..., ge=1, le=5, example=2)
#     exercise:            str   = Field(..., example="Yes")
#     skin_type:           str   = Field(..., example="Normal")


# class DietPredictResponse(BaseModel):
#     risk_level:       str
#     confidence_score: float
#     main_causes:      List[str]
#     recommendations:  List[str]


# # ── Helpers ───────────────────────────────────────────────────────────────────
# _FOOD_CAT_MAP = {
#     "Red Meat":          "High_Inflammation",
#     "Processed Meat":    "High_Inflammation",
#     "Starchy/Potato":    "High_Glycemic",
#     "Balanced Veg":      "Balanced",
#     "Leafy Greens":      "Antioxidant_Rich",
#     "Legumes/Pulses":    "Antioxidant_Rich",
#     "Fish":              "Omega3_Rich",
#     "Nuts/Seeds":        "Omega3_Rich",
#     "Paneer/Dairy Rich": "Hormonal_Trigger",
#     "Eggs":              "Hormonal_Trigger",
#     "Chicken":           "Balanced",
# }


# def _derive_category(food: str) -> str:
#     return _FOOD_CAT_MAP.get(food, "Balanced")


# def _build_causes(data: DietPredictRequest, food_cat: str, feat_importances: dict) -> List[str]:
#     causes = []
#     if data.sugar_intake == "High":         causes.append("High Sugar Intake")
#     if data.processed_food_freq == "High":  causes.append("Processed Food")
#     if data.dairy_intake == "Daily":        causes.append("High Dairy Intake")
#     if data.oily_food_level == "High":      causes.append("Oily Food")
#     if data.stress_level >= 4:              causes.append("High Stress")
#     if data.sleep_hours < 6:               causes.append("Sleep Deprivation")
#     if data.water_intake_liters < 1.5:     causes.append("Low Water Intake")
#     if data.exercise == "No":               causes.append("Lack of Exercise")
#     if food_cat == "High_Inflammation":     causes.append("Inflammatory Diet")
#     if food_cat == "Hormonal_Trigger":      causes.append("Hormonal Triggers (Dairy/Eggs)")
#     if food_cat == "High_Glycemic":         causes.append("High Glycemic Food")
#     if data.skin_type == "Oily":            causes.append("Oily Skin Type")
#     return causes[:5]  # top 5 max


# def _build_recs(data: DietPredictRequest, risk: str, food_cat: str) -> List[str]:
#     recs = []
#     if food_cat == "High_Inflammation":
#         recs.append("🥩 Red/processed meat is inflammatory. Switch to fish or chicken for lower acne risk.")
#     if food_cat == "Hormonal_Trigger":
#         recs.append("🥛 Dairy and eggs can trigger hormonal acne. Try reducing intake and monitor breakouts.")
#     if food_cat == "High_Glycemic":
#         recs.append("🥔 High-glycemic foods spike insulin. Replace with leafy greens or legumes.")
#     if data.sugar_intake == "High":
#         recs.append("🍬 Reduce added sugar — it raises IGF-1 and promotes sebum production.")
#     if data.processed_food_freq == "High":
#         recs.append("🍟 Cut down on processed food. Whole foods reduce systemic inflammation.")
#     if data.spicy_food_freq == "High" and data.skin_type == "Oily":
#         recs.append("🌶️ Spicy food + oily skin is a flare-up combo. Reduce spicy meals.")
#     if data.water_intake_liters < 2.0:
#         recs.append("💧 Drink at least 2.5L of water daily — hydration helps flush toxins through skin.")
#     if data.sleep_hours < 7:
#         recs.append("😴 Aim for 7–8 hours of sleep. Poor sleep raises cortisol and worsens acne.")
#     if data.stress_level >= 4:
#         recs.append("🧘 High stress detected. Try 10 min daily meditation or light exercise to lower cortisol.")
#     if data.exercise == "No":
#         recs.append("🏃 Regular exercise (3×/week) improves blood flow and reduces stress hormones.")
#     if data.skin_type == "Oily":
#         recs.append("🧴 Use a gentle foaming cleanser twice daily to manage oil production.")
#     if risk == "Low" and not recs:
#         recs.append("✅ Your diet and lifestyle look great! Keep up the healthy habits.")
#     return recs


# # ── Endpoint ─────────────────────────────────────────────────────────────────
# @router.post("/predict", response_model=DietPredictResponse)
# async def predict_diet_risk(
#     body: DietPredictRequest,
#     current_user: dict = Depends(get_current_user),
# ):
#     """
#     Predict acne risk from diet + lifestyle inputs.
#     Returns risk level (Low/Medium/High), confidence, causes, and recommendations.
#     """
#     # if _model is None:
#     if _model is None:
#         return {
#             "risk_level": "unknown",
#             "confidence_score": 0.0,
#             "main_causes": ["Model not loaded"],
#             "recommendations": ["Backend issue: model missing on server"]
#         }

#     try:
#         food_cat = _derive_category(body.specific_food_item)

#         row = body.dict()
#         row["food_category"] = food_cat
#         df = pd.DataFrame([row])

#         X = _preprocessor.transform(df)
#         pred_idx  = _model.predict(X)[0]
#         risk      = _target_enc.inverse_transform([pred_idx])[0]
#         probs     = _model.predict_proba(X)[0]
#         confidence = float(probs[pred_idx])

#         engine = AcneGuardEngine()
#         explanation = await engine.generate_diet_explanation(body.dict(), risk)
#         causes = explanation.main_causes
#         recs = explanation.recommendations

#         return DietPredictResponse(
#             risk_level=risk,
#             confidence_score=round(confidence, 4),
#             main_causes=causes,
#             recommendations=recs,
#         )

#     except Exception as e:
#         import traceback; traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
