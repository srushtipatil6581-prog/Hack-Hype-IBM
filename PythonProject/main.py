# backend/main.py
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, RelationsOptions

# Load env
from dotenv import load_dotenv
load_dotenv()

HF_MODEL = os.getenv("HF_MODEL", "medical-ner-proj/bert-medical-ner-proj")
WATSON_APIKEY = os.getenv("WATSON_APIKEY")
WATSON_URL = os.getenv("WATSON_URL")

app = FastAPI(title="AI Prescription Verifier (HF NER + IBM Watson)")

# Load Hugging Face NER pipeline (token-classification)
print("Loading HF model:", HF_MODEL)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForTokenClassification.from_pretrained(HF_MODEL)
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Initialize Watson NLU client if keys provided
watson_nlu = None
if WATSON_APIKEY and WATSON_URL:
    auth = IAMAuthenticator(WATSON_APIKEY)
    watson_nlu = NaturalLanguageUnderstandingV1(
        version="2023-08-01",
        authenticator=auth
    )
    watson_nlu.set_service_url(WATSON_URL)
    print("IBM Watson NLU configured.")
else:
    print("Warning: IBM Watson credentials not found in env. Watson enrichment will be skipped.")

# Request/response models
class PrescriptionIn(BaseModel):
    text: str
    patient_age: int = None
    patient_weight_kg: float = None

class Entity(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int

class PrescriptionOut(BaseModel):
    entities: List[Entity]
    watson_entities: List[Dict[str, Any]] = []
    interactions: List[Dict[str, Any]] = []
    dosage_flags: List[str] = []

# --- Helper functions ---

def hf_extract_entities(text: str) -> List[Dict[str, Any]]:
    """Run HF NER pipeline and return normalized entities."""
    raw = ner_pipeline(text)
    # pipeline with aggregation_strategy="simple" returns items with 'entity_group'
    normalized = []
    for item in raw:
        normalized.append({
            "text": item.get("word") or item.get("entity"),
            "label": item.get("entity_group") or item.get("entity"),
            "score": float(item.get("score", 0.0)),
            "start": int(item.get("start", -1)),
            "end": int(item.get("end", -1))
        })
    return normalized

def watson_enrich(text: str) -> Dict[str, Any]:
    """Call IBM Watson NLU to extract entities/relations. Returns dict or empty dict if not configured."""
    if not watson_nlu:
        return {}
    try:
        response = watson_nlu.analyze(
            text=text,
            features=Features(
                entities=EntitiesOptions(sentiment=False, limit=50),
                relations=RelationsOptions()
            ),
            language="en"
        ).get_result()
        return response
    except Exception as e:
        # log and return empty; don't crash for Watson errors
        print("Watson NLU error:", str(e))
        return {}

# MOCK: drug interaction check - replace with Micromedex/DrugBank/openFDA in prod
def check_interaction(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Basic mock rules:
      - If combination contains 'warfarin' and 'aspirin' -> major interaction
      - If two antihypertensives listed -> monitor BP
    Replace this with a real clinical API (Micromedex / DrugBank / openFDA).
    """
    drugs_lower = [d.lower() for d in drugs]
    results = []
    if "warfarin" in drugs_lower and "aspirin" in drugs_lower:
        results.append({
            "pair": ["warfarin", "aspirin"],
            "severity": "major",
            "explanation": "Increased bleeding risk (warfarin + aspirin)."
        })
    # simple example
    antihypertensives = {"lisinopril","enalapril","losartan","amlodipine","metoprolol"}
    found_anti = [d for d in drugs_lower if d in antihypertensives]
    if len(found_anti) >= 2:
        results.append({
            "pair": found_anti[:2],
            "severity": "moderate",
            "explanation": "Multiple antihypertensives — monitor blood pressure and renal function."
        })
    return results

def flag_dosage_issues(entities: List[Dict[str, Any]], age: int = None, weight: float = None) -> List[str]:
    """
    Very simple rule-based dosage flags:
      - detect if 'mg' present with large-sounding dose for small-age child
      - If no numeric dose found for entities labeled 'DOSAGE' -> flag
    Replace/extend with clinical dosing rules or connect to clinical dosing API.
    """
    flags = []
    text_join = " ".join([e["text"] for e in entities]).lower()
    # barebones checks:
    if age and age < 12 and "mg" in text_join:
        flags.append("Child under 12 mentioned: verify pediatric dosing (some doses per kg required).")
    # check for drugs mentioned without dose
    drug_entities = [e for e in entities if e["label"].lower() in ("drug","med","medication","drug_name")]
    for ent in drug_entities:
        if not any(ch.isdigit() for ch in ent["text"]):
            flags.append(f"Drug '{ent['text']}' missing a clear numeric dose in prescription.")
    return flags

# --- API endpoints ---

@app.post("/analyze_prescription", response_model=PrescriptionOut)
def analyze_prescription(req: PrescriptionIn):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty prescription text provided.")
    # HF NER
    hf_entities = hf_extract_entities(text)
    # Watson enrichment
    watson_resp = watson_enrich(text)
    watson_entities = watson_resp.get("entities", []) if isinstance(watson_resp, dict) else []
    # Build list of drug names (simple heuristic)
    drug_names = []
    for e in hf_entities:
        label = e["label"].lower()
        if "drug" in label or "med" in label or "medicine" in label or "substance" in label or "brand" in label:
            drug_names.append(e["text"])
    # As fallback, consider Watson-detected entities that look like medications
    for we in watson_entities:
        if we.get("type","").lower() in ("drug","medication","medicine"):
            if we.get("text") not in drug_names:
                drug_names.append(we.get("text"))
    # Interaction check (mock)
    interactions = check_interaction(drug_names)
    # Dosage flags
    dosage_flags = flag_dosage_issues(hf_entities, age=req.patient_age, weight=req.patient_weight_kg)
    # Response
    out = {
        "entities": hf_entities,
        "watson_entities": watson_entities,
        "interactions": interactions,
        "dosage_flags": dosage_flags
    }
    return out

@app.get("/")
def root():
    return {"status":"ok", "info":"AI Prescription Verifier backend. POST /analyze_prescription"}

