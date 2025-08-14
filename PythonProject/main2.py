# backend/main.py
import os
import io
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "").strip() or None
OCR_MODEL = os.getenv("OCR_MODEL", "microsoft/trocr-base-handwritten")
NER_MODEL = os.getenv("NER_MODEL", "medical-ner-proj/bert-medical-ner-proj")

app = FastAPI(title="OCR + Medical NER Backend")

# Allow local streamlit frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def call_hf_ocr(image_bytes: bytes, model_name: str = OCR_MODEL) -> str:
    """Call Hugging Face Inference API for OCR (vision->text)."""
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not configured for HF OCR")
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    # send raw image bytes
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model_name}",
        headers=headers,
        data=image_bytes,
        timeout=60
    )
    resp.raise_for_status()
    result = resp.json()
    # Typical TroCR returns list with {'generated_text': '...'}
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        if "generated_text" in result[0]:
            return result[0]["generated_text"]
    # If result is dict with 'error':
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"HF OCR error: {result['error']}")
    # Fallback: stringify
    return str(result)

def call_hf_ner(text: str, model_name: str = NER_MODEL) -> List[Dict[str, Any]]:
    """Call Hugging Face Inference API for NER (token-classification)."""
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not configured for HF NER")
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": text}
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model_name}",
        headers=headers,
        json=payload,
        timeout=60
    )
    resp.raise_for_status()
    result = resp.json()
    # result normally is a list of dicts with entity info
    return result

def local_tesseract_ocr(image_bytes: bytes) -> str:
    """Run local Tesseract OCR (requires Tesseract installation)."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise RuntimeError(f"Cannot open image for Tesseract OCR: {e}")
    text = pytesseract.image_to_string(image)
    return text

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an image file (jpg/png) and returns:
      - ocr_text: extracted text (HF OCR or Tesseract fallback)
      - ner_results: list of NER items (if HF NER available)
    """
    try:
        img_bytes = await file.read()
        # 1) OCR: try HF OCR if key present; otherwise fallback to local tesseract
        if HF_API_KEY:
            try:
                ocr_text = call_hf_ocr(img_bytes)
            except Exception as e:
                # if HF OCR fails, fallback to Tesseract if available, but keep error logged
                print("HF OCR failed:", e)
                ocr_text = local_tesseract_ocr(img_bytes)
        else:
            # No HF key: local Tesseract
            ocr_text = local_tesseract_ocr(img_bytes)

        # Normalize OCR text (strip)
        ocr_text = ocr_text.strip()

        # 2) NER: call HF NER if HF_API_KEY present
        ner_results = []
        if HF_API_KEY and ocr_text:
            try:
                raw_ner = call_hf_ner(ocr_text)
                # raw_ner often a list of entity dicts; we return as-is but map keys
                # Normalize each entry to: text, label, score, start, end
                ner_results = []
                if isinstance(raw_ner, list):
                    for r in raw_ner:
                        # different HF NER models may use different keys: 'word' or 'entity'
                        text_val = r.get("word") or r.get("entity") or r.get("text") or r.get("token") or ""
                        label = r.get("entity_group") or r.get("entity") or r.get("label") or r.get("type") or ""
                        score = float(r.get("score", 0.0))
                        start = r.get("start", None)
                        end = r.get("end", None)
                        ner_results.append({
                            "text": text_val,
                            "label": label,
                            "score": score,
                            "start": start,
                            "end": end
                        })
            except Exception as e:
                print("HF NER failed:", e)
                # leave ner_results empty; don't crash
        # Response
        return {
            "ocr_text": ocr_text,
            "ner_results": ner_results
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/")
def root():
    return {"status": "ok", "info": "OCR + Medical NER backend"}
