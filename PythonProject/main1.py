from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load Hugging Face model & tokenizer
MODEL_NAME = "medical-ner-proj/bert-medical-ner-proj"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Create NER pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

app = FastAPI()

# Input format
class PrescriptionIn(BaseModel):
    text: str

# Output format
class EntityOut(BaseModel):
    text: str
    label: str
    score: float

@app.post("/analyze")
def analyze_prescription(prescription: PrescriptionIn):
    results = ner_pipeline(prescription.text)
    entities = [
        {"text": r["word"], "label": r["entity_group"], "score": float(r["score"])}
        for r in results
    ]
    return {"entities": entities}

@app.get("/")
def root():
    return {"message": "Prescription NER backend is running"}
