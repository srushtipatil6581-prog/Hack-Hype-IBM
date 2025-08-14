# frontend/app.py
import os
import requests
import streamlit as st
from PIL import Image
import io

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="OCR + Medical NER", layout="centered")
st.title("📄 OCR + Medical NER (Prescription)")

st.markdown("""
Upload a prescription image (jpg/png). The app will:
1. Extract text using OCR (Hugging Face TroCR if backend has HF key; otherwise local Tesseract).
2. Run medical NER (Hugging Face Inference API) on the extracted text (if HF key available).
""")

uploaded_file = st.file_uploader("Upload prescription image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # show preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze image"):
        with st.spinner("Sending to backend..."):
            try:
                # prepare multipart form
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                resp = requests.post(f" http://localhost:8000/analyze_image", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()

                st.subheader("OCR text")
                if data.get("ocr_text"):
                    st.text_area("Extracted text", value=data["ocr_text"], height=200)
                else:
                    st.info("No text extracted.")

                st.subheader("NER Entities (if available)")
                ner = data.get("ner_results", [])
                if ner:
                    for ent in ner:
                        st.write(f"- **{ent.get('text')}** — {ent.get('label')} (score: {ent.get('score'):.3f})")
                else:
                    st.write("No NER results (Hugging Face NER requires HF_API_KEY in backend).")

            except Exception as e:
                st.error(f"Error while analyzing image: {e}")
