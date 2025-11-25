from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import numpy as np
import cv2
import io
import base64
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_pil(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return thresh


def extract_date(text):
    patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
        r"[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}"
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            return match.group()
    return None


def extract_total(text):
    lines = text.lower().splitlines()[::-1]
    for line in lines:
        if any(k in line for k in ("total", "amount", "subtotal", "grand")):
            nums = re.findall(r"\d+\.\d+|\d+", line)
            if nums:
                return nums[-1]
    return None


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    processed = preprocess_pil(pil)
    processed_pil = Image.fromarray(processed)

    text = pytesseract.image_to_string(processed_pil, lang="eng")

    date = extract_date(text)
    total = extract_total(text)

    _, buff = cv2.imencode(".png", processed)
    processed_b64 = "data:image/png;base64," + base64.b64encode(buff).decode()

    return {
        "text": text,
        "date": date,
        "total": total,
        "processed_image": processed_b64
    }
