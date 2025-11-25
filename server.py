from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import base64
import numpy as np
import cv2
import pytesseract
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
    thresh = cv2.adaptiveThreshold(denoise, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    return thresh

def extract_date(text):
    patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
        r"[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m: return m.group()
    return None

def extract_total(text):
    lines = text.lower().splitlines()
    for line in lines[::-1]:
        if any(k in line for k in ("total","amount","subtotal","grand")):
            nums = re.findall(r"\d+\.\d+|\d+", line)
            if nums: return nums[-1]
    return None

# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     raw = await file.read()
#     pil = Image.open(io.BytesIO(raw)).convert("RGB")

#     processed = preprocess_pil(pil)
#     processed_pil = Image.fromarray(processed)

#     text = pytesseract.image_to_string(processed_pil, lang="eng", config="--oem 3 --psm 6")

#     date = extract_date(text)
#     total = extract_total(text)

#     _, buff = cv2.imencode(".png", processed)
#     processed_b64 = "data:image/png;base64," + base64.b64encode(buff).decode()

#     return {
#         "filename": file.filename,
#         "text": text,
#         "date": date,
#         "total": total,
#         "processed_image": processed_b64
#     }
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    return {"status": "OK", "size": len(raw)}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

