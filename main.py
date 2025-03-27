import os
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

# Google Drive File IDs
MODEL_ID = "16JvfwZGglWG9h_freKo2Ep6CzKUoH-B-"  # Ganti dengan File ID model kamu

# Folder penyimpanan model
MODEL_PATH = "models/t5-model"
TOKENIZER_PATH = "token/"

# Pastikan folder models ada
os.makedirs("models", exist_ok=True)

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# Load model dan tokenizer
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)

class QuestionRequest(BaseModel):
    question: str
    context: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    input_text = f"question: {request.question} context: {request.context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"question": request.question, "answer": answer}

@app.get("/")
def root():
    return {"message": "T5 Model API is running!"}
    