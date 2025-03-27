from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
import os

app = FastAPI()

# Load Hugging Face Token dari .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load Model dari Hugging Face
model_name = "goy2/t5-squad2-checkpoint"  # Ganti dengan nama model di Hugging Face
tokenizer = T5Tokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name, token=HF_TOKEN)

class InputText(BaseModel):
    text: str

@app.post("/generate")
def generate_text(input_data: InputText):
    input_text = input_data.text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"output": result}

@app.get("/")
def root():
    return {"message": "T5 Model API is running!"}
