from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
import os
import wikipedia
from deep_translator import GoogleTranslator

app = FastAPI()

# Load Hugging Face Token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load Model from Hugging Face
model_name = "goy2/t5-squad2-checkpoint"
try:
    tokenizer = T5Tokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = T5ForConditionalGeneration.from_pretrained(model_name, token=HF_TOKEN)
except Exception as e:
    tokenizer, model = None, None

class QuestionInput(BaseModel):
    question: str

def translate_to_english(text: str) -> str:
    """Translate text to English if necessary."""
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except:
        return text  # Jika gagal, gunakan teks asli

def search_context(query: str) -> str:
    """Search for context in Wikipedia and translate to English."""
    try:
        search_results = wikipedia.search(query)
        if search_results:
            summary = wikipedia.summary(search_results[0], sentences=5)
        else:
            return "Error: No relevant Wikipedia article found."
    except wikipedia.exceptions.DisambiguationError as e:
        summary = wikipedia.summary(e.options[0], sentences=5)
    except wikipedia.exceptions.PageError:
        return "Error: No relevant Wikipedia article found."
    except:
        return "Error: An unexpected error occurred."

    print(f"Original Summary: {summary}")  # Debugging print
    print(f"Translated Summary: {GoogleTranslator(source='auto', target='en').translate(summary)}")  # Debugging print

    return GoogleTranslator(source="auto", target="en").translate(summary)

@app.post("/generate")
def generate_answer(input_data: QuestionInput):
    if tokenizer is None or model is None:
        return {"error": "Model failed to load."}
    
    translated_question = translate_to_english(input_data.question)
    context = search_context(translated_question)
    
    if "Error" in context:
        return {"output": "Sorry, I couldn't find relevant information."}
    
    input_text = f"answer the question: {translated_question} based on this context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "original_question": input_data.question,
        "translated_question": translated_question,
        "context": context,
        "model_input": input_text,
        "output": result
    }

@app.get("/")
def root():
    return {"message": "T5 Model API is running!"}
