from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel
import nltk
nltk.download('punkt')

app = FastAPI()

class Item(BaseModel):
    model_name: str  = None
    email_content: str = None

def generate_answer(model, text):
  tokenizer = AutoTokenizer.from_pretrained(model)
  model = AutoModelForSeq2SeqLM.from_pretrained(model)

  inputs = ["Answer this AIML Question: " + text]

  inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt")
  output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=1, max_length=256)
  decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
  predicted_answer = nltk.sent_tokenize(decoded_output.strip())[0]
  return predicted_answer

@app.get("/")
async def get_answer(item: Item):
    if item.question is None:
        item.question = """
            what is linear regression?
            """
    if item.model_name is None:
        item.model_name = "anukvma/bart-aiml-question-answer-v2"
    return generate_answer(item.model_name, item.question)