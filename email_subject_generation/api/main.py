from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel
import nltk
nltk.download('punkt')

app = FastAPI()

class Item(BaseModel):
    model_name: str  = None
    email_content: str = None

def generate_subject(model, text):
  tokenizer = AutoTokenizer.from_pretrained(model)
  model = AutoModelForSeq2SeqLM.from_pretrained(model)

  inputs = ["provide email subject: " + text]

  inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt")
  output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=1, max_length=10)
  decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
  predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
  return predicted_title

@app.get("/")
async def get_subject(item: Item):
    if item.email_content is None:
        item.email_content = """
            Harry - I got kicked out of the system, so I'm sending this from Tom's account.
            He can fill you in on the potential deal with STEAG.
            I left my resume on your chair.
            I'll e-mail a copy when I have my home account running.
            My contact info is:
            """
    if item.model_name is None:
        item.model_name = "anukvma/bart-base-medium-email-subject-generation-v5"
    return generate_subject(item.model_name, item.email_content)