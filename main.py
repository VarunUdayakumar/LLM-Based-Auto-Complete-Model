from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers.models.gpt2 import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

model_path = "./gpt2-custom"
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
model.eval()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/generate/")
async def generate_text(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
    return {"output": tokenizer.decode(outputs[0], skip_special_tokens=True)}
