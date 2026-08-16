# -*- coding: utf-8 -*-
"""微调前后对比：同一问题，base vs base+LoRA"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = r"D:\ai_learning\models\qwen2.5-3b-instruct"
LORA = r"D:\ai_learning\models\qwen2.5-3b-lora"

QUESTIONS = [
    "星云科技新员工的试用期是多长时间？",
    "公积金缴纳比例是多少？",
    "高铁出差的报销标准是什么？",
    "年假需要提前几天申请？",
]

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=q, device_map="auto", trust_remote_code=True)

def ask(m, question):
    msgs = [{"role": "user", "content": question}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    enc = {k: v.to(m.device) for k, v in enc.items()}
    out = m.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("======== 微调前（base） ========")
for qq in QUESTIONS:
    print("Q:", qq)
    print("A:", ask(model, qq))
    print()

print("======== 微调后（base+LoRA） ========")
model = PeftModel.from_pretrained(model, LORA)
for qq in QUESTIONS:
    print("Q:", qq)
    print("A:", ask(model, qq))
    print()
