# -*- coding: utf-8 -*-
"""合并 LoRA 到 base，输出完整 fp16 模型（供转 GGUF 接入 Ollama）"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = r"D:\ai_learning\models\qwen2.5-3b-instruct"
LORA = r"D:\ai_learning\models\qwen2.5-3b-lora"
OUT  = r"D:\ai_learning\models\qwen2.5-3b-merged"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("已合并保存到:", OUT)
