# -*- coding: utf-8 -*-
"""
第8周：qwen2.5-3b QLoRA 指令微调（员工手册制度问答）
- 4bit bitsandbytes + LoRA（peft）
- transformers 原生 Trainer（适配 5.15）
- 只对 assistant 回答部分计算 loss
用法: python train_lora.py
"""
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig,
    Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------ 配置（按需改） ------------------
MODEL_PATH = r"D:\ai_learning\models\qwen2.5-3b-instruct"
DATA_PATH  = r"D:\ai_learning\data\staff_qa_dataset.json"
OUTPUT_DIR = r"D:\ai_learning\models\qwen2.5-3b-lora"
MAX_SEQ_LEN = 1024
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EPOCHS = 3
LR = 2e-4
BATCH = 1
GRAD_ACC = 4

tokenizer = None  # main() 中赋值


def _ids(x):
    # apply_chat_template(tokenize=True) 在 transformers 5.x 返回 BatchEncoding(Mapping)
    ids = getattr(x, "ids", None)
    if ids is None and hasattr(x, "get"):
        ids = x.get("input_ids")
    return list(ids)


def format_and_tokenize(ex):
    user_msgs = [{"role": "user", "content": ex["instruction"]}]
    all_msgs = user_msgs + [{"role": "assistant", "content": ex["output"]}]
    prompt_ids = _ids(tokenizer.apply_chat_template(user_msgs, tokenize=True, add_generation_prompt=True))
    full_ids = _ids(tokenizer.apply_chat_template(all_msgs, tokenize=True, add_generation_prompt=False))
    if len(full_ids) >= len(prompt_ids) and full_ids[:len(prompt_ids)] == prompt_ids:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    else:
        labels = list(full_ids)
    input_ids = full_ids[:MAX_SEQ_LEN]
    labels = labels[:MAX_SEQ_LEN]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def main():
    global tokenizer
    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA 不可用"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=quant_config, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data = json.load(open(DATA_PATH, encoding="utf-8"))
    ds = Dataset.from_list(data).map(format_and_tokenize, remove_columns=["instruction", "input", "output"])
    split = ds.train_test_split(test_size=0.15, seed=42)

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("LoRA 已保存到:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
