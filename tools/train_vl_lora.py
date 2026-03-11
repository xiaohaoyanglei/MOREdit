"""
LoRA fine-tuning for prompt decomposition on Qwen2.5-0.5B-Instruct
Usage: python train_lora.py
Output: /workspace/models/prompt_decompose_lora/
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/workspace/models/Qwen2.5-0.5B-Instruct"
DATA_PATH    = "/workspace/dataset/prompt_decompose_500.jsonl"
OUTPUT_DIR   = "/workspace/models/prompt_decompose_lora"

EPOCHS       = 5
BATCH_SIZE   = 8
LR           = 2e-4
MAX_LEN      = 256
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are an image editing instruction parser. "
    "Split the user's instruction into two fields: "
    "\"pointer\" (who/where to edit) and \"edit\" (what to change). "
    "Output JSON only, no explanation."
)


def make_prompt(tokenizer, user_input, assistant_output=None):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_input},
    ]
    if assistant_output:
        messages.append({"role": "assistant", "content": assistant_output})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=(assistant_output is None)
    )


class DecomposeDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                answer = json.dumps(
                    {"pointer": d["pointer"], "edit": d["edit"]},
                    ensure_ascii=False
                )
                full   = make_prompt(tokenizer, d["input"], answer)
                prefix = make_prompt(tokenizer, d["input"])
                # 修正：两次 encode 均关闭 add_special_tokens，避免单独编码时
                # 重复添加 special token 导致 prefix_len 与 full 序列错位
                prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
                self.items.append((full, prefix_len))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, prefix_len = self.items[idx]
        enc = self.tokenizer(
            text, truncation=True, max_length=MAX_LEN,
            return_tensors="pt", padding=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        labels[:prefix_len] = -100   # 只对 assistant 部分计算 loss
        return input_ids, labels


def collate_fn(tokenizer, batch):
    input_ids, labels = zip(*batch)
    max_len = max(x.size(0) for x in input_ids)
    # 修正：使用 tokenizer 的真实 pad_token_id，而非硬编码 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids_padded = torch.stack([
        torch.cat([x, torch.full((max_len - x.size(0),), pad_id)]) for x in input_ids
    ])
    labels_padded = torch.stack([
        torch.cat([x, torch.full((max_len - x.size(0),), -100)]) for x in labels
    ])
    attention_mask = (input_ids_padded != pad_id).long()
    return input_ids_padded, labels_padded, attention_mask


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=device
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = DecomposeDataset(DATA_PATH, tokenizer)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    print(f"Training {EPOCHS} epochs, {len(dataset)} samples, {len(loader)} steps/epoch")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for step, (input_ids, labels, attention_mask) in enumerate(loader, 1):
            input_ids     = input_ids.to(device)
            labels        = labels.to(device)
            attention_mask = attention_mask.to(device)

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"  Epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f}")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")

    print(f"Saving LoRA weights to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
