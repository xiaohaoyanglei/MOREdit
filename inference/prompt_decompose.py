"""
Prompt decomposition module for MOREdit.
Splits a combined user prompt into pointer (who) + edit (what) using a LoRA-tuned Qwen2.5-0.5B.
"""

from __future__ import annotations
import json
import torch

_SYSTEM = (
    "You are an image editing instruction parser. "
    "Split the user's instruction into two fields: "
    '"pointer" (who/where to edit) and "edit" (what to change). '
    "Output JSON only, no explanation."
)


class PromptDecomposer:
    def __init__(self, base_model: str, lora_weights: str, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print(f"[decompose] Loading prompt decomposer from {lora_weights}")
        self.tokenizer = AutoTokenizer.from_pretrained(lora_weights)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.float16, device_map=device
        )
        self.model = PeftModel.from_pretrained(base, lora_weights)
        self.model.eval()
        self.device = device

    def decompose(self, prompt: str) -> tuple[str, str]:
        """Returns (pointer_prompt, edit_prompt). Falls back to (prompt, prompt) on failure."""
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        result = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        try:
            d = json.loads(result)
            pointer = d.get("pointer", "").strip()
            edit = d.get("edit", "").strip()
            if pointer and edit:
                print(f"[decompose] pointer: {pointer!r}")
                print(f"[decompose] edit:    {edit!r}")
                return pointer, edit
        except json.JSONDecodeError:
            pass
        print(f"[decompose] Warning: failed to parse output {result!r}, falling back to original prompt")
        return prompt, prompt

    def offload(self):
        """Move model to CPU to free GPU memory before loading the main pipeline."""
        self.model.to("cpu")
        torch.cuda.empty_cache()
