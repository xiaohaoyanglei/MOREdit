"""
测试 prompt 解耦 LoRA 模型效果
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "/workspace/models/Qwen2.5-0.5B-Instruct"
LORA_WEIGHTS = "/workspace/models/prompt_decompose_lora"

SYSTEM = (
    "You are an image editing instruction parser. "
    "Split the user's instruction into two fields: "
    "\"pointer\" (who/where to edit) and \"edit\" (what to change). "
    "Output JSON only, no explanation."
)

TEST_CASES = [
    "Change the outfit of the third person from the right to a blue suit.",
    "Give the tallest person in the group a red baseball cap.",
    "把左边那群人里穿深色衣服的换成白色T恤",
    "Make the woman sitting in the front row smile.",
    "右边第二个人的头发变成金色",
    "Add glasses to the person standing near the door.",
    "把最高的那个男生的夹克换成皮衣",
    "Change the hair color of the person on the far left to black.",
    "给站在中间的那个人加一条围巾",
    "Make the guy in the back row with the beard look younger.",
]

def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, LORA_WEIGHTS)
    model.eval()
    return tokenizer, model

def infer(tokenizer, model, user_input):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    tokenizer, model = load_model()
    print(f"\n{'='*60}")
    ok, fail = 0, 0
    for case in TEST_CASES:
        result = infer(tokenizer, model, case)
        try:
            parsed = json.loads(result)
            status = "✓" if "pointer" in parsed and "edit" in parsed else "△"
            if "pointer" in parsed and "edit" in parsed:
                ok += 1
            else:
                fail += 1
        except json.JSONDecodeError:
            status = "✗"
            parsed = result
            fail += 1

        print(f"[{status}] input  : {case}")
        if isinstance(parsed, dict):
            print(f"     pointer: {parsed.get('pointer','')}")
            print(f"     edit   : {parsed.get('edit','')}")
        else:
            print(f"     raw    : {parsed}")
        print()

    print(f"结果: {ok}/{len(TEST_CASES)} 条 JSON 解析成功")

if __name__ == "__main__":
    main()
