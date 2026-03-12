"""
用 Gemini API 批量生成 prompt 解耦数据集（专注对人的编辑）
输出格式: JSONL，每行 {"input": "...", "pointer": "...", "edit": "..."}
"""

import json
import os
import random
import time
import google.generativeai as genai

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OUTPUT_RAW = "/workspace/dataset/raw_1000.jsonl"
OUTPUT_FILE = "/workspace/dataset/prompt_decompose_500.jsonl"
TARGET_COUNT = 1000
FINAL_COUNT = 500

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

POINTER_TYPES = [
    "绝对位置（如：最左边的人、最右边的人、中间的人）",
    "序数+方向（如：从左数第二个人、右边第三个人）",
    "外貌特征（如：穿红衣服的人、戴眼镜的、最高的那个）",
    "分组+位置（如：左边一群人中最高的、右侧两人中穿深色的那个）",
    "空间聚类（如：左边那一群人、站在一起的几个人、右侧的人群）",
    "前后遮挡关系（如：站在最前面的人、后排中间的人）",
    "与场景元素的相对位置（如：站在门口的人、坐在椅子上的那个）",
    "多人组合（如：最左边的两个人、中间三个人）",
]

EDIT_TYPES = [
    "服装替换（如：换成西装、改成运动服、穿上旗袍）",
    "颜色变换（如：衣服变成蓝色、裤子改成黑色）",
    "配件添加（如：戴上帽子、加上眼镜、围上围巾）",
    "发型/发色（如：改成短发、头发变成金色）",
    "表情变换（如：改成微笑、换成严肃表情）",
    "姿态调整（如：改成侧身站立、换成坐姿）",
    "风格迁移（如：改成古装风格、换成赛博朋克造型）",
    "年龄/外观（如：变成老年人形象、改成儿童外貌）",
]

LANGUAGES = ["纯英文", "纯英文", "纯中文", "中英混合"]


SYSTEM_PROMPT = """你是一个数据集生成专家。请生成图像编辑指令数据，格式严格为 JSON 数组。

场景背景：图像中包含多个人物，用户用自然语言描述要编辑哪个人以及怎么编辑。

每条数据包含：
- input: 用户输入的自然语言编辑指令（一句话，口语化）
- pointer: 从 input 中提取的"目标人物定位"描述（简洁，用于图像中定位目标人物）
- edit: 从 input 中提取的"具体编辑操作"描述（清晰完整，用于描述如何修改）

要求：
1. input 必须是自然流畅的一句话
2. pointer 只描述人物定位信息，不包含编辑操作
3. edit 只描述编辑操作，不重复位置信息
4. 生成 {count} 条，覆盖以下场景：
   - 人物指代方式: {pointer_type}
   - 编辑类型: {edit_type}
   - 语言风格: {language}
5. 只输出 JSON 数组，不要任何其他文字

示例格式：
[
  {{"input": "把右边第三个人的衣服换成蓝色西装", "pointer": "右边第三个人", "edit": "将衣服换成蓝色西装，保持其他不变"}},
  {{"input": "左边那群人里穿深色衣服的那个换成红色卫衣", "pointer": "左边人群中穿深色衣服的人", "edit": "将衣服换成红色卫衣"}},
  {{"input": "给站在最前面的人戴上一顶棒球帽", "pointer": "站在最前面的人", "edit": "添加一顶棒球帽"}}
]"""


def generate_batch(pointer_type, edit_type, language, count=10):
    prompt = SYSTEM_PROMPT.format(
        pointer_type=pointer_type,
        edit_type=edit_type,
        language=language,
        count=count,
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return [d for d in data if all(k in d for k in ("input", "pointer", "edit"))]
    except Exception as e:
        print(f"  [错误] {e}")
        return []


FILTER_PROMPT = """你是数据质量审核员。请对以下图像编辑指令解耦数据进行质量评分。

评分标准（每项 0-10 分，取平均）：
1. input 是否自然流畅、贴近真实用户输入
2. pointer 是否准确提取了人物定位信息（且不包含编辑操作）
3. edit 是否完整描述了编辑操作（且不重复位置信息）
4. pointer + edit 合并后是否能完整还原 input 的语义

只输出一个 JSON 数组，每条包含原始数据 + score 字段（0-10 的浮点数），不要其他文字：
[
  {{"input": "...", "pointer": "...", "edit": "...", "score": 8.5}},
  ...
]

待评分数据：
{data}"""


def filter_by_quality(all_data, final_count):
    print(f"\n开始质量过滤，从 {len(all_data)} 条中筛选 {final_count} 条...")
    scored = []
    batch_size = 20

    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        data_str = json.dumps(batch, ensure_ascii=False, indent=2)
        prompt = FILTER_PROMPT.format(data=data_str)

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            results = json.loads(text)
            scored.extend(results)
            print(f"  已评分 {min(i + batch_size, len(all_data))}/{len(all_data)} 条")
        except Exception as e:
            print(f"  [评分错误] {e}，该批次使用默认分数 5.0")
            for item in batch:
                item["score"] = 5.0
                scored.append(item)

        time.sleep(1)

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    top = scored[:final_count]

    for item in top:
        item.pop("score", None)

    return top


def main():
    all_data = []
    seen_inputs = set()
    batches_needed = TARGET_COUNT // 8 + 5  # 多几批防止去重损耗

    print(f"=== 阶段1：生成 {TARGET_COUNT} 条原始数据 ===")
    for i in range(batches_needed):
        if len(all_data) >= TARGET_COUNT:
            break

        pointer_type = random.choice(POINTER_TYPES)
        edit_type = random.choice(EDIT_TYPES)
        language = random.choice(LANGUAGES)

        print(f"[{i+1}] 指代: {pointer_type[:14]}... | 编辑: {edit_type[:10]}...")

        batch = generate_batch(pointer_type, edit_type, language, count=10)

        new_count = 0
        for item in batch:
            if item["input"] not in seen_inputs:
                seen_inputs.add(item["input"])
                all_data.append(item)
                new_count += 1

        print(f"  +{new_count} 条，累计 {len(all_data)} 条")
        time.sleep(1)

    # 保存原始数据
    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n原始数据已保存至 {OUTPUT_RAW}（共 {len(all_data)} 条）")

    # 质量过滤
    print(f"\n=== 阶段2：质量过滤，保留 {FINAL_COUNT} 条 ===")
    final_data = filter_by_quality(all_data, FINAL_COUNT)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n=== 完成 ===")
    print(f"原始数据: {OUTPUT_RAW}（{len(all_data)} 条）")
    print(f"过滤后数据: {OUTPUT_FILE}（{len(final_data)} 条）")


if __name__ == "__main__":
    main()
