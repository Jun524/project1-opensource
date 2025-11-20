from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# 로컬 모델 다운로드 & 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def parse_user_text(user_text):
    prompt = f"""
    다음 문장에서 성별(gender), 계절(season), 스타일(style), 가격대(price_range)를 JSON으로 추출해줘.

    반드시 아래 형식으로 출력해줘:

    {{
        "gender": "...",
        "season": "...",
        "style": "...",
        "price_range": "..."
    }}

    문장: "{user_text}"
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # JSON 부분만 추출
    start = text.find("{")
    end = text.rfind("}")

    json_text = text[start:end+1]

    return json.loads(json_text)
