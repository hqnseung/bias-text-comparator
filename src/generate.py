from transformers import AutoTokenizer, GPT2LMHeadModel

def generate_text(model_path, prompt, max_length=100):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_path)

    inputs = tokenizer(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


if __name__ == "__main__":
    try:
        print("=" * 40)
        print("보수적 우 성향 지지자".center(40))
        print("-" * 40)
        print(generate_text("model/RIGHT_model", " ").strip())
        print("=" * 40)
        print("진보적 좌 성향 지지자".center(40))
        print("-" * 40)
        print(generate_text("model/LEFT_model", " ").strip())
        print("=" * 40)
    except Exception as e:
        print("오류 발생:", e)
