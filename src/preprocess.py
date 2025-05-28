import json

def save_by_bias(input_path, output_a, output_b):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_a, 'w', encoding='utf-8') as fa, open(output_b, 'w', encoding='utf-8') as fb:
        for item in data:
            if item["bias"] == "RIGHT":
                fa.write(item["text"].strip() + "\n")
            elif item["bias"] == "LEFT":
                fb.write(item["text"].strip() + "\n")

if __name__ == "__main__":
    save_by_bias("data/raw_data.json", "data/train_RIGHT.txt", "data/train_LEFT.txt")
