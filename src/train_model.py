from transformers import AutoTokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os

def train_model(train_path, output_dir, model_name='skt/kogpt2-base-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id >= tokenizer.vocab_size:
        tokenizer.pad_token_id = tokenizer.vocab_size - 1

    model = GPT2LMHeadModel.from_pretrained(model_name)

    datasets = load_dataset('text', data_files={'train': train_path})

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128, padding='max_length')

    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)   

if __name__ == "__main__":
    train_model("data/train_RIGHT.txt", "model/RIGHT_model", model_name="skt/kogpt2-base-v2")
    train_model("data/train_LEFT.txt", "model/LEFT_model", model_name="skt/kogpt2-base-v2")
