from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from model import ModelConfig, LORA_CONFIG, TRAINING_CONFIG

def prepare_dataset(df_path: str) -> Dataset:
    df = pd.read_csv(df_path)
    dataset = Dataset.from_pandas(df)
    
    def create_instruction(examples):
        instructions = []
        for content, prompt, A, B, C, D, E in zip(
            examples['content'], examples['prompt'],
            examples['A'], examples['B'], examples['C'], 
            examples['D'], examples['E']
        ):
            instruction = f"{content}\n{prompt}\n"
            instruction += f"A) {A}\nB) {B}\nC) {C}\nD) {D}\nE) {E}\n"
            instructions.append(instruction)
        return {'instruction': instructions}

    dataset = dataset.map(create_instruction, batched=True)
    return dataset

def train_model(
    config: ModelConfig,
    dataset_path: str,
    save_path: str
):
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_path,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        **LORA_CONFIG
    )
    
    # Prepare dataset
    dataset = prepare_dataset(dataset_path)
    
    # Configure training arguments
    training_args = TrainingArguments(
        **TRAINING_CONFIG,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # Train
    trainer_stats = trainer.train()
    
    # Save model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    return trainer_stats