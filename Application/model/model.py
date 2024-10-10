from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    model_id: str = "reisguilherme/enem-llama3.1-8b"  # HF model ID
    max_seq_length: int = 2048
    dtype = None
    load_in_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    token: str = None  # Optional: HuggingFace token for private models

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

# Configuration for different model options
AVAILABLE_MODELS = {
    "reisguilherme/enem-llama3.1-8b": {
        "model_id": "reisguilherme/enem-llama3.1-8b",
        "description": "Meta Llama 3.1 8B model fine tuned for ENEM Question Answering"
    },
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration for a specific model"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_name]
    return ModelConfig(model_id=model_info["model_id"])