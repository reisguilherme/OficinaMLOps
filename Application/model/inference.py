from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
from typing import Optional, Dict, Any
from model import ModelConfig, get_model_config
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load model from Hugging Face Hub"""
        logger.info(f"Loading model {self.config.model_id}...")
        
        try:
            load_kwargs = {
                "model_name": self.config.model_id,
                "max_seq_length": self.config.max_seq_length,
                "dtype": self.config.dtype,
                "load_in_4bit": self.config.load_in_4bit
            }
            
            # Add token if provided (for private models)
            if self.config.token:
                load_kwargs["token"] = self.config.token
                
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
            FastLanguageModel.for_inference(self.model)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def generate_response(
        self, 
        instruction: str, 
        input_text: str, 
        max_new_tokens: int = 128,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using the model"""
        try:
            inputs = self.tokenizer(
                [instruction + "\n" + input_text],
                return_tensors="pt",
            ).to(self.config.device)
            
            # Merge default and custom generation parameters
            gen_params = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "max_new_tokens": max_new_tokens,
            }
            if generation_params:
                gen_params.update(generation_params)
                
            text_streamer = TextStreamer(self.tokenizer)
            outputs = self.model.generate(
                **inputs,
                streamer=text_streamer,
                **gen_params
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
        
    @classmethod
    def from_model_name(cls, model_name: str, token: Optional[str] = None) -> 'ModelHandler':
        """Create ModelHandler instance from a predefined model name"""
        try:
            config = get_model_config(model_name)
            if token:
                config.token = token
            return cls(config)
        except Exception as e:
            logger.error(f"Error creating model handler: {str(e)}")
            raise