from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from inference import ModelHandler
from model import ModelConfig, AVAILABLE_MODELS
import uvicorn

app = FastAPI(
    title="ENEM Question Answering API",
    description="API for answering ENEM questions using various a fine tuned Llama 3.1"
)

class QuestionRequest(BaseModel):
    instruction: str = Field(..., description="The question context and instruction")
    input_text: str = Field(..., description="The actual question and options")
    max_new_tokens: int = Field(default=128, description="Maximum number of tokens to generate")
    model_name: str = Field(
        default="reisguilherme/enem-llama3.1-8b", 
        description="Model to use for generation"
    )
    generation_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional generation parameters (temperature, top_p, etc.)"
    )
    model_config = ConfigDict(protected_namespaces=())

# Model handlers cache
model_handlers = {}

def get_model_handler(model_name: str):
    """Get or create model handler for the specified model"""
    if model_name not in model_handlers:
        try:
            model_handlers[model_name] = ModelHandler.from_model_name(model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return model_handlers[model_name]

@app.post("/predict")
async def predict(request: QuestionRequest):
    try:
        model_handler = get_model_handler(request.model_name)
        response = model_handler.generate_response(
            instruction=request.instruction,
            input_text=request.input_text,
            max_new_tokens=request.max_new_tokens,
            generation_params=request.generation_params
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all available models"""
    return {"models": ["reisguilherme/enem-llama3.1-8b"]}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)