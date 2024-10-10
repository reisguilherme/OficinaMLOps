from typing import List, Tuple
import torch
from transformers import PreTrainedTokenizer, TextStreamer

def create_enem_prompt(instruction: str, input_text: str, response: str = "") -> str:
    return f"""Você é um monitor que ajuda a responder aos usuários as respostas corretas das questões, como um gabarito. Sempre ajude o usuário respondendo a alternativa correta.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}"""

def create_instruction(content: str, prompt: str, options: List[str]) -> str:
    instruction = f"{content}\n{prompt}\n"
    for letter, option in zip(['A', 'B', 'C', 'D', 'E'], options):
        instruction += f"{letter}) {option}\n"
    return instruction

def prepare_model_input(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    input_text: str,
    device: str = "cuda"
) -> torch.Tensor:
    prompt = create_enem_prompt(instruction, input_text)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    return inputs