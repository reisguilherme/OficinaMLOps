# -*- coding: utf-8 -*-
"""Enem-Llama-3.1 8b+Unslothfinetuning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1edwc_nDYAbYIZHj3FbYvoorPtXigh4NH

O **Unsloth** é uma plataforma inovadora que visa facilitar o ajuste fino (fine-tuning) de modelos de linguagem de grande escala (LLMs). Seu objetivo é tornar o processo de adaptação desses modelos para tarefas específicas mais eficiente e acessível, permitindo que desenvolvedores e pesquisadores personalizem modelos pré-treinados de acordo com suas necessidades.
<div class="align-center">
  <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/u54VK8m8tk"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
  <a href="https://ko-fi.com/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Kofi button.png" width="145"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install unsloth
# # Also get the latest nightly Unsloth!
# !pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

# exemplo de modelos
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B", # iremos utiliar o Llama3.1 8B quantizado a 4 bits
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

"""Tambem utilizaremos LoRA para atualizar menos parâmetros."""

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

"""Agora usaremos um conjunto de dados do Enem, que possui 1886 questões dos mais variados assuntos. O intuto é ter um modelo especialista em questões do enem.

[NOTA] Lembre-se de adicionar o EOS_TOKEN à saída tokenizada!! Caso contrário, você terá gerações infinitas!
"""

import pandas as pd
df = pd.read_csv('/content/train.csv')

df

from datasets import Dataset

dataset = Dataset.from_pandas(df)

def create_instruction(examples):
    instructions = []
    for content, prompt, A, B, C, D, E in zip(
        examples['content'], examples['prompt'],
        examples['A'], examples['B'], examples['C'], examples['D'], examples['E']
    ):
        instruction = f"{content}\n{prompt}\n"
        instruction += f"A) {A}\n"
        instruction += f"B) {B}\n"
        instruction += f"C) {C}\n"
        instruction += f"D) {D}\n"
        instruction += f"E) {E}\n"
        instructions.append(instruction)
    return {'instruction': instructions}

dataset = dataset.map(create_instruction, batched=True)

enem_prompt = """Você é um monitor que ajuda a responder aos usuários as respostas corretas das questões, como um gabarito. Sempre ajude o usuário respondendo a alternativa correta.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Certifique-se de que 'tokenizer' está definido

def formatting_prompts_func(examples):
    texts = []
    for content, prompt, A, B, C, D, E, answer in zip(
        examples["content"], examples["prompt"], examples['A'],
        examples['B'], examples['C'], examples['D'], examples['E'], examples["answer"]
    ):
        input_text = f"{prompt}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nE) {E}"
        text = enem_prompt.format(content, input_text, answer) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

dataset['text'][0]

"""Treinamento do modelo usando nosso conjunto de dados"""

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        num_train_epochs = 5,
        #max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

"""Inferência

Você pode usar um `TextStreamer` para inferência contínua - assim você pode ver a geração token por token, em vez de esperar o tempo todo!
"""

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
     enem_prompt.format( # questão do enem 2023
        """O acesso às Práticas Corporais/Atividades Físicas
            (PC/AF) é desigual no Brasil, à semelhança de outros
            indicadores sociais e de saúde. Em geral, PC/AF prazerosas,
            diversificadas, mais afeitas ao período de lazer estão
            concentradas nas populações mais abastadas. As atividades
            físicas de deslocamento, trajetos a pé ou de bicicleta para
            estudar ou trabalhar, por exemplo, são mais frequentes
            na classe social menos favorecida. Aqui, há uma relação
            inversa e perversa entre variáveis socioeconômicas de acesso
            àsPC/AF.As maiores prevalências de inatividade física foram
            em mulheres, pessoas com 60 anos ou mais, negros, pessoas
            com autoavaliação de saúde ruim ou muito ruim, com renda
            familiar de até quatro salários mínimos por pessoa, pessoas
            que desconhecem programas públicos dePC/AFe residentes
            em áreas sem locais públicos para a prática. """, # instruction
        """O fator central que impacta a realização de práticas
            corporais/atividades físicas no tempo de lazer no Brasil é a
            A diferença entre homens e mulheres.
            B inexistência de políticas públicas.
            C diversidade de faixa etária.
            D variação de condição étnica.
            E desigualdade entre classes sociais.""", # input
        "", # output
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

"""<a name="Save"></a>
### Salvando, carregando modelos finetuned
Para salvar o modelo final como adaptadores LoRA, use `push_to_hub` do Huggingface para um salvamento online ou `save_pretrained` para um salvamento local.

**[NOTA]** Isso salva SOMENTE os adaptadores LoRA, e não o modelo completo. Para salvar em 16 bits ou GGUF, role para baixo!
"""

model.save_pretrained("enem-llama3.1") # Local saving
tokenizer.save_pretrained("enem-llama3.1")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

"""Agora, se você quiser carregar os adaptadores LoRA que acabamos de salvar para inferência, defina `False` como `True`:"""

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "enem-llama3.1",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)


inputs = tokenizer(
[
    enem_prompt.format(
        """Superar a história da escravidão como principal marca
          da trajetória do negro no país tem sido uma tônica daqueles
          que se dedicam a pesquisar as heranças de origem afro
          à cultura brasileira. A esse esforço de reconstrução
          da própria história do país, alia-se agora a criação da
          plataforma digital Ancestralidades. “A história do negro
          no Brasil vai continuar sendo contada, e cada passo que
          a gente dá para trás é um passo que a gente avança”, diz
          Márcio Black, idealizador da plataforma, sobre o estudo
          de figuras ainda encobertas pela perspectiva histórica
          imposta pelos colonizadores da América.
          FIORATI, G. Projeto joga luz sobre negros e revê perspectiva histórica.""", # instruction
        """Em relação ao conhecimento sobre a formação cultural
          brasileira, iniciativas como a descrita no texto favorecem o(a)
          A recuperação do tradicionalismo.
          B estímulo ao antropocentrismo.
          C reforço do etnocentrismo.
          D resgate do teocentrismo.
          E crítica ao eurocentrismo""", # input
        "", # output
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)