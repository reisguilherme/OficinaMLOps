import requests

# List available models
models_response = requests.get("http://localhost:8000/models")
print("Available models:", models_response.json())

# Make a prediction
url = "http://localhost:8000/predict"
payload = {
            "instruction": """Superar a história da escravidão como principal marca
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
                FIORATI, G. Projeto joga luz sobre negros e revê perspectiva histórica.""",
            "input_text": """Em relação ao conhecimento sobre a formação cultural
                            brasileira, iniciativas como a descrita no texto favorecem o(a)
                                A recuperação do tradicionalismo.
                                B estímulo ao antropocentrismo.
                                C reforço do etnocentrismo.
                                D resgate do teocentrismo.
                                E crítica ao eurocentrismo""",
    "max_new_tokens": 2048,
    "selected_model": "reisguilherme/enem-llama3.1-8b", 
    "generation_params": {
        "temperature": 0.8,
        "top_p": 0.95
    }
}
response = requests.post(url, json=payload)
print(response.json())