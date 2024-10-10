import gradio as gr
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_URL = "http://modelo.conectaceia.com"

def make_prediction(texto_apoio: str, pergunta: str) -> str:
    """Make a prediction using the API"""
    try:
        # Prepare the payload
        payload = {
            "instruction": texto_apoio,
            "input_text": pergunta,
            "max_new_tokens": 2048,
            "selected_model": "reisguilherme/enem-llama3.1-8b",
            "generation_params": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

        # Make the request
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        
        # Get the response
        result = response.json()
        return result["response"]

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making prediction: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            error_msg = f"Error: {e.response.status_code} - {e.response.text}"
        else:
            error_msg = f"Error: {str(e)}"
        return error_msg
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.Textbox(
            label="Texto de apoio",
            placeholder="Cole aqui o texto motivador da questão.",
            lines=5
        ),
        gr.Textbox(
            label="Pergunta e alternativas",
            placeholder="Cole aqui a pergunta e as alternativas.",
            lines=5
        )
    ],
    outputs=gr.Textbox(label="Resposta", lines=3),
    title="ENEM Question Answering",
    description="Insira o texto de apoio e a pergunta com as alternativas para obter a resposta.",
    examples=[
        [
            """O acesso às Práticas Corporais/Atividades Físicas (PC/AF) é desigual no Brasil, à semelhança de outros
            indicadores sociais e de saúde. Em geral, PC/AF prazerosas, diversificadas, mais afeitas ao período de lazer estão
            concentradas nas populações mais abastadas.""",
            
            """O fator central que impacta a realização de práticas corporais/atividades físicas no tempo de lazer no Brasil é a:
            A) diferença entre homens e mulheres.
            B) inexistência de políticas públicas.
            C) diversidade de faixa etária.
            D) variação de condição étnica.
            E) desigualdade entre classes sociais."""
        ]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # Check if API is available
    try:
        health_check = requests.get(f"{API_URL}/health")
        health_check.raise_for_status()
        logger.info("API is available and healthy")
    except requests.exceptions.RequestException as e:
        logger.error(f"API is not available: {str(e)}")
        print("Warning: API is not available. Make sure the API server is running at", API_URL)
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )