"""
Module de génération de texte — Text-to-Text
Modèle : TinyLlama/TinyLlama-1.1B-Chat-v1.0 (HuggingFace Transformers)

TinyLlama est un modèle de langage compact (1,1 milliard de paramètres) fine-tuné
pour le dialogue. Sa petite taille le rend utilisable localement sans GPU dédié,
tout en produisant des réponses cohérentes à des questions simples.
"""

import torch
from transformers import pipeline


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Consigne système : guide le comportement du modèle
SYSTEM_PROMPT = (
    "You are a concise and helpful assistant. "
    "Answer the user's message in 2 to 3 sentences."
)


def load_llm(model_name: str = DEFAULT_MODEL):
    """
    Charge le pipeline de génération de texte.

    Le pipeline 'text-generation' gère à la fois la tokenisation,
    l'inférence et le décodage du texte.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[LLM] Chargement de '{model_name}' sur {device.upper()}...")

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )
    return pipe


def generate_response(pipe, text: str, max_new_tokens: int = 200) -> str:
    """
    Génère une réponse textuelle à partir d'une transcription.

    Le texte transcrit est transmis au modèle sous forme de message utilisateur.
    Le modèle construit sa réponse en suivant le format de chat qui lui a été
    appris lors du fine-tuning.

    Args:
        pipe           : Pipeline LLM chargé.
        text           : Texte d'entrée (issu de la transcription STT).
        max_new_tokens : Nombre maximum de tokens à générer en réponse.

    Returns:
        Réponse générée sous forme de chaîne de caractères.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]

    # apply_chat_template construit le prompt dans le format attendu par le modèle
    # (par exemple <|system|>...<|user|>...<|assistant|> pour TinyLlama)
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    # Le modèle retourne le prompt + la réponse ; on conserve uniquement la réponse
    full_text = output[0]["generated_text"]
    response = full_text[len(prompt):].strip()
    return response
