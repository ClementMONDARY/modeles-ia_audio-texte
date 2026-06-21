"""
Module de transcription audio — Speech-to-Text
Modèle : openai/whisper-large-v3 (HuggingFace Transformers)

Whisper est un modèle multilingue entraîné par OpenAI. La version large-v3
offre les meilleures performances en transcription et en détection automatique
de la langue parlée.
"""

import torch
from transformers import pipeline


DEFAULT_MODEL = "openai/whisper-large-v3"


def load_stt_model(model_name: str = DEFAULT_MODEL):
    """
    Charge le pipeline Whisper pour la reconnaissance vocale.

    Utilise float16 sur GPU pour accélérer l'inférence, float32 sur CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[STT] Chargement de '{model_name}' sur {device.upper()}...")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )
    return pipe


def transcribe_audio(pipe, audio_path: str) -> dict:
    """
    Transcrit un fichier audio en texte.

    L'audio est d'abord chargé en mémoire via librosa (qui utilise soundfile
    pour les WAV/FLAC, sans dépendance à ffmpeg), puis transmis au pipeline
    sous forme de tableau numpy. Whisper attend de l'audio mono à 16 kHz.

    Args:
        pipe       : Pipeline Whisper chargé.
        audio_path : Chemin vers le fichier audio (wav, flac, ogg…).

    Returns:
        Dictionnaire {"text": str, "chunks": list}.
        "text" contient la transcription complète.
        "chunks" contient les segments horodatés (si disponibles).
    """
    import librosa

    # Chargement et rééchantillonnage à 16 kHz (fréquence attendue par Whisper)
    audio_array, _ = librosa.load(audio_path, sr=16_000, mono=True)

    result = pipe(
        {"array": audio_array, "sampling_rate": 16_000},
        return_timestamps=True,
        generate_kwargs={"language": None},  # Détection automatique de la langue
    )
    return result
