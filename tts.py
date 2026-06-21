"""
Module de synthèse vocale — Text-to-Speech
Modèle : canopylabs/orpheus-3b-0.1-ft + codec hubertsiuzdak/snac_24khz

Orpheus est un LLM (basé sur Llama 3.1) fine-tuné pour générer des tokens audio
plutôt que du texte. Ces tokens audio suivent la structure hiérarchique du codec
SNAC (Simple Non-Autoregressive Codec), qui les décode en signal audio à 24 kHz.

Architecture SNAC — 3 niveaux de résolution temporelle :
  Niveau 0 : 1 code  / groupe  → résolution la plus basse (structure globale)
  Niveau 1 : 2 codes / groupe  → résolution intermédiaire
  Niveau 2 : 4 codes / groupe  → haute résolution (détails acoustiques fins)
  ─────────────────────────────────────────────────
  Total     : 7 tokens / groupe de frame
"""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC


ORPHEUS_MODEL = "canopylabs/orpheus-3b-0.1-ft"
SNAC_MODEL    = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE   = 24_000          # Fréquence d'échantillonnage du codec SNAC

# Vocabulaire étendu d'Orpheus (sur la base de Llama 3.1 — vocab 128 256 tokens)
# Les tokens de codec audio débutent à partir de cet ID.
AUDIO_CODE_OFFSET = 128_266
SNAC_VOCAB_SIZE   = 4_096       # Chaque niveau du codec possède 4 096 codes possibles

# Voix disponibles dans le modèle fine-tuné
AVAILABLE_VOICES = ["tara", "leo", "luna", "stella", "ryan", "jess"]


def load_tts_model():
    """
    Charge le modèle Orpheus et le décodeur SNAC.

    Returns:
        tuple : (tokenizer, model, snac_model, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[TTS] Chargement d'Orpheus ({ORPHEUS_MODEL}) sur {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        ORPHEUS_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    print(f"[TTS] Chargement du codec SNAC ({SNAC_MODEL})...")
    snac_model = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    return tokenizer, model, snac_model, device


def _decode_snac_tokens(audio_tokens: list, snac_model, device) -> np.ndarray | None:
    """
    Convertit les tokens audio générés par Orpheus en signal audio via SNAC.

    Orpheus génère les tokens dans l'ordre suivant pour chaque groupe de 7 :
      [L0_0, L1_0, L2_0, L2_1, L1_1, L2_2, L2_3]

    Ce pattern entrelacé est ensuite réorganisé en 3 tenseurs séparés
    (un par niveau) pour être décodé par SNAC.

    Args:
        audio_tokens : Liste de tokens audio bruts (avec AUDIO_CODE_OFFSET déjà soustrait).
        snac_model   : Modèle SNAC pour le décodage.
        device       : Device PyTorch.

    Returns:
        Signal audio numpy (float32) à 24 kHz, ou None si trop peu de tokens.
    """
    if len(audio_tokens) < 7:
        return None

    # On ne traite que des groupes complets de 7 tokens
    n = len(audio_tokens) - (len(audio_tokens) % 7)
    tokens = audio_tokens[:n]

    codes_0, codes_1, codes_2 = [], [], []

    for i in range(0, n, 7):
        g = tokens[i : i + 7]

        # Niveau 0 : 1 code, plage [0, 4095]
        codes_0.append(g[0])

        # Niveau 1 : 2 codes, plage [4096, 8191] → on soustrait SNAC_VOCAB_SIZE
        codes_1.append(g[1] - SNAC_VOCAB_SIZE)
        codes_1.append(g[4] - SNAC_VOCAB_SIZE)

        # Niveau 2 : 4 codes, plage [8192, 12287] → on soustrait 2 × SNAC_VOCAB_SIZE
        codes_2.append(g[2] - 2 * SNAC_VOCAB_SIZE)
        codes_2.append(g[3] - 2 * SNAC_VOCAB_SIZE)
        codes_2.append(g[5] - 2 * SNAC_VOCAB_SIZE)
        codes_2.append(g[6] - 2 * SNAC_VOCAB_SIZE)

    codes = [
        torch.tensor(codes_0, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(codes_1, dtype=torch.long).unsqueeze(0).to(device),
        torch.tensor(codes_2, dtype=torch.long).unsqueeze(0).to(device),
    ]

    with torch.no_grad():
        audio = snac_model.decode(codes)

    return audio.squeeze().cpu().float().numpy()


def generate_speech(
    tokenizer,
    model,
    snac_model,
    device,
    text: str,
    voice: str = "tara",
    output_path: str = "output.wav",
) -> str | None:
    """
    Génère un fichier audio WAV à partir d'un texte.

    Orpheus utilise un prompt spécial qui encode la voix choisie et le texte
    à synthétiser. Le modèle génère alors une séquence de tokens audio que
    le codec SNAC décode en signal audio.

    Args:
        tokenizer   : Tokenizer Orpheus.
        model       : Modèle Orpheus.
        snac_model  : Décodeur SNAC.
        device      : Device PyTorch.
        text        : Texte à synthétiser (en anglais de préférence).
        voice       : Identifiant de voix parmi AVAILABLE_VOICES.
        output_path : Chemin du fichier WAV de sortie.

    Returns:
        Chemin du fichier généré, ou None en cas d'échec.
    """
    if voice not in AVAILABLE_VOICES:
        print(f"[TTS] Voix '{voice}' inconnue. Voix disponibles : {AVAILABLE_VOICES}")
        voice = "tara"

    # Format de prompt attendu par le modèle fine-tuné Orpheus
    prompt = f"<custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    preview = text[:60] + ("..." if len(text) > 60 else "")
    print(f"[TTS] Synthèse vocale — voix '{voice}' : \"{preview}\"")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1_200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=128_258,   # Token de fin de séquence audio dans Orpheus
        )

    # On ne conserve que les tokens nouvellement générés (hors prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :].tolist()

    # Filtre : seuls les IDs >= AUDIO_CODE_OFFSET correspondent à des codes audio
    audio_tokens = [t - AUDIO_CODE_OFFSET for t in new_tokens if t >= AUDIO_CODE_OFFSET]

    audio = _decode_snac_tokens(audio_tokens, snac_model, device)

    if audio is None:
        print("[TTS] Erreur : trop peu de tokens audio générés.")
        return None

    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"[TTS] Fichier audio sauvegardé : {output_path}")
    return output_path
