"""
Vocalis Configuration Module

Loads and provides access to configuration settings from environment variables
and the .env file.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# API Endpoints
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")
TTS_API_ENDPOINT = os.getenv("TTS_API_ENDPOINT", "http://localhost:5005/v1/audio/speech")

# ASR (Whisper) Configuration
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "openai/whisper-large-v3") # Hardcoding to large-v3
ASR_DEVICE = os.getenv("ASR_DEVICE", "auto")  # "auto", "cuda", "cpu"
ASR_TORCH_DTYPE = os.getenv("ASR_TORCH_DTYPE", "auto") # "auto", "float16", "bfloat16", "float32"
# These will be used by the transcriber for its fixed German transcription task
ASR_LANGUAGE = "de"
ASR_TASK = "transcribe"
#WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")

# TTS Configuration
TTS_MODEL = os.getenv("TTS_MODEL", "orpheus-3b-german-ft-q8_0")
TTS_VOICE = os.getenv("TTS_VOICE", "jana")
TTS_FORMAT = os.getenv("TTS_FORMAT", "wav")

# WebSocket Server Configuration
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8000))

# Audio Processing
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", 0.5))
VAD_BUFFER_SIZE = int(os.getenv("VAD_BUFFER_SIZE", 30))
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))

def get_config() -> Dict[str, Any]:
    """
    Returns all configuration settings as a dictionary.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration settings
    """
    return {
        "llm_api_endpoint": LLM_API_ENDPOINT,
        "tts_api_endpoint": TTS_API_ENDPOINT,
        "asr_model_name": ASR_MODEL_NAME,
        "asr_device": ASR_DEVICE,
        "asr_torch_dtype": ASR_TORCH_DTYPE,
        "asr_language": ASR_LANGUAGE, # Fixed to German
        "asr_task": ASR_TASK,         # Fixed to transcribe
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
        "tts_format": TTS_FORMAT,
        "websocket_host": WEBSOCKET_HOST,
        "websocket_port": WEBSOCKET_PORT,
        "vad_threshold": VAD_THRESHOLD,
        "vad_buffer_size": VAD_BUFFER_SIZE,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
    }
