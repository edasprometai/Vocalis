"""
Vocalis Backend Server

FastAPI application entry point.
"""
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from . import config
from .services.transcription import WhisperTranscriber
from .services.llm import LLMClient
from .services.tts import TTSClient
from .services.vision import vision_service
from .routes.websocket import websocket_endpoint
import os
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

transcription_service: Optional[WhisperTranscriber] = None
llm_service: Optional[LLMClient] = None
tts_service: Optional[TTSClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = config.get_config()
    logger.info("Initializing services...")
    global transcription_service, llm_service, tts_service
    
    transcription_service = WhisperTranscriber(
        model_name=cfg.get("asr_model_name"),
        device=cfg.get("asr_device"),
        compute_type=cfg.get("asr_torch_dtype"), # This maps to torch_dtype_str in WhisperTranscriber
        sample_rate=cfg.get("audio_sample_rate")
    )
    
    llm_service = LLMClient(api_endpoint=cfg["llm_api_endpoint"])
    tts_service = TTSClient(
        api_endpoint=cfg["tts_api_endpoint"],
        model=cfg["tts_model"],
        voice=cfg["tts_voice"],
        output_format=cfg["tts_format"]
    )
    
    logger.info("Initializing vision service but not loading model.")
    #if not vision_service.is_ready():
        #vision_service.initialize()
        #print("notloading")
    logger.info("All services initialized successfully")
    yield
    logger.info("Shutting down services...")
    logger.info("Shutdown complete")

app = FastAPI(
    title="Vocalis Backend",
    description="Speech-to-Speech AI Assistant Backend",
    version="1.5.2",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def get_transcription_service() -> WhisperTranscriber:
    if not transcription_service: raise HTTPException(status_code=503, detail="Transcription service not available")
    return transcription_service

def get_llm_service() -> LLMClient:
    if not llm_service: raise HTTPException(status_code=503, detail="LLM service not available")
    return llm_service

def get_tts_service() -> TTSClient:
    if not tts_service: raise HTTPException(status_code=503, detail="TTS service not available")
    return tts_service

@app.get("/")
async def root(): return {"status": "ok", "message": "Vocalis backend is running"}

@app.get("/health")
async def health_check(ts: WhisperTranscriber = Depends(get_transcription_service)):
    return {
        "status": "ok",
        "services": {
            "transcription": ts is not None and ts.model is not None,
            "llm": llm_service is not None,
            "tts": tts_service is not None,
            "vision": vision_service.is_ready()
        },
        "config": {
            "asr_model_name": config.ASR_MODEL_NAME,
            "asr_language_fixed": config.ASR_LANGUAGE,
            "asr_task_fixed": config.ASR_TASK,
            "tts_voice": config.TTS_VOICE,
            "websocket_port": config.WEBSOCKET_PORT
        }
    }

@app.get("/config")
async def get_full_config_endpoint(
    ts: WhisperTranscriber = Depends(get_transcription_service),
    ls: LLMClient = Depends(get_llm_service),
    tts_cli: TTSClient = Depends(get_tts_service)
):
    return {
        "transcription": ts.get_config(),
        "llm": ls.get_config(),
        "tts": tts_cli.get_config(),
        "vision": vision_service.model_name if vision_service.is_ready() else "Not initialized",
        "system": config.get_config()
    }

@app.websocket("/ws")
async def websocket_route_endpoint(
    websocket: WebSocket,
    transcriber: WhisperTranscriber = Depends(get_transcription_service),
    llm_client: LLMClient = Depends(get_llm_service),
    tts_client: TTSClient = Depends(get_tts_service)
):
    await websocket_endpoint(websocket, transcriber, llm_client, tts_client)

if __name__ == "__main__":
    # Define paths to your SSL certificate and key files
    # Assumes key.pem and cert.pem are in the same directory as this main.py (i.e., backend/)
    # You should generate these using mkcert as previously described.
    keyfile_path = os.path.join(os.path.dirname(__file__), "key.pem")
    certfile_path = os.path.join(os.path.dirname(__file__), "cert.pem")

    logger.info("Vocalis Backend Starting...")
    if os.path.exists(keyfile_path) and os.path.exists(certfile_path):
        logger.info(f"Found SSL certs. Attempting to start Uvicorn with HTTPS on port {config.WEBSOCKET_PORT}.")
        logger.info(f"Keyfile: {os.path.abspath(keyfile_path)}")
        logger.info(f"Certfile: {os.path.abspath(certfile_path)}")
        uvicorn.run(
            "backend.main:app", # Point to the FastAPI app instance
            host=config.WEBSOCKET_HOST, # Should be "0.0.0.0"
            port=config.WEBSOCKET_PORT,
            reload=True, # Be cautious with reload if SSL certs are dynamically generated or change
            ssl_keyfile=keyfile_path,
            ssl_certfile=certfile_path
        )
    else:
        logger.warning("SSL key file or certificate file not found in the backend directory.")
        logger.warning(f"Expected keyfile at: {os.path.abspath(keyfile_path)}")
        logger.warning(f"Expected certfile at: {os.path.abspath(certfile_path)}")
        logger.warning("Uvicorn will start WITHOUT HTTPS. Microphone access from network devices will fail.")
        uvicorn.run(
            "backend.main:app",
            host=config.WEBSOCKET_HOST,
            port=config.WEBSOCKET_PORT,
            reload=True
        )