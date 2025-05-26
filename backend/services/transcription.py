"""
Speech-to-Text Transcription Service

Uses Hugging Face Transformers for Whisper-based transcription.
"""

import torch
import torchaudio
import numpy as np
import logging
import io
import time
from typing import Dict, Any, Tuple, Optional
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    Speech-to-Text service using Hugging Face Transformers Whisper models.
    Hardcoded for German transcription using the configured ASR model.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3", # Default, but will be overridden by config
        device: str = "auto", 
        compute_type: str = "auto", # Maps to torch_dtype
        sample_rate: int = 16000,   # This is the sample rate the transcriber expects to receive or will resample to.
                                    # For Whisper, this *should* align with whisper_internal_sample_rate.
        beam_size: int = 2 
    ):
        self.model_name = model_name
        self.device_str = device
        self.torch_dtype_str = compute_type
        
        # self.input_audio_configured_sample_rate = sample_rate # Informational: what the config said the rate should be
        self.whisper_internal_sample_rate = 16000  # Whisper model's fixed expected sample rate. This is crucial.
        
        self.beam_size = beam_size # Retained if used by generate; for basic generate, it's often implicit.

        self.model = None
        self.processor = None
        self.torch_dtype_actual = None
        self.device_actual = None
        
        # Hardcoded for this simplified request
        self.language = "de" 
        self.task = "transcribe"

        self._initialize_model()
        
        self.is_processing = False
        logger.info(f"Initialized Whisper Transcriber with model={self.model_name}, "
                   f"language={self.language}, task={self.task}, "
                   f"requested_device={self.device_str}, requested_dtype={self.torch_dtype_str}, "
                   f"Whisper internal SR: {self.whisper_internal_sample_rate}Hz")

    def _determine_torch_dtype(self) -> Optional[torch.dtype]:
        if self.device_actual is not None and self.device_actual.type == "cpu":
            logger.info("Device is CPU, using torch.float32 for model if dtype is 'auto' or 'float32'.")
            if self.torch_dtype_str.lower() in ["auto", "float32"]:
                return torch.float32
            else: # If user explicitly set a non-float32 dtype for CPU, warn but try.
                logger.warning(f"Requested dtype {self.torch_dtype_str} for CPU; float32 is generally recommended.")
                if self.torch_dtype_str == "float16": return torch.float16
                if hasattr(torch, 'bfloat16') and self.torch_dtype_str == "bfloat16": return torch.bfloat16


        if self.torch_dtype_str == "float16":
            return torch.float16
        elif hasattr(torch, 'bfloat16') and self.torch_dtype_str == "bfloat16":
            return torch.bfloat16
        elif self.torch_dtype_str == "float32":
            return torch.float32
        elif self.device_actual is not None and self.device_actual.type == "cuda" and torch.cuda.is_available():
             if hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported():
                logger.info("BF16 is supported on CUDA, auto-selecting torch.bfloat16 for model.")
                return torch.bfloat16
             else:
                logger.info("Auto-selecting torch.float16 for CUDA model.")
                return torch.float16
        
        logger.info("Dtype is 'auto' and not on CUDA or specific type not matched; letting Transformers decide model dtype.")
        return None


    def _initialize_model(self):
        try:
            logger.info(f"Loading ASR model: {self.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            if self.device_str == "cuda" and torch.cuda.is_available():
                self.device_actual = torch.device("cuda")
            elif self.device_str == "cpu":
                self.device_actual = torch.device("cpu")
            else: # auto
                self.device_actual = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Target device selected: {self.device_actual}")

            self.torch_dtype_actual = self._determine_torch_dtype()
            
            model_kwargs = {"low_cpu_mem_usage": (self.device_actual.type == "cuda")}
            if self.torch_dtype_actual:
                model_kwargs["torch_dtype"] = self.torch_dtype_actual
                logger.info(f"Attempting to load model with dtype: {self.torch_dtype_actual}")

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.to(self.device_actual)
            
            # Optional: If your transformers version supports it and model benefits.
            # if self.device_actual.type == "cuda" and hasattr(self.model, "to_bettertransformer"):
            #     try:
            #         self.model = self.model.to_bettertransformer()
            #         logger.info("Applied to_bettertransformer() for potential speedup on CUDA.")
            #     except Exception as e:
            #         logger.warning(f"Could not apply to_bettertransformer: {e}")

            logger.info(f"ASR model '{self.model_name}' loaded on device: {self.model.device} "
                        f"with dtype: {self.model.dtype}") # Log actual loaded dtype
            self._warmup_model()

        except Exception as e:
            logger.error(f"Failed to load Whisper model or processor: {e}", exc_info=True)
            raise

    def _warmup_model(self):
        if not self.model or not self.processor:
            logger.error("Model or processor not initialized for warmup.")
            return
        try:
            logger.info("Running ASR warmup...")
            # Warmup audio MUST be at the Whisper internal sample rate (16000 Hz)
            warmup_audio_np = np.zeros(self.whisper_internal_sample_rate, dtype=np.float32) 
            
            # Processor MUST be told the audio is 16000 Hz
            input_features = self.processor(
                warmup_audio_np, sampling_rate=self.whisper_internal_sample_rate, return_tensors="pt"
            ).input_features

            # Move to device and ensure correct dtype for model input
            input_features = input_features.to(self.device_actual)
            if self.model.dtype == torch.bfloat16: # Check model's actual dtype
                input_features = input_features.to(torch.bfloat16)
            elif self.model.dtype == torch.float16:
                input_features = input_features.to(torch.float16)
            
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
            
            with torch.no_grad():
                _ = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=10)
            logger.info("ASR model warmed up successfully.")
        except Exception as e:
            logger.error(f"Error during ASR warmup: {e}", exc_info=True)
            # We don't re-raise here to allow app to start, but transcription might fail.

    def transcribe(self, audio: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio data to German text.
        
        Args:
            audio: Audio data as numpy array. Can be int/float PCM or uint8 for WAV bytes.
            
        Returns:
            Tuple[str, Dict[str, Any]]: 
                - Transcribed text (German).
                - Dictionary with metadata.
        """
        if not self.model or not self.processor:
            logger.error("Model or processor not initialized for transcription.")
            return "Error: Transcription service not ready.", {"error": "Service not initialized"}

        start_time = time.time()
        self.is_processing = True
        
        try:
            if audio.dtype == np.uint8:
                logger.debug("Received uint8 audio data, attempting to load as WAV.")
                try:
                    waveform, sr = torchaudio.load(io.BytesIO(audio.tobytes()))
                except Exception as e:
                    logger.error(f"Failed to load uint8 audio as WAV: {e}. Assuming it's raw 8-bit PCM might be incorrect.")
                    raise ValueError("uint8 audio data could not be loaded as WAV and raw uint8 PCM is not directly handled here.") from e
            elif np.issubdtype(audio.dtype, np.floating) or np.issubdtype(audio.dtype, np.integer):
                logger.debug(f"Received numeric audio data (dtype: {audio.dtype}).")
                audio_float = audio.astype(np.float32)
                if np.issubdtype(audio.dtype, np.integer):
                    max_val = np.iinfo(audio.dtype).max
                    if max_val != 0: audio_float = audio_float / max_val
                
                if audio_float.ndim == 1: audio_float = audio_float[np.newaxis, :]
                waveform = torch.from_numpy(audio_float)
                # Assume sample rate of incoming raw numerical array needs to be checked/passed correctly.
                # For this function, we'll assume it's coming from a source that should provide 16kHz
                # or it's already handled by the caller (e.g. frontend AudioService).
                # If it were arbitrary, we'd need sr from an argument.
                # For now, we'll use self.whisper_internal_sample_rate as the assumed SR for raw numpy arrays too.
                sr = self.whisper_internal_sample_rate 
            else:
                raise TypeError(f"Unsupported audio data type: {audio.dtype}")

            if sr != self.whisper_internal_sample_rate:
                logger.info(f"Resampling audio from {sr}Hz to {self.whisper_internal_sample_rate}Hz.")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.whisper_internal_sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                logger.debug("Converting stereo to mono.")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            processed_waveform_np = waveform.squeeze().numpy()

            # Crucial: Tell the processor the audio is now at 16000 Hz
            input_features = self.processor(
                processed_waveform_np, sampling_rate=self.whisper_internal_sample_rate, return_tensors="pt"
            ).input_features.to(self.device_actual)

            # Ensure dtype matches model's expected input dtype
            if self.model.dtype == torch.bfloat16:
                input_features = input_features.to(torch.bfloat16)
            elif self.model.dtype == torch.float16:
                 input_features = input_features.to(torch.float16)
            # Add other dtypes if necessary, or ensure model.dtype matches what input_features becomes

            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
            
            with torch.no_grad():
                # Use beam_size if it's a relevant parameter for model.generate and > 1
                generate_kwargs = {"forced_decoder_ids": forced_decoder_ids}
                if self.beam_size > 1 : # Typically beam search is default if num_beams not set or 1
                    # Note: whisper `generate` often uses `num_beams` if not using specialized methods
                    # For basic .generate(), it might imply greedy if num_beams=1 or not set.
                    # Check docs for the specific model's generate behavior if beam_size is critical.
                    # For now, we'll assume it's handled or default is fine.
                    # generate_kwargs["num_beams"] = self.beam_size # Example
                    pass


                predicted_ids = self.model.generate(input_features, **generate_kwargs)
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription (German) completed in {processing_time:.2f}s: {transcription[:50]}...")
            
            metadata = {
                "language": self.language,
                "task": self.task,
                "processing_time": processing_time,
                "model_used": self.model_name
            }
            return transcription, metadata
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return "", {"error": str(e)}
        finally:
            self.is_processing = False
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": str(self.device_actual),
            "compute_type_requested": self.torch_dtype_str, # What was asked for
            "compute_type_actual": str(self.model.dtype) if self.model else "N/A", # What model is using
            "beam_size": self.beam_size,
            "whisper_internal_sample_rate": self.whisper_internal_sample_rate,
            "is_processing": self.is_processing,
            "language_fixed": self.language,
            "task_fixed": self.task
        }