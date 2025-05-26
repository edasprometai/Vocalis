# backend/services/transcription.py

import torch
import torchaudio
import numpy as np
import logging
import io
import time
from typing import Dict, Any, Tuple, Optional
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "openai/whisper-medium",
        device: str = "auto",
        compute_type: str = "auto", # Maps to torch_dtype for non-quantized or compute_dtype for quantization
        sample_rate: int = 16000, # Informational, Whisper uses 16kHz internally
        beam_size: int = 2
    ):
        self.model_name = model_name
        self.device_str = device
        self.torch_dtype_str = compute_type # User's preference for compute, e.g., "bfloat16"
        self.whisper_internal_sample_rate = 16000
        self.beam_size = beam_size

        self.model = None
        self.processor = None
        self.device_actual = None
        self.compute_dtype_actual = None # This will be the dtype for bnb_4bit_compute_dtype

        self.language = "de"
        self.task = "transcribe"

        self._initialize_model()
        
        self.is_processing = False
        logger.info(
            f"Initialized Whisper Transcriber: model={self.model_name}, lang={self.language}, task={self.task}, "
            f"req_device={self.device_str}, req_dtype={self.torch_dtype_str}, "
            f"Whisper SR: {self.whisper_internal_sample_rate}Hz"
        )

    def _get_bnb_compute_dtype(self) -> Optional[torch.dtype]:
        if self.torch_dtype_str == "bfloat16" and hasattr(torch, 'bfloat16'):
            return torch.bfloat16
        elif self.torch_dtype_str == "float16":
            return torch.float16
        elif self.torch_dtype_str == "float32":
            return torch.float32
        # Default for bnb compute_dtype if "auto" or not specified above
        if self.device_actual.type == "cuda" and hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 # Default compute dtype for BnB if not bfloat16

    def _initialize_model(self):
        try:
            logger.info(f"Loading ASR processor: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            if self.device_str == "cuda" and torch.cuda.is_available():
                self.device_actual = torch.device("cuda")
            elif self.device_str == "cpu":
                self.device_actual = torch.device("cpu")
            else: # auto
                self.device_actual = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Target device selected: {self.device_actual}")

            model_kwargs = {
                "low_cpu_mem_usage": True, # Useful for large models, loads on meta device first
                "attn_implementation": "sdpa", # Use Scaled Dot Product Attention
            }
            
            self.compute_dtype_actual = self._get_bnb_compute_dtype() # For BnB

            if self.device_actual.type == "cuda":
                logger.info(f"Attempting to apply 8-bit quantization with compute_dtype: {self.compute_dtype_actual}")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    # bnb_8bit_compute_dtype=self.compute_dtype_actual # Not a standard param for load_in_8bit
                                                                     # 8-bit usually infers or uses fp32 for certain ops
                )
                model_kwargs["quantization_config"] = quantization_config
                # device_map="auto" is crucial for bitsandbytes to work correctly with quantization
                # and distribute the model if it's very large, or place it on the correct device.
                model_kwargs["device_map"] = "auto" 
            else: # CPU
                logger.info("Device is CPU. Quantization with bitsandbytes is typically for CUDA. Loading model in default precision for CPU.")
                # For CPU, torch_dtype can be explicitly float32 if needed, but often default is fine.
                if self.torch_dtype_str == "float32": # Explicitly set float32 for CPU if requested
                     model_kwargs["torch_dtype"] = torch.float32


            logger.info(f"Loading ASR model: {self.model_name} with kwargs: {model_kwargs}")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # If device_map was not used (e.g. CPU path), ensure model is on the correct device.
            # With device_map="auto", from_pretrained handles device placement.
            if "device_map" not in model_kwargs and self.model.device != self.device_actual:
                self.model.to(self.device_actual)

            logger.info(
                f"ASR model '{self.model_name}' loaded. Effective device: {self.model.device}, "
                f"Effective dtype: {self.model.dtype}"
            )
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
            warmup_audio_np = np.zeros(self.whisper_internal_sample_rate, dtype=np.float32)
            
            input_features = self.processor(
                warmup_audio_np, sampling_rate=self.whisper_internal_sample_rate, return_tensors="pt"
            ).input_features
            
            # Move input_features to the model's device
            # For quantized models (esp. with device_map="auto"), model.device might be 'meta' initially
            # or point to the first device. It's safer to use self.device_actual if model is not on a specific device.
            target_device_for_inputs = self.model.device if str(self.model.device) != 'meta' else self.device_actual
            input_features = input_features.to(target_device_for_inputs)

            # Dtype for inputs usually float32 before quantization, or match compute_dtype if specified for BnB.
            # If model is 8-bit, inputs are typically float32 or float16/bfloat16.
            # The model layers handle the conversion.
            if self.model.dtype in [torch.float16, torch.bfloat16] and self.device_actual.type == 'cuda':
                 input_features = input_features.to(self.model.dtype)


            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
            
            with torch.no_grad():
                _ = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=10)
            logger.info("ASR model warmed up successfully.")
        except Exception as e:
            logger.error(f"Error during ASR warmup: {e}", exc_info=True)


    def transcribe(self, audio: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        if not self.model or not self.processor:
            logger.error("Model or processor not initialized for transcription.")
            return "Error: Transcription service not ready.", {"error": "Service not initialized"}

        start_time = time.time()
        self.is_processing = True
        
        try:
            if audio.dtype == np.uint8:
                logger.debug("Received uint8 audio data, attempting to load as WAV.")
                waveform, sr = torchaudio.load(io.BytesIO(audio.tobytes()))
            elif np.issubdtype(audio.dtype, np.floating) or np.issubdtype(audio.dtype, np.integer):
                logger.debug(f"Received numeric audio data (dtype: {audio.dtype}).")
                audio_float = audio.astype(np.float32)
                if np.issubdtype(audio.dtype, np.integer):
                    max_val = np.iinfo(audio.dtype).max
                    if max_val != 0: audio_float = audio_float / max_val
                
                if audio_float.ndim == 1: audio_float = audio_float[np.newaxis, :]
                waveform = torch.from_numpy(audio_float)
                sr = self.whisper_internal_sample_rate # Assume raw numpy is at correct rate or needs to be resampled
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

            input_features = self.processor(
                processed_waveform_np, sampling_rate=self.whisper_internal_sample_rate, return_tensors="pt"
            ).input_features
            
            target_device_for_inputs = self.model.device if str(self.model.device) != 'meta' else self.device_actual
            input_features = input_features.to(target_device_for_inputs)
            
            # Match input feature dtype to model's computation dtype if quantized or specific half-precision
            if self.model.dtype in [torch.float16, torch.bfloat16] and self.device_actual.type == 'cuda':
                 input_features = input_features.to(self.model.dtype)


            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
            
            generate_kwargs = {"forced_decoder_ids": forced_decoder_ids}
            # beam_size logic can be added here if needed, e.g. generate_kwargs["num_beams"] = self.beam_size

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, **generate_kwargs)
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription (German) completed in {processing_time:.2f}s: {transcription[:50]}...")
            
            return transcription, {
                "language": self.language, "task": self.task, 
                "processing_time": processing_time, "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return "", {"error": str(e)}
        finally:
            self.is_processing = False
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device_requested": self.device_str,
            "device_actual": str(self.device_actual),
            "compute_type_requested": self.torch_dtype_str,
            "compute_type_actual_for_bnb": str(self.compute_dtype_actual) if self.compute_dtype_actual else "N/A",
            "model_dtype_loaded": str(self.model.dtype) if self.model else "N/A",
            "beam_size": self.beam_size,
            "whisper_internal_sample_rate": self.whisper_internal_sample_rate,
            "is_processing": self.is_processing,
            "language_fixed": self.language,
            "task_fixed": self.task
        }