import { useEffect, useRef, useState } from 'react';
import audioService, { AudioEvent } from '../services/audio';

export const useAudioAnalyzer = () => {
  // Store the raw FFT data (0-255)
  const [audioData, setAudioData] = useState<Uint8Array>(new Uint8Array(128).fill(0)); // Default to Uint8Array
  // No animationFrameRef needed here unless you were doing UI updates *within* the hook

  useEffect(() => {
    const handleAudioData = (data: any) => {
      if (data.buffer && data.buffer instanceof Float32Array) {
        const fftSize = 256; // Should match or be related to audioData length (128 bins)
        const frequencyData = performFFT(data.buffer, fftSize); // Returns Uint8Array
        setAudioData(frequencyData); // Set the Uint8Array directly
      }
    };

    const performFFT = (buffer: Float32Array, fftSize: number): Uint8Array => {
      const frequencyBins = fftSize / 2; // This will be 128
      const result = new Uint8Array(frequencyBins);
      const binSize = Math.floor(buffer.length / frequencyBins);

      if (binSize === 0) { // Prevent division by zero if buffer is too small
         console.warn("Audio buffer too small for FFT bin size");
         return result; // Return empty/zeroed array
      }

      for (let i = 0; i < frequencyBins; i++) {
        let sum = 0;
        const start = i * binSize;
        const end = Math.min(start + binSize, buffer.length);
        for (let j = start; j < end; j++) {
          sum += Math.abs(buffer[j]);
        }
        // Scale to 0-255 range. The '* 1000' might be too aggressive or too little.
        // Consider the typical range of Math.abs(buffer[j]) and sum/binSize
        // If buffer[j] is -1 to 1, then Math.abs is 0 to 1. sum/binSize is also likely 0 to 1.
        // A value like 0.1 * 1000 = 100. A value like 0.01 * 1000 = 10. This seems reasonable.
        result[i] = Math.min(255, Math.floor((sum / binSize) * 1000));
      }
      return result;
    };

    audioService.addEventListener(AudioEvent.RECORDING_DATA, handleAudioData);

    return () => {
      audioService.removeEventListener(AudioEvent.RECORDING_DATA, handleAudioData);
      // No animationFrameRef to cancel here
    };
  }, []);

  return audioData; // Returns Uint8Array
};