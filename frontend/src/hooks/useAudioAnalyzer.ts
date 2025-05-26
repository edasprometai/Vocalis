import { useEffect, useRef, useState } from 'react';
import audioService, { AudioEvent } from '../services/audio';

export const useAudioAnalyzer = () => {
  const [audioData, setAudioData] = useState<number[]>(new Array(128).fill(0));
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    // Listen to raw audio data from the audio service
    const handleAudioData = (data: any) => {
      if (data.buffer && data.buffer instanceof Float32Array) {
        // Perform FFT analysis on the audio buffer
        const fftSize = 256;
        const frequencyData = performFFT(data.buffer, fftSize);
        
        // Normalize the frequency data to 0-1 range
        const normalizedData = Array.from(frequencyData).map(value => value / 255);
        setAudioData(normalizedData);
      }
    };

    // Simple FFT approximation (for real implementation, use a proper FFT library)
    const performFFT = (buffer: Float32Array, fftSize: number): Uint8Array => {
      // This is a simplified frequency extraction
      // In production, you'd want to use a proper FFT implementation
      const frequencyBins = fftSize / 2;
      const result = new Uint8Array(frequencyBins);
      
      // Simple energy calculation per frequency bin
      const binSize = Math.floor(buffer.length / frequencyBins);
      for (let i = 0; i < frequencyBins; i++) {
        let sum = 0;
        const start = i * binSize;
        const end = Math.min(start + binSize, buffer.length);
        
        for (let j = start; j < end; j++) {
          sum += Math.abs(buffer[j]);
        }
        
        // Scale to 0-255 range
        result[i] = Math.min(255, Math.floor(sum / binSize * 1000));
      }
      
      return result;
    };

    audioService.addEventListener(AudioEvent.RECORDING_DATA, handleAudioData);

    return () => {
      audioService.removeEventListener(AudioEvent.RECORDING_DATA, handleAudioData);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return audioData;
};