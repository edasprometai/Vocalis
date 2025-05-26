import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import audioService, { AudioEvent } from '../services/audio';

interface PlexusOrbProps {
  state: 'idle' | 'greeting' | 'listening' | 'processing' | 'speaking' | 'vision_file' | 'vision_processing' | 'vision_asr';
}

const PlexusOrb: React.FC<PlexusOrbProps> = ({ state }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const orbGroupRef = useRef<THREE.Group>();
  const nodesRef = useRef<THREE.Vector3[]>([]);
  const nodePositionsRef = useRef<THREE.Vector3[]>([]);
  const nodeSpheresRef = useRef<THREE.Sprite[]>([]);
  const connectionsRef = useRef<THREE.Line[]>([]);
  const animationIdRef = useRef<number>();
  
  // Audio data
  const [audioLevel, setAudioLevel] = useState(0);
  const [waveformData, setWaveformData] = useState<Float32Array | null>(null);
  const audioHistoryRef = useRef<number[]>([]);
  const maxHistoryLength = 128; // Store last 128 audio samples
  
  // Configuration
  const config = {
    nodeCount: 200,
    connectionDistance: 1,
    waveIntensity: 0.3,
    glowIntensity: 1,
    radius: 1.5
  };

  // Orange color palette
  const colors = {
    primary: 0xff6b35,
    secondary: 0xff8555,
    glow: 0xffa500,
    connection: 0xff7f50
  };

  // Listen to audio events
  useEffect(() => {
    const handleAudioData = (data: any) => {
      if (data.energy !== undefined) {
        setAudioLevel(data.energy);
        
        // Store audio history for waveform visualization
        audioHistoryRef.current.push(data.energy);
        if (audioHistoryRef.current.length > maxHistoryLength) {
          audioHistoryRef.current.shift();
        }
      }
      
      if (data.buffer) {
        setWaveformData(data.buffer);
      }
    };

    audioService.addEventListener(AudioEvent.RECORDING_DATA, handleAudioData);

    return () => {
      audioService.removeEventListener(AudioEvent.RECORDING_DATA, handleAudioData);
    };
  }, []);

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0a0a0a, 5, 15);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xff6b35, 1, 10);
    pointLight.position.set(0, 0, 0);
    scene.add(pointLight);

    // Create orb group
    const orbGroup = new THREE.Group();
    scene.add(orbGroup);
    orbGroupRef.current = orbGroup;

    // Create plexus orb
    createPlexusOrb();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;
      
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Create glowing node material
  const createNodeMaterial = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;
    
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255, 107, 53, 1)');
    gradient.addColorStop(0.3, 'rgba(255, 107, 53, 0.8)');
    gradient.addColorStop(0.6, 'rgba(255, 133, 85, 0.3)');
    gradient.addColorStop(1, 'rgba(255, 165, 0, 0)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(32, 32, 32, 0, Math.PI * 2);
    ctx.fill();
    
    const texture = new THREE.CanvasTexture(canvas);
    
    return new THREE.SpriteMaterial({
      map: texture,
      blending: THREE.AdditiveBlending,
      color: 0xffffff,
      opacity: 0.8
    });
  };

  // Create plexus orb
  const createPlexusOrb = () => {
    if (!orbGroupRef.current) return;

    const orbGroup = orbGroupRef.current;
    const nodeMaterial = createNodeMaterial();
    
    const latitudes = Math.floor(Math.sqrt(config.nodeCount));
    const longitudes = Math.floor(config.nodeCount / latitudes);
    
    for (let lat = 0; lat < latitudes; lat++) {
      for (let lon = 0; lon < longitudes; lon++) {
        const phi = (Math.PI * lat) / (latitudes - 1);
        const theta = (2 * Math.PI * lon) / longitudes;
        
        const r = config.radius + (Math.random() - 0.5) * 0.05;
        
        const x = r * Math.sin(phi) * Math.cos(theta);
        const y = r * Math.sin(phi) * Math.sin(theta);
        const z = r * Math.cos(phi);
        
        const position = new THREE.Vector3(x, y, z);
        nodesRef.current.push(position);
        nodePositionsRef.current.push(position.clone());
        
        const sprite = new THREE.Sprite(nodeMaterial.clone());
        sprite.position.copy(position);
        sprite.scale.set(0.15, 0.15, 0.15);
        sprite.userData = { lat, lon };
        nodeSpheresRef.current.push(sprite);
        orbGroup.add(sprite);
      }
    }

    // Create connections
    createConnections();
  };

  // Create connections between neighbors
  const createConnections = () => {
    if (!orbGroupRef.current) return;

    const latitudes = Math.floor(Math.sqrt(config.nodeCount));
    const longitudes = Math.floor(config.nodeCount / latitudes);

    for (let i = 0; i < nodeSpheresRef.current.length; i++) {
      const sprite = nodeSpheresRef.current[i];
      const lat = sprite.userData.lat;
      const lon = sprite.userData.lon;
      
      const neighbors = [
        { lat: lat, lon: (lon + 1) % longitudes },
        { lat: lat + 1, lon: lon },
        { lat: lat + 1, lon: (lon + 1) % longitudes },
        { lat: lat + 1, lon: (lon - 1 + longitudes) % longitudes }
      ];
      
      neighbors.forEach(neighbor => {
        if (neighbor.lat < latitudes) {
          const neighborIndex = neighbor.lat * longitudes + neighbor.lon;
          if (neighborIndex < nodesRef.current.length && neighborIndex > i) {
            const distance = nodesRef.current[i].distanceTo(nodesRef.current[neighborIndex]);
            
            if (distance < config.connectionDistance * 1.5) {
              const geometry = new THREE.BufferGeometry();
              const positions = new Float32Array(6);
              geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
              
              const material = new THREE.LineBasicMaterial({
                color: colors.connection,
                transparent: true,
                opacity: 0.4 * config.glowIntensity,
                blending: THREE.AdditiveBlending
              });
              
              const line = new THREE.Line(geometry, material);
              line.userData = { start: i, end: neighborIndex };
              connectionsRef.current.push(line);
              orbGroupRef.current!.add(line);
            }
          }
        }
      });
    }
  };

  // Get audio-based waveform value
  const getAudioWaveformValue = (nodeIndex: number, time: number): number => {
    // Use real audio data when available
    if (state === 'listening' || state === 'speaking') {
      // Map node index to audio history
      const historyIndex = Math.floor((nodeIndex / config.nodeCount) * audioHistoryRef.current.length);
      const audioValue = audioHistoryRef.current[historyIndex] || 0;
      
      // Amplify for visual effect
      return audioValue * 20 * config.waveIntensity;
    }
    
    // Fallback to simulated waves for other states
    const frequency = state === 'processing' ? 3 : 
                     state === 'greeting' ? 2 : 0.5;
    const amplitude = state === 'processing' ? 0.2 : 
                     state === 'greeting' ? 0.15 : 0.1;
    
    return Math.sin(time * frequency + nodeIndex * 0.1) * amplitude * config.waveIntensity;
  };

  // Animation loop
  useEffect(() => {
    let waveTime = 0;
    
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);

      if (!rendererRef.current || !sceneRef.current || !cameraRef.current || !orbGroupRef.current) return;

      // Update time based on state
      const timeIncrement = state === 'speaking' ? 0.05 : 
                          state === 'processing' ? 0.04 : 
                          state === 'listening' ? 0.03 : 0.02;
      waveTime += timeIncrement;

      // Rotate orb
      orbGroupRef.current.rotation.y += state === 'speaking' ? 0.01 : 
                                       state === 'processing' ? 0.008 : 0.005;
      orbGroupRef.current.rotation.x += state === 'speaking' ? 0.005 : 0.002;

      // Update nodes based on audio
      for (let i = 0; i < nodeSpheresRef.current.length; i++) {
        const waveValue = getAudioWaveformValue(i, waveTime);
        const scale = 1 + waveValue;
        
        nodeSpheresRef.current[i].position.x = nodePositionsRef.current[i].x * scale;
        nodeSpheresRef.current[i].position.y = nodePositionsRef.current[i].y * scale;
        nodeSpheresRef.current[i].position.z = nodePositionsRef.current[i].z * scale;
        
        nodesRef.current[i].copy(nodeSpheresRef.current[i].position);
        
        const nodeScale = 0.15 + waveValue * 0.1;
        nodeSpheresRef.current[i].scale.set(nodeScale, nodeScale, nodeScale);
        
        const brightness = state === 'speaking' ? 1.5 : 
                         state === 'listening' ? 1.2 : 
                         state === 'processing' ? 1.3 : 1;
        nodeSpheresRef.current[i].material.opacity = 0.8 * brightness * config.glowIntensity;
      }

      // Update connections
      connectionsRef.current.forEach((line, index) => {
        const positions = line.geometry.attributes.position.array;
        const startIndex = line.userData.start;
        const endIndex = line.userData.end;
        
        if (nodesRef.current[startIndex] && nodesRef.current[endIndex]) {
          positions[0] = nodesRef.current[startIndex].x;
          positions[1] = nodesRef.current[startIndex].y;
          positions[2] = nodesRef.current[startIndex].z;
          positions[3] = nodesRef.current[endIndex].x;
          positions[4] = nodesRef.current[endIndex].y;
          positions[5] = nodesRef.current[endIndex].z;
          
          line.geometry.attributes.position.needsUpdate = true;
        }
        
        const pulseSpeed = state === 'speaking' ? 3 : 2;
        const pulse = Math.sin(waveTime * pulseSpeed + index * 0.1) * 0.2 + 0.8;
        line.material.opacity = 0.4 * pulse * config.glowIntensity;
      });

      rendererRef.current.render(sceneRef.current, cameraRef.current);
    };

    animate();

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    };
  }, [state]);

  return (
    <div 
      ref={mountRef} 
      style={{ 
        width: '208px', 
        height: '208px',
        position: 'relative'
      }}
    />
  );
};

export default PlexusOrb;