import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

interface PlexusOrbProps {
  state?: 'idle' | 'greeting' | 'listening' | 'processing' | 'speaking' | 'vision_file' | 'vision_processing' | 'vision_asr';
  audioData?: Uint8Array; // Real audio frequency data
  config?: {
    nodeCount: number;
    connectionDistance: number;
    waveIntensity: number;
    glowIntensity: number;
    radius: number;
    audioScaling: number;
    nodeMovementScaling: number;
    nodeScaleMultiplier: number;
    tempo: number;
    rotationSpeed: number;
    bassMultiplier: number;
    midMultiplier: number;
    highMultiplier: number;
    beatIntensity: number;
  };
}

const PlexusOrb: React.FC<PlexusOrbProps> = ({ state = 'idle', audioData, config: propConfig }) => {
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
  
  // Audio data storage
  const audioHistoryRef = useRef<number[]>([]);
  const maxHistoryLength = 128;
  
  // Configuration - merge with props
  const config = {
    nodeCount: 400,
    connectionDistance: 1.10,
    waveIntensity: 1.40,
    glowIntensity: 3.0,
    radius: 2.00,
    audioScaling: 0.30,
    nodeMovementScaling: 1.00,
    nodeScaleMultiplier: 1.30,
    tempo: 2.00,
    rotationSpeed: 2.00,
    bassMultiplier: 2.00,
    midMultiplier: 2.00,
    highMultiplier: 2.00,
    beatIntensity: 0.60,
    ...propConfig
  };

  // Orange color palette
  const colors = {
    primary: 0xff6b35,
    secondary: 0xff8555,
    glow: 0xffa500,
    connection: 0xff7f50
  };

  // Update audio data when it changes
  useEffect(() => {
    if (audioData && audioData.length > 0) {
      // Convert Uint8Array to normalized array (0-1)
      const normalizedData = Array.from(audioData).map(value => value / 255);
      audioHistoryRef.current = normalizedData;
    } else if (state !== 'listening' && state !== 'speaking') {
      // Clear audio data when not in audio states
      audioHistoryRef.current = [];
    }
  }, [audioData, state]);

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
    renderer.setClearColor(0x000000, 0);
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
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
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
        
        // Map node position to frequency range for audio visualization
        const nodeIndex = lat * longitudes + lon;
        const frequencyIndex = Math.floor((nodeIndex / config.nodeCount) * maxHistoryLength);
        
        sprite.userData = { 
          lat, 
          lon, 
          nodeIndex, 
          frequencyIndex: Math.min(frequencyIndex, maxHistoryLength - 1),
          normalizedLat: lat / (latitudes - 1),
          normalizedLon: lon / longitudes
        };
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

  // Map frequency ranges to sphere positions for realistic audio visualization
  const getAudioWaveformValue = (nodeIndex: number, time: number): number => {
    if ((state === 'listening' || state === 'speaking') && audioHistoryRef.current.length > 0) {
      const sprite = nodeSpheresRef.current[nodeIndex];
      if (!sprite || !sprite.userData) return 0;
      
      // Get the frequency index for this node
      const frequencyIndex = sprite.userData.frequencyIndex || 0;
      const audioValue = audioHistoryRef.current[frequencyIndex] || 0;
      
      // Map frequency to spatial regions
      const normalizedLat = sprite.userData.normalizedLat || 0;
      const freqRatio = frequencyIndex / maxHistoryLength;
      
      // Map frequency to spatial regions with configurable response
      let spatialMultiplier = 1;
      if (freqRatio < 0.3) {
        // Bass frequencies - more active in bottom half
        spatialMultiplier = normalizedLat > 0.6 ? config.bassMultiplier : 1.0;
      } else if (freqRatio < 0.7) {
        // Mid frequencies - active in middle band
        const distanceFromEquator = Math.abs(normalizedLat - 0.5);
        spatialMultiplier = config.midMultiplier - (distanceFromEquator * (config.midMultiplier - 1));
      } else {
        // High frequencies - more active in top half
        spatialMultiplier = normalizedLat < 0.4 ? config.highMultiplier : 1.0;
      }
      
      // Scale the audio response with configurable scaling
      const scaledAudioValue = audioValue * config.audioScaling * spatialMultiplier;
      
      // Add gentle time-based variation for organic feel
      const timeVariation = Math.sin(time * 1.5 + nodeIndex * 0.2) * 0.05;
      
      return (scaledAudioValue + timeVariation) * config.waveIntensity;
    }
    
    // Fallback to simulated waves for other states
    const frequency = state === 'processing' ? 3 : 
                     state === 'greeting' ? 2 : 0.5;
    const amplitude = state === 'processing' ? 0.05 :
                     state === 'greeting' ? 0.03 : 0.02;
    
    return Math.sin(time * frequency + nodeIndex * 0.1) * amplitude * config.waveIntensity;
  };

  // Animation loop
  useEffect(() => {
    let waveTime = 0;
    
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);

      if (!rendererRef.current || !sceneRef.current || !cameraRef.current || !orbGroupRef.current) return;

      // Configurable animation speeds
      const timeIncrement = (state === 'speaking' ? 0.02 : 
                            state === 'processing' ? 0.03 : 
                            state === 'listening' ? 0.025 : 
                            0.015) * config.tempo;
      waveTime += timeIncrement;

      // Configurable rotation speeds
      const baseRotationY = state === 'speaking' ? 0.004 : 
                           state === 'processing' ? 0.003 : 
                           0.002;
      const baseRotationX = state === 'speaking' ? 0.002 : 0.001;
      
      orbGroupRef.current.rotation.y += baseRotationY * config.rotationSpeed;
      orbGroupRef.current.rotation.x += baseRotationX * config.rotationSpeed;

      // Update nodes based on audio with configurable scaling
      for (let i = 0; i < nodeSpheresRef.current.length; i++) {
        const waveValue = getAudioWaveformValue(i, waveTime);
        
        // Configurable scaling while maintaining sphere shape
        const scale = 1 + waveValue * config.nodeMovementScaling;
        
        nodeSpheresRef.current[i].position.x = nodePositionsRef.current[i].x * scale;
        nodeSpheresRef.current[i].position.y = nodePositionsRef.current[i].y * scale;
        nodeSpheresRef.current[i].position.z = nodePositionsRef.current[i].z * scale;
        
        nodesRef.current[i].copy(nodeSpheresRef.current[i].position);
        
        // Configurable node scale changes
        const nodeScale = 0.15 + Math.abs(waveValue) * config.nodeScaleMultiplier;
        nodeSpheresRef.current[i].scale.set(nodeScale, nodeScale, nodeScale);
        
        // Dynamic opacity changes
        const baseBrightness = state === 'speaking' ? 1.3 : 
                              state === 'listening' ? 1.2 : 
                              state === 'processing' ? 1.15 : 1;
        const dynamicBrightness = 1 + Math.abs(waveValue) * 3;
        const finalOpacity = Math.min(1, 0.8 * baseBrightness * dynamicBrightness * config.glowIntensity);
        nodeSpheresRef.current[i].material.opacity = finalOpacity;
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
        
        // Configurable pulse speed
        const pulseSpeed = (state === 'speaking' ? 1.5 : 1) * config.tempo;
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
  }, [state, config]);

  return (
    <div 
      ref={mountRef} 
      style={{ 
        width: '208px', 
        height: '208px',
        position: 'relative',
        background: 'radial-gradient(circle, rgba(10,10,10,0.8) 0%, rgba(0,0,0,1) 100%)'
      }}
    />
  );
};

export default PlexusOrb;