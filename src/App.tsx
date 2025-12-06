import React, { useState, useEffect, useRef, useMemo, Suspense, useLayoutEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Environment, Html, Loader, useTexture, Sparkles } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { damp3, damp } from 'maath/easing';
// 修复1: 移除了未使用的 'random' 引用
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// --- 0. 配置 ---
const CONFIG = {
  colors: {
    green: "#00A844", darkGreen: "#003311", gold: "#FFD700", red: "#EA2E49", white: "#F0F0F0", sockRed: "#D42426"
  },
  counts: { foliage: 5000, ornaments: 150, gifts: 60, socks: 50, polaroids: 15 }, 
  tree: { height: 12, radius: 5.5 },
  camDist: { idle: 30, active: 24 }
};

// --- 1. 几何算法 ---

const getTreePos = (i: number, count: number) => {
  const pct = i / count; 
  const y = (0.5 - pct) * CONFIG.tree.height; 
  let rBase = Math.pow(pct, 1.2) * CONFIG.tree.radius;
  rBase += Math.sin(pct * 12 * Math.PI) * 0.4 * pct; 
  const angle = i * 2.4; 
  const noise = (Math.sin(angle * 3) + Math.cos(angle * 5)) * 0.2;
  const finalRadius = Math.max(0, rBase + noise);
  return new THREE.Vector3(Math.cos(angle) * finalRadius, y, Math.sin(angle) * finalRadius);
};

const getSurfacePos = (i: number, count: number, offset = 0.0) => {
    const pct = i / count;
    const adjustedPct = Math.pow(pct, 0.9) * 0.9 + 0.05; 
    const y = (0.5 - adjustedPct) * CONFIG.tree.height;
    
    let rBase = Math.pow(adjustedPct, 1.2) * CONFIG.tree.radius;
    rBase += Math.sin(adjustedPct * 12 * Math.PI) * 0.4 * adjustedPct;
    
    const angle = i * 137.5; 
    const finalRadius = Math.max(0, rBase + 0.2 + offset); 

    return new THREE.Vector3(Math.cos(angle) * finalRadius, y, Math.sin(angle) * finalRadius);
}

const getRingPos = () => {
  const v = new THREE.Vector3();
  const u = Math.random();
  const v_rand = Math.random();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v_rand - 1);
  const r = 10 + Math.random() * 4; 
  
  v.x = r * Math.sin(phi) * Math.cos(theta);
  v.y = r * Math.sin(phi) * Math.sin(theta) * 0.8; 
  v.z = r * Math.cos(phi);
  return v;
};

// --- 2. AI Controller ---
const AIController = ({ onUpdate, debugMode }: { onUpdate: any, debugMode: boolean }) => {
  const videoRef = useRef<HTMLVideoElement>(null); const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let handLandmarker: any = null; let frameId: number;
    const init = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
          runningMode: "VIDEO", numHands: 1
        });
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } } });
        if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.onloadeddata = () => predict(); }
      } catch (e) { console.error("Camera error:", e); }
    };
    const predict = () => {
      if (videoRef.current && handLandmarker) {
        const result = handLandmarker.detectForVideo(videoRef.current, performance.now());
        if (result.landmarks && result.landmarks.length > 0) {
          const lm = result.landmarks[0];
          const dist = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y);
          const isOpen = dist > 0.08;
          const rawX = (1 - lm[0].x) * 2 - 1;
          const rawY = (1 - lm[0].y) * 2 - 1;
          const x = Math.max(-1.5, Math.min(1.5, rawX));
          const y = Math.max(-1.5, Math.min(1.5, rawY));
          onUpdate({ isOpen, x, y, dist: dist.toFixed(2) });
          if (debugMode && canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            if (ctx) {
              ctx.clearRect(0, 0, 320, 240); ctx.strokeStyle = isOpen ? "#00FF00" : "#FF0000"; ctx.lineWidth = 4;
              ctx.beginPath(); ctx.moveTo(lm[4].x * 320, lm[4].y * 240); ctx.lineTo(lm[8].x * 320, lm[8].y * 240); ctx.stroke();
            }
          }
        }
      }
      frameId = requestAnimationFrame(predict);
    };
    init(); return () => { cancelAnimationFrame(frameId); handLandmarker?.close(); };
  }, [debugMode]);
  return (
    <div className={`fixed bottom-24 left-4 z-50 bg-black/90 border-2 border-gold rounded p-2 transition-opacity duration-300 ${debugMode ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        <div className="relative w-[120px] h-[90px] bg-gray-900"><video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover opacity-60" style={{transform: "scaleX(-1)"}} /><canvas ref={canvasRef} width={320} height={240} className="absolute inset-0 w-full h-full object-cover" style={{transform: "scaleX(-1)"}} /></div>
    </div>
  );
};

// --- 3. Foliage ---
const Foliage = ({ active }: { active: boolean }) => {
  const shaderRef = useRef<THREE.ShaderMaterial>(null);
  const data = useMemo(() => {
    const count = CONFIG.counts.foliage; 
    const pos = new Float32Array(count*3), sz = new Float32Array(count);
    for(let i=0; i<count; i++) { 
      const t = getTreePos(i, count); 
      pos.set([t.x, t.y, t.z], i*3); 
      sz[i] = Math.random() * 0.2 + 0.1;
    }
    return { pos, sz };
  }, []);
  
  useFrame((s, d) => { 
    if (shaderRef.current) { 
      shaderRef.current.uniforms.uTime.value = s.clock.elapsedTime; 
      damp(shaderRef.current.uniforms.uProgress, "value", active ? 0 : 1, 0.25, d); 
    } 
  });

  return (
    <points>
      <bufferGeometry>
        {/* 修复2: 使用 args={[array, itemSize]} 传递参数，这是 Strict 模式要求的写法 */}
        <bufferAttribute attach="attributes-position" args={[data.pos, 3]} />
        <bufferAttribute attach="attributes-aSize" args={[data.sz, 1]} />
      </bufferGeometry>
      <shaderMaterial ref={shaderRef} transparent={true} depthWrite={false} blending={THREE.NormalBlending}
        uniforms={{ 
            uTime: { value: 0 }, uProgress: { value: 1 }, 
            uColorA: { value: new THREE.Color(CONFIG.colors.green) }, 
            uColorB: { value: new THREE.Color(CONFIG.colors.darkGreen) } 
        }}
        vertexShader={`
          uniform float uTime, uProgress; attribute float aSize; varying vec3 vPos;
          void main() { 
            vec3 p = position; 
            if (uProgress > 0.8) { 
                float wind = sin(uTime * 1.5 + p.x * 0.5 + p.y * 0.3) * 0.08;
                p.x += wind; 
            }
            vec4 mv = modelViewMatrix * vec4(p, 1.0); 
            gl_Position = projectionMatrix * mv; 
            float scale = smoothstep(0.0, 0.2, uProgress);
            gl_PointSize = aSize * (450.0 / -mv.z) * scale; 
            vPos = p;
          }
        `}
        fragmentShader={`
          uniform float uProgress; uniform vec3 uColorA, uColorB; varying vec3 vPos;
          void main() { 
            if (uProgress < 0.01) discard;
            float d = distance(gl_PointCoord, vec2(0.5)); if(d > 0.5) discard; 
            float depth = smoothstep(5.0, -5.0, vPos.y) * 0.6 + 0.4;
            vec3 color = mix(uColorB, uColorA, depth);
            gl_FragColor = vec4(color, uProgress); 
          }
        `}
      />
    </points>
  );
};

// --- 4. Top Star ---
const TopStar = ({ active }: { active: boolean }) => {
  const ref = useRef<THREE.Group>(null);
  const [target] = useState(() => new THREE.Vector3(0, CONFIG.tree.height / 2 + 0.6, 0));
  const [chaos] = useState(() => new THREE.Vector3(0, 8, 0));

  useLayoutEffect(() => { if (ref.current) ref.current.position.copy(target); }, [target]);

  useFrame((state, d) => {
    if (!ref.current) return;
    damp3(ref.current.position, active ? chaos : target, 0.3, d);
    ref.current.rotation.y += d * 0.5;
    if (active) ref.current.lookAt(state.camera.position);
    const s = active ? 2.5 : 1;
    damp3(ref.current.scale, [s, s, s], 0.3, d);
  });

  const starShape = useMemo(() => {
    const shape = new THREE.Shape();
    const outer = 0.8, inner = 0.4; 
    for (let i = 0; i < 10; i++) {
      const r = (i % 2 === 0) ? outer : inner;
      const a = (i / 10) * Math.PI * 2;
      shape[i===0?'moveTo':'lineTo'](Math.cos(a+Math.PI/2)*r, Math.sin(a+Math.PI/2)*r);
    }
    shape.closePath(); return shape;
  }, []);

  return (
    <group ref={ref} position={[0, CONFIG.tree.height/2+0.6, 0]}>
      <mesh>
        <extrudeGeometry args={[starShape, { depth: 0.3, bevelEnabled: true, bevelThickness: 0.1, bevelSize: 0.1, bevelSegments: 1 }]} />
        <meshStandardMaterial color={CONFIG.colors.gold} emissive={CONFIG.colors.gold} emissiveIntensity={1} metalness={1} roughness={0} />
      </mesh>
      <pointLight distance={10} intensity={80} color="#FFD700" />
    </group>
  );
};

// --- 5. Item Component ---
const Item = ({ type, index, active, countOffset, imgUrl }: any) => {
  const ref = useRef<THREE.Group>(null);
  
  const target = useMemo(() => {
    let offset = 0;
    if (type === 'gift') offset = 0.2;
    if (type === 'sock') offset = 0.1;
    if (type === 'photo') offset = 0.3; 
    return getSurfacePos(index + countOffset, 500, offset);
  }, [index, countOffset, type]);

  const chaos = useMemo(() => getRingPos(), []); 

  // 修复3: 强制将 useTexture 返回值断言为 THREE.Texture，解决类型不匹配
  const texture = useTexture(imgUrl || `https://picsum.photos/seed/${index+500}/200/200`) as THREE.Texture;

  const scales: [number, number] = useMemo(() => {
      if (type === 'photo') return [0.8, 2.0];
      if (type === 'ball') return [0.15, 0.5];
      if (type === 'gift') return [0.4, 0.8];
      if (type === 'sock') return [0.3, 0.7];
      return [1, 1];
  }, [type]);

  useFrame((state, d) => {
    if (!ref.current) return;
    const dest = active ? chaos : target;
    damp3(ref.current.position, dest, 0.3, d);
    
    const targetScale = active ? scales[1] : scales[0];
    damp3(ref.current.scale, [targetScale, targetScale, targetScale], 0.3, d);

    if (active) {
       ref.current.lookAt(state.camera.position);
    } else {
       damp(ref.current.rotation, "y", Math.atan2(target.x, target.z), 0.4, d);
       damp(ref.current.rotation, "x", type === 'photo' ? -0.2 : 0, 0.4, d);
       damp(ref.current.rotation, "z", 0, 0.4, d);
    }
  });

  if (type === 'ball') {
    const color = index % 2 === 0 ? CONFIG.colors.gold : CONFIG.colors.red;
    return (
      <group ref={ref}>
        <mesh><sphereGeometry args={[1, 16, 16]} /><meshStandardMaterial color={color} metalness={0.7} roughness={0.2} /></mesh>
      </group>
    );
  } else if (type === 'gift') {
    const isRed = index % 2 === 0;
    return (
      <group ref={ref}>
        <mesh><boxGeometry args={[1, 0.8, 1]} /><meshStandardMaterial color={isRed?CONFIG.colors.red:CONFIG.colors.gold} /></mesh>
        <mesh scale={[1.05, 1.05, 0.1]}><boxGeometry args={[1, 0.8, 1]} /><meshStandardMaterial color="#FFF" /></mesh>
        <mesh scale={[0.1, 1.05, 1.05]}><boxGeometry args={[1, 0.8, 1]} /><meshStandardMaterial color="#FFF" /></mesh>
      </group>
    );
  } else if (type === 'sock') {
    return (
      <group ref={ref}>
        <mesh position={[0, 0.5, 0]}><cylinderGeometry args={[0.4, 0.4, 1, 16]} /><meshStandardMaterial color={CONFIG.colors.sockRed} /></mesh>
        <mesh position={[0, 1.0, 0]}><cylinderGeometry args={[0.45, 0.45, 0.3, 16]} /><meshStandardMaterial color="#FFF" roughness={1} /></mesh>
        <mesh position={[0.3, 0.1, 0]} rotation={[0, 0, -Math.PI/4]}><capsuleGeometry args={[0.38, 0.6, 4, 8]} /><meshStandardMaterial color={CONFIG.colors.sockRed} /></mesh>
      </group>
    );
  } else if (type === 'photo') {
    return (
      <group ref={ref}>
        <mesh><boxGeometry args={[1.0, 1.2, 0.05]} /><meshStandardMaterial color="#FFF" /></mesh>
        <mesh position={[0, 0.1, 0.06]}><planeGeometry args={[0.85, 0.85]} /><meshBasicMaterial map={texture} /></mesh>
      </group>
    );
  }
  return null;
};

// --- 6. Scene ---
const Scene = ({ active, gestureData, userImages }: { active: boolean, gestureData: any, userImages: string[] }) => {
  const { camera } = useThree();
  
  useFrame((_, d) => {
    const radius = active ? CONFIG.camDist.active : CONFIG.camDist.idle;
    const theta = gestureData.x * Math.PI * 0.6; 
    const elevation = gestureData.y * 8 + 2; 

    const tx = Math.sin(theta) * radius;
    const tz = Math.cos(theta) * radius;
    
    damp3(camera.position, [tx, elevation, tz], 0.25, d);
    camera.lookAt(0, 1, 0); 
  });

  return (
    <>
      <color attach="background" args={['#050505']} />
      <ambientLight intensity={0.6} /> 
      <spotLight position={[10, 20, 10]} intensity={180} color="#FFD700" castShadow angle={0.5} penumbra={0.5} />
      <pointLight position={[-10, 5, -10]} intensity={60} color="#FFF" />
      
      <Environment preset="night" blur={0.8} />

      <Foliage active={active} />
      <TopStar active={active} />
      <Sparkles count={200} scale={25} size={3} speed={0.2} opacity={0.5} color="#FFF" />

      {Array.from({ length: CONFIG.counts.ornaments }).map((_, i) => (
        <Item key={`ball-${i}`} type="ball" index={i} countOffset={0} active={active} />
      ))}
      {Array.from({ length: CONFIG.counts.gifts }).map((_, i) => (
        <Item key={`gift-${i}`} type="gift" index={i} countOffset={200} active={active} />
      ))}
      {Array.from({ length: CONFIG.counts.socks }).map((_, i) => (
        <Item key={`sock-${i}`} type="sock" index={i} countOffset={400} active={active} />
      ))}
      {Array.from({ length: CONFIG.counts.polaroids }).map((_, i) => (
        <Item key={`pol-${i}`} type="photo" index={i} countOffset={600} active={active} imgUrl={userImages[i % userImages.length]} />
      ))}

      {/* 修复4: 移除了 EffectComposer 上不存在的属性 disableNormalPass */}
      <EffectComposer>
        <Bloom luminanceThreshold={0.7} intensity={0.8} radius={0.5} mipmapBlur />
        <Vignette eskil={false} offset={0.1} darkness={1.0} />
      </EffectComposer>
    </>
  );
};

// --- 7. Entry ---
export default function App() {
  const [gesture, setGesture] = useState({ isOpen: false, x: 0, y: 0, dist: "0" });
  const [debug, setDebug] = useState(false);
  const [images, setImages] = useState<string[]>([]);
  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => { if (e.target.files) { const newImages = Array.from(e.target.files).map(f => URL.createObjectURL(f)); setImages(prev => [...newImages, ...prev]); } };

  return (
    <div className="w-full h-full bg-black relative overflow-hidden select-none font-serif">
      <AIController onUpdate={setGesture} debugMode={debug} />
      
      <div className={`absolute top-8 w-full text-center z-10 pointer-events-none px-4 mix-blend-screen transition-opacity duration-500 ${gesture.isOpen ? 'opacity-0' : 'opacity-100'}`}>
        <h1 className="text-5xl md:text-7xl text-[#FFD700] tracking-widest uppercase drop-shadow-[0_0_15px_rgba(255,215,0,0.6)]">
          The Gilded Pine
        </h1>
        <p className="text-emerald-300/80 text-sm tracking-[0.4em] mt-3 border-t border-b border-emerald-800/50 py-2 inline-block">
          MERRY CHRISTMAS
        </p>
      </div>

      <button onClick={() => setDebug(!debug)} className="absolute bottom-6 left-6 z-50 flex items-center justify-center w-10 h-10 bg-black/40 border border-[#FFD700]/30 rounded-full text-xl hover:bg-[#FFD700]/20 hover:text-[#FFD700] transition-all opacity-40 hover:opacity-100" title="Debug">🐞</button>
      
      <div className={`absolute bottom-12 w-full flex justify-center z-40 pointer-events-auto transition-opacity duration-500 ${gesture.isOpen ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
         <label className="cursor-pointer group relative">
            <div className="absolute -inset-1 rounded-full bg-gradient-to-r from-[#FFD700] to-[#E5C100] opacity-70 blur transition duration-500 group-hover:opacity-100 group-hover:blur-md"></div>
            <div className="relative flex items-center gap-3 bg-black/80 px-10 py-4 rounded-full border border-[#FFD700]/50 text-[#FFD700] transition duration-300 group-hover:bg-[#FFD700] group-hover:text-black group-active:scale-95">
                <span className="text-2xl">🎁</span><span className="tracking-[0.2em] font-bold">UPLOAD PHOTOS</span>
            </div>
            <input type="file" multiple accept="image/*" className="hidden" onChange={handleUpload} />
         </label>
      </div>

      <Canvas shadows dpr={[1, 1.5]} camera={{ position: [0, 2, 30], fov: 45 }} gl={{ antialias: false }}>
        <Suspense fallback={<Html center><div className="text-[#FFD700] tracking-widest animate-pulse">SUMMONING CHRISTMAS...</div></Html>}>
          <Scene active={gesture.isOpen} gestureData={gesture} userImages={images} />
        </Suspense>
      </Canvas>
      <Loader />
    </div>
  );
}