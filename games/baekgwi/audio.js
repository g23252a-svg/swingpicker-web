/* 백귀야행 — 소리
   타격음은 노이즈 버스트 + 짧은 음으로 겹쳐 만들고,
   배경음은 16분음표 격자에 미리 예약해 둔다 (setInterval로는 박자가 흔들린다). */
'use strict';
const SND = (function(){

let AC = null, comp = null, sfxBus = null, musBus = null, noiseBuf = null;
let ready = false, muted = false;

/* 배경음 상태 */
let playing = false, step = 0, nextTime = 0, intensity = 0;
const STEP_DUR = 0.1153;                 // 16분음표 @ 130BPM
const ROOTS = [110.00, 87.31, 98.00, 82.41];          // Am F G Em
const PENTA = [440.00, 523.25, 587.33, 659.25, 783.99];

/* 같은 순간에 소리가 몰리면 귀가 아프다. 짧은 간격으로 잘라낸다. */
const lastAt = {};
function throttle(key, gap){
  const t = AC.currentTime;
  if(lastAt[key] !== undefined && t - lastAt[key] < gap) return false;
  lastAt[key] = t;
  return true;
}

function init(){
  if(ready) return true;
  try{
    AC = new (window.AudioContext || window.webkitAudioContext)();
  }catch(e){ return false; }

  comp = AC.createDynamicsCompressor();
  comp.threshold.value = -15; comp.knee.value = 24; comp.ratio.value = 9;
  comp.attack.value = 0.003; comp.release.value = 0.18;

  const master = AC.createGain();
  master.gain.value = 0.95;
  comp.connect(master); master.connect(AC.destination);

  sfxBus = AC.createGain(); sfxBus.gain.value = 1.0;  sfxBus.connect(comp);
  musBus = AC.createGain(); musBus.gain.value = 0.30; musBus.connect(comp);

  const n = Math.floor(AC.sampleRate * 0.7);
  noiseBuf = AC.createBuffer(1, n, AC.sampleRate);
  const d = noiseBuf.getChannelData(0);
  for(let i=0;i<n;i++) d[i] = Math.random()*2 - 1;

  ready = true;
  return true;
}

function resume(){ if(AC && AC.state === 'suspended') AC.resume(); }

/* ── 소리 조각 ───────────────────────────── */
function tone(t, f0, f1, dur, type, vol, bus){
  const o = AC.createOscillator(); o.type = type || 'square';
  o.frequency.setValueAtTime(f0, t);
  if(f1 && f1 !== f0) o.frequency.exponentialRampToValueAtTime(Math.max(1, f1), t + dur);
  const g = AC.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(vol, t + Math.min(0.008, dur*0.2));
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g); g.connect(bus || sfxBus);
  o.start(t); o.stop(t + dur + 0.02);
}
function burst(t, dur, freq, q, vol, type, sweepTo){
  const s = AC.createBufferSource(); s.buffer = noiseBuf;
  const f = AC.createBiquadFilter();
  f.type = type || 'bandpass'; f.Q.value = q || 1;
  f.frequency.setValueAtTime(freq, t);
  if(sweepTo) f.frequency.exponentialRampToValueAtTime(Math.max(40, sweepTo), t + dur);
  const g = AC.createGain();
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  s.connect(f); f.connect(g); g.connect(sfxBus);
  s.start(t); s.stop(t + dur + 0.02);
}

/* ── 효과음 ──────────────────────────────── */
const S = {
  hit(combo){
    if(!ready || muted || !throttle('hit', 0.028)) return;
    const t = AC.currentTime, c = Math.min(combo||0, 24);
    burst(t, 0.05, 1500 + c*55, 1.7, 0.13);
    tone(t, 240 + c*4, 150, 0.045, 'square', 0.055);
  },
  crit(){
    if(!ready || muted || !throttle('crit', 0.05)) return;
    const t = AC.currentTime;
    burst(t, 0.13, 3200, 0.8, 0.26, 'bandpass', 900);
    tone(t, 820, 190, 0.14, 'square', 0.14);
    tone(t + 0.02, 1240, 400, 0.10, 'triangle', 0.09);
  },
  kill(combo){
    if(!ready || muted || !throttle('kill', 0.035)) return;
    const t = AC.currentTime, c = Math.min(combo||0, 30);
    tone(t, 190 + c*7, 68, 0.10, 'triangle', 0.10);
    burst(t, 0.085, 900, 1.0, 0.10, 'lowpass', 260);
  },
  bigKill(){
    if(!ready || muted) return;
    const t = AC.currentTime;
    tone(t, 130, 42, 0.30, 'sawtooth', 0.22);
    burst(t, 0.28, 1600, 0.6, 0.24, 'lowpass', 200);
  },
  hurt(){
    if(!ready || muted || !throttle('hurt', 0.08)) return;
    const t = AC.currentTime;
    tone(t, 165, 46, 0.24, 'sawtooth', 0.26);
    burst(t, 0.20, 520, 0.7, 0.22, 'lowpass', 120);
  },
  gem(streak){
    if(!ready || muted || !throttle('gem', 0.035)) return;
    const t = AC.currentTime;
    const f = 620 * Math.pow(1.0595, Math.min(streak||0, 14));
    tone(t, f, f*1.5, 0.07, 'triangle', 0.075);
  },
  heal(){
    if(!ready || muted) return;
    const t = AC.currentTime;
    [523.25, 659.25, 783.99].forEach((f,i) => tone(t + i*0.05, f, f, 0.16, 'triangle', 0.07));
  },
  level(){
    if(!ready || muted) return;
    const t = AC.currentTime;
    [523.25, 659.25, 783.99, 1046.5].forEach((f,i) =>
      tone(t + i*0.065, f, f, 0.26, 'triangle', 0.10));
    burst(t, 0.4, 4000, 0.5, 0.10, 'highpass', 800);
  },
  dash(){
    if(!ready || muted) return;
    const t = AC.currentTime;
    burst(t, 0.16, 400, 0.8, 0.16, 'highpass', 3600);
  },
  boss(){
    if(!ready || muted) return;
    const t = AC.currentTime;
    tone(t, 240, 34, 1.30, 'sawtooth', 0.28);
    tone(t, 121, 30, 1.30, 'square', 0.16);
    burst(t, 1.1, 200, 0.5, 0.20, 'lowpass', 2400);
  },
  win(){
    if(!ready || muted) return;
    const t = AC.currentTime;
    [[523.25,0],[659.25,0.14],[783.99,0.28],[1046.5,0.42],[1318.5,0.60]]
      .forEach(([f,d]) => { tone(t+d, f, f, 0.5, 'triangle', 0.11); tone(t+d, f/2, f/2, 0.5, 'square', 0.05); });
  },
  dead(){
    if(!ready || muted) return;
    const t = AC.currentTime;
    tone(t, 300, 55, 1.1, 'sawtooth', 0.22);
    tone(t + 0.1, 200, 40, 1.2, 'square', 0.14);
    burst(t, 0.9, 900, 0.6, 0.16, 'lowpass', 90);
  },
  ui(){
    if(!ready || muted) return;
    tone(AC.currentTime, 720, 980, 0.09, 'triangle', 0.09);
  },
};

/* ── 배경음 ──────────────────────────────── */
function note(t, f, dur, type, vol){ tone(t, f, f, dur, type, vol, musBus); }
function drum(t, kind){
  if(kind === 'kick'){
    tone(t, 150, 44, 0.16, 'sine', 0.42, musBus);
    const s = AC.createBufferSource(); s.buffer = noiseBuf;
    const f = AC.createBiquadFilter(); f.type='lowpass'; f.frequency.value = 220;
    const g = AC.createGain();
    g.gain.setValueAtTime(0.18, t); g.gain.exponentialRampToValueAtTime(0.0001, t+0.05);
    s.connect(f); f.connect(g); g.connect(musBus);
    s.start(t); s.stop(t+0.07);
  } else if(kind === 'snare'){
    const s = AC.createBufferSource(); s.buffer = noiseBuf;
    const f = AC.createBiquadFilter(); f.type='bandpass'; f.frequency.value = 1900; f.Q.value = 0.8;
    const g = AC.createGain();
    g.gain.setValueAtTime(0.20, t); g.gain.exponentialRampToValueAtTime(0.0001, t+0.13);
    s.connect(f); f.connect(g); g.connect(musBus);
    s.start(t); s.stop(t+0.15);
  } else {                                   // hat
    const s = AC.createBufferSource(); s.buffer = noiseBuf;
    const f = AC.createBiquadFilter(); f.type='highpass'; f.frequency.value = 7000;
    const g = AC.createGain();
    g.gain.setValueAtTime(0.055, t); g.gain.exponentialRampToValueAtTime(0.0001, t+0.035);
    s.connect(f); f.connect(g); g.connect(musBus);
    s.start(t); s.stop(t+0.05);
  }
}

function playStep(i, t){
  const inBar = i % 16, bar = Math.floor(i/16) % 4;
  const root = ROOTS[bar];

  if(inBar === 0 || inBar === 8) drum(t, 'kick');
  if(intensity > 0 && (inBar === 6 || inBar === 14)) drum(t, 'kick');
  if(inBar === 4 || inBar === 12) drum(t, 'snare');
  if(inBar % 2 === 1) drum(t, 'hat');

  // 베이스
  if(inBar % 3 === 0 || inBar === 8){
    const f = (inBar === 6 || inBar === 12) ? root*1.5 : root;
    note(t, f, 0.14, 'square', 0.12);
  }
  // 선율 — 후반부와 보스전에만 얹는다
  if(intensity > 0 && (inBar === 2 || inBar === 7 || inBar === 10 || inBar === 15)){
    const f = PENTA[(i*7 + bar*3) % PENTA.length];
    note(t, f, 0.19, 'triangle', 0.055 + intensity*0.03);
  }
  if(intensity > 1 && inBar % 4 === 2){
    note(t, ROOTS[bar]*2, 0.09, 'sawtooth', 0.035);
  }
}

function tick(){
  if(!ready || !playing || muted) return;
  const ahead = AC.currentTime + 0.22;
  let guard = 0;
  while(nextTime < ahead && guard++ < 40){
    playStep(step, nextTime);
    step++; nextTime += STEP_DUR;
  }
}

return {
  init, resume,
  get ready(){ return ready; },
  sfx: S,
  startMusic(){
    if(!ready) return;
    playing = true; step = 0; nextTime = AC.currentTime + 0.08; intensity = 0;
  },
  stopMusic(){ playing = false; },
  /* 0 = 평상시, 1 = 후반, 2 = 보스전 */
  setIntensity(v){
    if(!ready || v === intensity) return;
    intensity = v;
    musBus.gain.cancelScheduledValues(AC.currentTime);
    musBus.gain.setTargetAtTime(0.30 + v*0.06, AC.currentTime, 0.4);
  },
  setMuted(m){
    muted = m;
    if(ready) musBus.gain.setTargetAtTime(m ? 0 : 0.30 + intensity*0.06, AC.currentTime, 0.05);
  },
  get muted(){ return muted; },
  tick,
};
})();
