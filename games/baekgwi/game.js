/* 백귀야행 — 게임 본체
   조작은 이동과 회피 둘뿐. 공격은 전부 자동으로 나간다. */
'use strict';
(function(){

/* ═══════════════ 자료 ═══════════════ */

const RUN_TIME = 382;                 // 저승사자가 나오는 시각
/* 카메라를 당겨 캐릭터를 크게 본다. 멀리서 구경하면 타격이 안 느껴진다.
   당긴 만큼 세계에서 보이는 넓이는 줄어드므로 소환·정리도 이 값을 쓴다. */
const ZOOM = 1.5;
const MAX_FOES = 430, MAX_PARTS = 150, MAX_TEXTS = 12;

const FOES = {
  honbul:  { spr:'honbul',  nm:'혼불',    hp:9,   spd:74, dmg:6,  xp:1, r:8,  from:0,   ai:'chase'  },
  heosu:   { spr:'heosu',   nm:'허수아비', hp:28, spd:58, dmg:9,  xp:2, r:10, from:12,  ai:'chase'  },
  dokkae:  { spr:'dokkae',  nm:'도깨비',  hp:48,  spd:62, dmg:12, xp:3, r:11, from:35,  ai:'charge' },
  haegol:  { spr:'haegol',  nm:'해골',    hp:36,  spd:88, dmg:11, xp:3, r:10, from:70,  ai:'chase'  },
  yeou:    { spr:'yeou',    nm:'불여우',  hp:80,  spd:90, dmg:15, xp:6, r:11, from:110, ai:'shoot'  },
  geuseun: { spr:'geuseun', nm:'그슨대',  hp:190, spd:52, dmg:22, xp:14, r:15, from:160, ai:'split' },
};

/* 무기 5단 + 짝이 되는 보조 3단이면 진화한다.
   숫자만 오르는 카드 대신 '무엇을 향해 키울까'를 만들어 주는 장치다. */
const EVO = {
  geom:   { need:'sutdol', nm:'월광참', ic:'🌙', desc:'사방을 한 번에 벤다. 피해 2.4배' },
  bujeok: { need:'buchae', nm:'만다라', ic:'🌀', desc:'부적이 소용돌이로 퍼진다' },
  byeorak:{ need:'jaseok', nm:'뇌신',   ic:'🌩', desc:'낙뢰가 세 번 튄다' },
  bulti:  { need:'sansam', nm:'겁화',   ic:'☄', desc:'장막이 넓어지고 태워 죽이면 터진다' },
  hwasal: { need:'jipsin', nm:'백발백중', ic:'🎯', desc:'사방으로 여덟 발이 꿰뚫는다' },
  bell:   { need:'tugu',   nm:'풍경',   ic:'🎐', desc:'방울이 커지고 충격파를 남긴다' },
};
/* ── 막(幕)과 이야기 ─────────────────────────────
   한 벌판에서 6분을 버티는 대신 세 마당을 지나가게 한다.
   땅빛과 나오는 것이 바뀌면 같은 시간도 길게 느껴진다. */
const ACTS = [
  { t:0,   nm:'1막 · 마을 어귀', line:'등이 하나씩 꺼진다. 아직은 잡것들이다.',
    ground:'#151129', dot1:'#191539', dot2:'#110e24', decal:'#22376a',
    pool:['honbul','heosu','dokkae'] },
  { t:125, nm:'2막 · 밤 산길',   line:'길이 사라지고 안개가 찼다. 뒤를 보지 마라.',
    ground:'#101c1e', dot1:'#16282a', dot2:'#0b1416', decal:'#2c6a5a',
    pool:['honbul','heosu','dokkae','haegol','yeou'] },
  { t:265, nm:'3막 · 저승 문턱', line:'여기서부터는 산 자의 땅이 아니다.',
    ground:'#1a0e1e', dot1:'#26142c', dot2:'#120a15', decal:'#6a2a55',
    pool:['dokkae','haegol','yeou','geuseun'] },
];

const CHARS = [
  { k:'musa',   nm:'무사', desc:'균형 잡힌 검객', arm:'bujeok',
    hp:100, spd:108, dmg:1.00, ic:'🗡' },
  { k:'munyeo', nm:'무녀', desc:'빠르지만 약하다', arm:'hwasal',
    hp:82,  spd:126, dmg:1.12, ic:'🏹' },
  { k:'yeoksa', nm:'역사', desc:'느리지만 단단하다', arm:'geom',
    hp:140, spd:94,  dmg:1.15, ic:'🪨' },
];
let charKey = 'musa';
const CH = () => CHARS.find(c => c.k === charKey) || CHARS[0];

const STORY = {
  intro: ['해가 지자 마을의 등이 하나씩 꺼졌다.',
          '백 가지 귀신이 줄지어 산을 내려온다.',
          '칼을 쥐고 그 앞에 선다.'],
  win:   ['저승사자가 먹물처럼 흩어지고',
          '동쪽 하늘이 희끗해졌다.',
          '칼을 거두고 마을로 돌아간다.'],
  lose:  ['이름이 명부에 올랐다.',
          '백귀의 밤은 아직 끝나지 않았다.'],
};

const BOSSES = [
  { spr:'dalgyal', nm:'달걀귀신', hp:520,  spd:44, dmg:8,  xp:80,  r:20, at:108, scale:5,
    line:'얼굴 없는 것이 웃었다.' },
  { spr:'gumiho',  nm:'구미호',   hp:820,  spd:38, dmg:9,  xp:150, r:22, at:238, scale:5,
    line:'꼬리 아홉이 달을 가렸다.' },
  { spr:'saja',    nm:'저승사자', hp:1500, spd:34, dmg:11, xp:600, r:26, at:382, scale:6,
    line:'명부를 든 자가 이름을 부른다.' },
];

// 무기 — lv 1..5. 값은 레벨별 배열.
const ARMS = {
  geom: { nm:'검기', ic:'⚔', desc:'앞쪽을 부채꼴로 벤다',
    cd:[0.80,0.70,0.62,0.55,0.48], dmg:[16,23,31,41,53], arc:[2.2,2.5,2.8,3.1,3.4], rng:[92,106,120,134,150] },
  bujeok:{ nm:'부적', ic:'📜', desc:'적을 쫓아가는 부적',
    cd:[0.90,0.80,0.70,0.62,0.54], dmg:[14,20,27,36,47], cnt:[2,2,3,3,4] },
  byeorak:{ nm:'벼락', ic:'⚡', desc:'하늘에서 내리꽂힌다',
    cd:[2.40,2.10,1.85,1.60,1.35], dmg:[34,46,60,78,100], cnt:[1,2,2,3,4] },
  bulti:{ nm:'불티', ic:'🔥', desc:'몸을 두른 불꽃이 태운다',
    cd:[0.45,0.42,0.39,0.36,0.32], dmg:[7,10,13,17,22], rad:[64,76,88,102,118] },
  hwasal:{ nm:'화살', ic:'🏹', desc:'적을 꿰뚫고 지나간다',
    cd:[1.40,1.22,1.06,0.90,0.75], dmg:[20,27,35,45,58], cnt:[1,2,2,3,3] },
  bell: { nm:'방울', ic:'🔔', desc:'몸 주위를 도는 방울',
    cd:[0.30,0.30,0.30,0.30,0.30], dmg:[11,15,20,26,34], cnt:[2,3,3,4,5], rad:[66,74,82,90,100] },
};
const PASSIVES = {
  jipsin:{ nm:'짚신', ic:'👣', desc:'이동 속도 +12%' },
  tugu:  { nm:'투구', ic:'🪖', desc:'최대 체력 +22' },
  sutdol:{ nm:'숫돌', ic:'🪨', desc:'모든 피해 +15%' },
  buchae:{ nm:'부채', ic:'🪭', desc:'공격 속도 +12%' },
  jaseok:{ nm:'자석', ic:'🧲', desc:'획득 범위 +35%' },
  sansam:{ nm:'산삼', ic:'🌿', desc:'초당 체력 회복 +0.45' },
};

/* ═══════════════ 스프라이트 굽기 ═══════════════ */

const BAKE = {};
function bake(name, scale){
  const s = SPR[name], w = s.d[0].length, h = s.d.length;
  const mk = white => {
    const c = document.createElement('canvas');
    c.width = w*scale; c.height = h*scale;
    const g = c.getContext('2d');
    for(let y=0;y<h;y++) for(let x=0;x<w;x++){
      const ch = s.d[y][x];
      if(ch === '.') continue;
      g.fillStyle = white ? '#ffffff' : s.p[ch];
      g.fillRect(x*scale, y*scale, scale, scale);
    }
    return c;
  };
  return { img:mk(false), lit:mk(true), w:w*scale, h:h*scale };
}
function sprite(name, scale){
  const k = name + '@' + scale;
  if(!BAKE[k]) BAKE[k] = bake(name, scale);
  return BAKE[k];
}

/* ═══════════════ 상태 ═══════════════ */

const cv = document.getElementById('game');
const ctx = cv.getContext('2d', { alpha:false });
let VW = 0, VH = 0, dpr = 1;
const vw = () => VW / ZOOM, vh = () => VH / ZOOM;

let running = false, paused = false, finished = false;
let time = 0, kills = 0, best = 0;
const cam = { x:0, y:0 };
let shake = 0, hitStop = 0, flashScreen = 0;

const P = {
  x:0, y:0, vx:0, vy:0, fx:1, fy:0,
  hp:100, maxHp:100, spd:108,
  lv:1, xp:0, xpNext:5,
  iframe:0, dashCd:0, dashT:0, bob:0, hurtT:0,
  dmgMul:1, cdMul:1, pickR:52, regen:0,
  arms:{}, pass:{}, evo:{},
};

let souls = 0, runSouls = 0;            // 넋 — 판을 넘겨 남는 재화
let actIdx = -1;                        // 지금 몇 막인가
let ult = 0;                            // 필살기 기력 (처치로 찬다)
const ULT_MAX = 300;
let waveIdx = 0, eliteT = 0;
let banner = '', bannerT = 0;

const foes = [], bullets = [], parts = [], drops = [], texts = [], slashes = [];
const near = [];   // 이번 프레임 화면 근처의 적. 무기 판정은 여기서만 한다.
let bossAlive = null, nextBoss = 0, spawnAcc = 0;

/* ═══════════════ 잡동사니 ═══════════════ */
const rnd = (a,b) => a + Math.random()*(b-a);
const rint = (a,b) => Math.floor(rnd(a,b+1));
const clamp = (v,a,b) => v<a?a:v>b?b:v;
const now = () => performance.now()/1000;
const el = id => document.getElementById(id);
function fmtTime(t){
  const m = Math.floor(t/60), s = Math.floor(t%60);
  return m + ':' + String(s).padStart(2,'0');
}

/* ═══════════════ 소리 ═══════════════ */
let muted = false;
const sfx = SND.sfx;

/* ═══════════════ 손맛 ═══════════════ */
let combo = 0, comboT = 0, gemStreak = 0, gemT = 0;
let burstAt = -9, burstN = 0;
let slowT = 0, slowScale = 1, hsCd = 0;
const rings = [], pops = [];

// 히트스톱은 몰아치면 끊겨 보인다. 최소 간격을 둔다.
function freeze(amount){
  if(hsCd > 0) return;
  hitStop = Math.max(hitStop, amount);
  hsCd = 0.14;
}
function slowmo(dur, scale){ slowT = Math.max(slowT, dur); slowScale = scale; }
function ring(x, y, r0, r1, life, col, w){
  if(rings.length > 22) rings.shift();
  rings.push({ x, y, r0, r1, life, max:life, col, w });
}
function popSprite(sprName, scale, x, y, flip){
  if(pops.length > 16) pops.shift();
  pops.push({ spr:sprName, scale, x, y, flip, life:0.17, max:0.17 });
}

/* ═══════════════ 효과 ═══════════════ */
function burst(x, y, n, col, spd, life){
  for(let i=0;i<n && parts.length<MAX_PARTS;i++){
    const a = Math.random()*6.283, s = rnd(spd*0.4, spd);
    parts.push({ x, y, vx:Math.cos(a)*s, vy:Math.sin(a)*s,
      life:life||rnd(0.25,0.5), max:life||0.45, col, r:rnd(2.4,4.8) });
  }
}
function popText(x, y, txt, col, big){
  if(texts.length >= MAX_TEXTS) texts.shift();
  texts.push({ x:x+rnd(-9,9), y, txt, col, life:0.7, vy:-46, big:!!big });
}

/* ═══════════════ 넋 강화 — 판을 넘겨 남는 것 ═══════════════ */
const META = {
  hp:   { nm:'무쇠 몸',   ic:'🛡', desc:'최대 체력 +12',  cost:[8,14,22,34,50] },
  dmg:  { nm:'날 세우기', ic:'⚔', desc:'모든 피해 +6%',  cost:[10,18,28,42,60] },
  spd:  { nm:'가벼운 발', ic:'👣', desc:'이동 속도 +4%',  cost:[8,14,22,34,50] },
  pick: { nm:'넋 부르기', ic:'🧲', desc:'획득 범위 +12%', cost:[10,18,28] },
  luck: { nm:'복',        ic:'🍀', desc:'치명타 확률 +3%', cost:[14,24,38] },
};
let metaLv = {};

function loadMeta(){
  try{
    const raw = JSON.parse(localStorage.getItem('baekgwi.meta') || '{}');
    souls = raw.souls || 0;
    metaLv = raw.lv || {};
    best = raw.best || 0;
  }catch(e){ souls = 0; metaLv = {}; }
}
function saveMeta(){
  try{
    localStorage.setItem('baekgwi.meta', JSON.stringify({ souls, lv:metaLv, best }));
  }catch(e){}
}

/* ═══════════════ 시작 ═══════════════ */
function reset(){
  time = 0; kills = 0;
  running = true; paused = false; finished = false;
  foes.length = bullets.length = parts.length = drops.length = texts.length = slashes.length = 0;
  bossAlive = null; nextBoss = 0; spawnAcc = 0; shake = 0; hitStop = 0; flashScreen = 0;
  rings.length = 0; pops.length = 0;
  combo = 0; comboT = 0; gemStreak = 0; gemT = 0;
  slowT = 0; slowScale = 1; hsCd = 0;
  SND.startMusic(); SND.setIntensity(0);

  P.x = P.y = 0; P.vx = P.vy = 0; P.fx = 1; P.fy = 0;
  P.maxHp = CH().hp + (metaLv.hp||0)*12; P.hp = P.maxHp;
  P.spd = CH().spd * (1 + (metaLv.spd||0)*0.04);
  P.lv = 1; P.xp = 0; P.xpNext = 5;
  P.iframe = 0; P.dashCd = 0; P.dashT = 0; P.hurtT = 0;
  P.dmgMul = CH().dmg * (1 + (metaLv.dmg||0)*0.06);
  P.cdMul = 1;
  P.pickR = 52 * (1 + (metaLv.pick||0)*0.12);
  P.regen = 0;
  critChance = 0.13 + (metaLv.luck||0)*0.03;
  P.arms = {}; P.arms[CH().arm] = 1;   // 고른 인물의 무기 한 자루로 시작
  P.pass = {}; P.evo = {};
  runSouls = 0; waveIdx = 0; eliteT = 55; banner = ''; bannerT = 0;
  evoAnnounced = {}; actIdx = -1; ult = 0; ground = null;
  P.dashHit = new Set();
  el('story').hidden = true;
  el('chest').hidden = true;
  cam.x = 0; cam.y = 0;
  for(const k in ARMS) armT[k] = 0;
  syncHud(true);
}

const armT = {};                        // 무기별 쿨타임 잔여

/* ═══════════════ 입력 ═══════════════ */
const stick = { on:false, id:-1, ox:0, oy:0, dx:0, dy:0 };
const STICK_R = 46;
const keys = {};

function stickVec(){
  if(stick.on){
    const len = Math.hypot(stick.dx, stick.dy);
    if(len < 7) return [0,0];
    const n = Math.min(len, STICK_R) / STICK_R;
    return [stick.dx/len*n, stick.dy/len*n];
  }
  let x = (keys.d||keys.ArrowRight?1:0) - (keys.a||keys.ArrowLeft?1:0);
  let y = (keys.s||keys.ArrowDown?1:0) - (keys.w||keys.ArrowUp?1:0);
  const l = Math.hypot(x,y);
  return l ? [x/l, y/l] : [0,0];
}

function bindInput(){
  const dashBtn = el('dash');

  cv.addEventListener('pointerdown', e => {
    if(!running || paused) return;
    e.preventDefault();
    if(stick.on) return;
    const r = cv.getBoundingClientRect();
    stick.on = true; stick.id = e.pointerId;
    stick.ox = e.clientX - r.left; stick.oy = e.clientY - r.top;
    stick.dx = 0; stick.dy = 0;
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove', e => {
    if(!stick.on || e.pointerId !== stick.id) return;
    const r = cv.getBoundingClientRect();
    stick.dx = (e.clientX - r.left) - stick.ox;
    stick.dy = (e.clientY - r.top)  - stick.oy;
  });
  const drop = e => { if(e.pointerId === stick.id){ stick.on = false; stick.id = -1; } };
  cv.addEventListener('pointerup', drop);
  cv.addEventListener('pointercancel', drop);

  dashBtn.addEventListener('pointerdown', e => { e.preventDefault(); e.stopPropagation(); dash(); });

  window.addEventListener('keydown', e => {
    keys[e.key] = true;
    if(e.key === ' ' || e.key === 'Shift'){ e.preventDefault(); dash(); }
    if(e.key === 'f' || e.key === 'F'){ e.preventDefault(); fireUlt(); }
    if(/^[wasd]$/i.test(e.key) || e.key.startsWith('Arrow')) e.preventDefault();
  });
  window.addEventListener('keyup', e => { keys[e.key] = false; });
}

/* 필살기 — 이 게임에서 유일하게 '내가 직접' 때리는 순간 */
function fireUlt(){
  if(!running || paused || ult < ULT_MAX) return;
  ult = 0;
  const dmg = (45 + P.lv*7) * P.dmgMul;
  slowmo(0.9, 0.28);
  flashScreen = 1.0;
  shake = 20;
  hitStop = 0.14;
  ring(P.x, P.y, 10, 760, 1.0, '#ffffff', 14);
  ring(P.x, P.y, 10, 560, 0.8, '#ffd35c', 9);
  burst(P.x, P.y, 60, '#fff3c4', 420, 0.9);
  popText(P.x, P.y - 54, '백귀참', '#ffd35c', true);
  for(let i=foes.length-1;i>=0;i--){
    const f = foes[i];
    const dx = f.x-P.x, dy = f.y-P.y;
    if(dx*dx+dy*dy > 720*720) continue;
    const d = Math.hypot(dx,dy) || 1;
    hurtFoe(f, dmg * (f.boss ? 0.55 : 1), dx/d*420, dy/d*420);
  }
  sfx.win();
}

function dash(){
  if(!running || paused || P.dashCd > 0) return;
  const [dx,dy] = stickVec();
  const ux = dx || P.fx, uy = dy || P.fy;
  const l = Math.hypot(ux,uy) || 1;
  P.dashT = 0.16; P.dashCd = 2.4; P.iframe = Math.max(P.iframe, 0.32);
  P.dashHit = new Set();
  P.vx = ux/l * 620; P.vy = uy/l * 620;
  burst(P.x, P.y, 14, '#8fd0ff', 170, 0.32);
  ring(P.x, P.y, 4, 52, 0.26, '#8fd0ff', 3);
  sfx.dash();
}

/* ═══════════════ 갱신 ═══════════════ */

function updatePlayer(dt){
  const [dx,dy] = stickVec();
  if(P.dashT > 0){
    P.dashT -= dt;
    P.x += P.vx*dt; P.y += P.vy*dt;
    P.vx *= 0.86; P.vy *= 0.86;
    if(parts.length < MAX_PARTS) burst(P.x, P.y, 1, '#6fb6ff', 40, 0.25);
    // 몸을 부딪쳐 지나가며 벤다 — 회피가 곧 공격이 된다
    const dmg = (18 + P.lv*2.2) * P.dmgMul;
    for(let i=near.length-1;i>=0;i--){
      const f = near[i];
      if(f.hp <= 0 || P.dashHit.has(f)) continue;
      const ddx = f.x-P.x, ddy = f.y-P.y;
      if(ddx*ddx + ddy*ddy > (f.r+22)*(f.r+22)) continue;
      P.dashHit.add(f);
      hurtFoe(f, dmg, ddx*3.2, ddy*3.2);
    }
  } else {
    const sp = P.spd;
    P.x += dx*sp*dt; P.y += dy*sp*dt;
    if(dx||dy){ P.fx = dx; P.fy = dy; P.bob += dt*11; } else P.bob += dt*3;
  }
  P.dashCd = Math.max(0, P.dashCd - dt);
  P.iframe = Math.max(0, P.iframe - dt);
  P.hurtT = Math.max(0, P.hurtT - dt);
  if(P.regen) P.hp = Math.min(P.maxHp, P.hp + P.regen*dt);

  // 카메라는 진행 방향으로 살짝 앞서 본다
  const leadX = P.x + dx*34, leadY = P.y + dy*34;
  cam.x += (leadX - cam.x) * Math.min(1, dt*7);
  cam.y += (leadY - cam.y) * Math.min(1, dt*7);
}

function foeCap(){ return Math.min(MAX_FOES, 90 + Math.floor(time*0.62)); }
function spawnRate(){
  const t = time;
  let r = 3.4 + t*0.10 + (t>180 ? (t-180)*0.16 : 0);
  // 최종보스전은 물량전이 아니라 결투여야 한다. 잡것을 오히려 줄인다.
  if(bossAlive && bossAlive.final) r *= 0.5;
  return r;
}
function foePool(){
  const act = ACTS[Math.max(0, actIdx)];
  const out = [];
  for(const k of act.pool) if(time >= FOES[k].from) out.push(k);
  return out.length ? out : ['honbul'];
}
/* 플레이어 화력은 무기·진화·보조가 곱해져 기하급수로 큰다.
   적을 선형으로 올리면 후반이 그냥 산책이 된다. */
function hpCurve(t){ return 0.62 * Math.pow(1.43, t/60); }
function dmgCurve(t){ return Math.pow(1.20, t/60); }

function spawnFoe(key, ang, dist, opt){
  if(foes.length >= foeCap()) return;
  const d = FOES[key];
  const mul = hpCurve(time);
  opt = opt || {};
  const a = ang !== undefined ? ang : Math.random()*6.283;
  // 화면이 세로로 길어서 원으로 뿌리면 좌우가 너무 멀다. 화면 모양대로 타원에 놓는다.
  const m = dist || 1;
  const rx = (vw()*0.62 + 40) * m, ry = (vh()*0.60 + 40) * m;
  const elite = !!opt.elite;
  const shrink = opt.shrink || 1;
  const hp = d.hp * mul * (elite ? 4.5 : 1) * shrink;
  const f = {
    k:key, spr:d.spr,
    x: opt.x !== undefined ? opt.x : cam.x + Math.cos(a)*rx,
    y: opt.y !== undefined ? opt.y : cam.y + Math.sin(a)*ry,
    hp, maxHp:hp,
    spd: d.spd * rnd(0.9,1.1) * (elite ? 0.82 : 1) * (1 + time/900),
    dmg: d.dmg * dmgCurve(time) * (elite ? 1.7 : 1),
    xp: d.xp * (elite ? 8 : 1) * shrink,
    r: d.r * (elite ? 1.45 : 1) * (shrink < 1 ? 0.72 : 1),
    kx:0, ky:0, flash:0, bellCd:0, bob:Math.random()*6.283, boss:false,
    scale: elite ? 4 : (shrink < 1 ? 2 : 3),
    orbit: rnd(-0.85,0.85),
    ai: d.ai, act:0, dash:0, tele:0, elite, child: shrink < 1,
  };
  foes.push(f);
  return f;
}
function spawnBoss(b){
  const a = Math.random()*6.283;
  const f = {
    k:'boss', spr:b.spr, x:cam.x+Math.cos(a)*(vw()*0.55+40), y:cam.y+Math.sin(a)*(vh()*0.5+40),
    hp:b.hp*hpCurve(b.at), maxHp:b.hp*hpCurve(b.at),
    spd:b.spd, dmg:b.dmg*dmgCurve(b.at), xp:b.xp, r:b.r, rage:0,
    kx:0, ky:0, flash:0, bellCd:0, bob:0, orbit:0, boss:true, nm:b.nm, scale:b.scale, ring:2.5,
    ai:'chase', act:0, dash:0, tele:0, elite:false, child:false, final:b.at>=RUN_TIME,
  };
  foes.push(f); bossAlive = f;
  el('bossbar').hidden = false;
  el('bossname').textContent = b.nm;
  shake = 14; flashScreen = 0.6; slowmo(0.5, 0.5);
  ring(f.x, f.y, 12, 200, 0.8, '#ff5d5d', 6);
  sfx.boss();
  announce(b.nm);
  popText(P.x, P.y-40, b.line || (b.nm + ' 등장'), '#ff5d5d', true);
}

const WAVES = [
  { t:45,  label:'혼불 무리',   key:'honbul',  n:16 },
  { t:88,  label:'허수아비 떼', key:'heosu',   n:14 },
  { t:152, label:'정예 출현',   elite:1 },
  { t:225, label:'해골 돌격',   key:'haegol',  n:20 },
  { t:262, label:'정예 둘',     elite:2 },
  { t:300, label:'그슨대 무리', key:'geuseun', n:8 },
  { t:330, label:'정예 셋',     elite:3 },
  { t:352, label:'해골 파도',   key:'haegol',  n:24 },
];

function announce(text){ banner = text; bannerT = 2.4; }

/* 막이 바뀌면 땅빛과 나오는 것이 바뀌고, 짧은 이야기 판이 뜬다 */
function enterAct(i){
  actIdx = i;
  const A = ACTS[i];
  ground = null;                       // 다음 그리기에서 새 땅빛으로 굽는다
  showStory(A.nm, [A.line], '계 속');
  ring(P.x, P.y, 10, 300, 0.9, '#ffffff', 6);
  flashScreen = 0.5;
}

let storyThen = null;
function showStory(title, lines, btn, then){
  paused = true;
  storyThen = then || null;
  el('storyTitle').textContent = title;
  el('storyBody').innerHTML = lines.map(l => `<p>${l}</p>`).join('');
  el('storyOk').textContent = btn || '계 속';
  el('story').hidden = false;
}
function closeStory(){
  el('story').hidden = true;
  const f = storyThen; storyThen = null;
  if(f) f(); else paused = false;
}

function spawnElite(){
  // 초반에는 혼불밖에 없다. 걸러서 비면 있는 것 중에 뽑는다.
  let pool = foePool().filter(k => FOES[k].from > 0);
  if(!pool.length) pool = foePool();
  if(!pool.length) return;
  const f = spawnFoe(pool[rint(0,pool.length-1)], undefined, undefined, { elite:true });
  if(f){
    ring(f.x, f.y, 8, 90, 0.6, '#f0b23c', 4);
    announce('정예 ' + FOES[f.k].nm);
    sfx.boss();
  }
}

function updateSpawns(dt){
  if(nextBoss < BOSSES.length && time >= BOSSES[nextBoss].at && !bossAlive){
    spawnBoss(BOSSES[nextBoss]); nextBoss++;
  }
  bannerT = Math.max(0, bannerT - dt);

  for(let i=ACTS.length-1;i>=0;i--){
    if(time >= ACTS[i].t && actIdx < i){ enterAct(i); break; }
  }

  // 짜여진 사건들 — 밋밋하게 흘러가지 않도록 리듬을 준다
  while(waveIdx < WAVES.length && time >= WAVES[waveIdx].t){
    const w = WAVES[waveIdx++];
    announce(w.label);
    if(w.elite){ for(let i=0;i<w.elite;i++) spawnElite(); }
    else {
      const base = Math.random()*6.283;
      for(let i=0;i<w.n;i++) spawnFoe(w.key, base + i/w.n*6.283, 0.92);
    }
  }
  // 그 사이사이에도 정예가 하나씩
  if(time > 62){
    eliteT -= dt;
    if(eliteT <= 0){ eliteT = 55; spawnElite(); }
  }
  spawnAcc += spawnRate()*dt;
  const pool = foePool();
  while(spawnAcc >= 1){
    // 가끔 한쪽에서 무리로 몰려온다 (등장 예산에서 인원수만큼 뺀다)
    if(spawnAcc >= 9 && Math.random() < 0.30){
      spawnAcc -= 9;
      const a = Math.random()*6.283;
      const k = pool[rint(0,pool.length-1)];
      for(let i=0;i<9;i++) spawnFoe(k, a + rnd(-0.34,0.34));
    } else {
      spawnAcc -= 1;
      spawnFoe(pool[rint(0,pool.length-1)]);
    }
  }
}

let critChance = 0.13;
const CRIT_MUL = 2.3;

function hurtFoe(f, dmg, kx, ky){
  const crit = Math.random() < critChance;
  if(crit) dmg *= CRIT_MUL;
  f.hp -= dmg; f.flash = crit ? 0.20 : 0.12;
  const kb = crit ? 1.9 : 1;
  f.kx += (kx||0)*kb; f.ky += (ky||0)*kb;
  // 초당 수십 대를 때리는 판에서 숫자를 다 띄우면 읽히지도 않고 비싸다.
  // 치명타와 굵은 한 방만 보여 준다.
  if(crit || dmg >= f.maxHp*0.5 || Math.random() < 0.12)
    popText(f.x, f.y - f.r - 6, String(Math.round(dmg)), crit ? '#ffd35c' : '#fff3c4', crit);
  if(f.boss) putStyle('bossfill', 'width', clamp(f.hp/f.maxHp*100,0,100).toFixed(1) + '%');

  if(crit){
    ring(f.x, f.y, 3, 40, 0.24, '#ffd35c', 3.5);
    burst(f.x, f.y, 7, '#ffe9a8', 200, 0.30);
    shake = Math.max(shake, 5);
    if(f.maxHp > 40 || f.hp <= 0) freeze(0.024);
    sfx.crit();
  } else {
    burst(f.x, f.y, 2, '#ffd98a', 90, 0.18);
  }
  if(f.hp <= 0) killFoe(f);
  else if(!crit) sfx.hit(combo);
}
function killFoe(f){
  const i = foes.indexOf(f);
  if(i < 0) return;
  foes.splice(i,1);
  kills++;
  combo++; comboT = 1.8;
  if(ult < ULT_MAX) ult = Math.min(ULT_MAX, ult + (f.boss ? 60 : (f.elite ? 14 : 1)));
  const notable = f.boss || f.elite || f.maxHp > 60;
  // 한 순간에 여럿이 스러지면 그만큼 크게 터뜨린다
  if(now() - burstAt < 0.09){ burstN++; } else { burstN = 1; }
  burstAt = now();
  if(burstN === 6 || burstN === 14 || burstN === 26){
    ring(P.x, P.y, 10, 90 + burstN*7, 0.32, '#ffd35c', 3);
    shake = Math.max(shake, 2 + burstN*0.2);
    popText(f.x, f.y - 26, burstN + ' 처치', '#ffd35c', true);
  }
  // 초당 아흔 마리가 죽는 판이다. 잡것에게까지 시체·충격파를 붙이면
  // 연출이 아니라 부하가 된다. 값나가는 처치에만 몰아준다.
  if(notable) popSprite(f.spr, f.scale, f.x, f.y, P.x < f.x);
  burst(f.x, f.y, f.boss?60:(notable?9:4), f.boss?'#ffd35c':'#ff8b5c', f.boss?300:130);
  if(f.boss){
    sfx.bigKill();
  } else if(f.maxHp > 90){
    ring(f.x, f.y, 4, 46, 0.28, '#ff9a5c', 3);
    shake = Math.max(shake, 3.5);
    freeze(0.035);
    sfx.bigKill();
  } else {
    if(notable) ring(f.x, f.y, 2, 22, 0.16, '#ffb07a', 2);
    sfx.kill(combo);
  }
  if(f.boss){
    bossAlive = null; el('bossbar').hidden = true;
    shake = 22; flashScreen = 0.85; hitStop = 0.30; slowmo(1.1, 0.32);
    ring(f.x, f.y, 10, 260, 0.9, '#ffd35c', 8);
    ring(f.x, f.y, 10, 170, 0.6, '#ffffff', 4);
    for(let i2=0;i2<14;i2++) dropGem(f.x+rnd(-40,40), f.y+rnd(-40,40), 3);
    for(let i2=0;i2<12;i2++) dropItem(f.x+rnd(-50,50), f.y+rnd(-50,50), 'soul');
    dropItem(f.x, f.y, 'heart'); dropItem(f.x+18, f.y, 'magnet');
    dropItem(f.x-22, f.y, 'chest');
    if(f.final){ victory(); return; }
  } else if(f.elite){
    ring(f.x, f.y, 8, 140, 0.6, '#f0b23c', 5);
    shake = Math.max(shake, 9); flashScreen = 0.35; freeze(0.06);
    dropItem(f.x, f.y, 'chest');
    for(let n=0;n<3;n++) dropItem(f.x+rnd(-24,24), f.y+rnd(-24,24), 'soul');
    for(let n=0;n<5;n++) dropGem(f.x+rnd(-30,30), f.y+rnd(-30,30), 2);
  } else {
    if(f.ai === 'split' && !f.child){
      for(let n=0;n<2;n++)
        spawnFoe(f.k, 0, 1, { x:f.x+rnd(-16,16), y:f.y+rnd(-16,16), shrink:0.42 });
    }
    dropGem(f.x, f.y, f.xp>=6 ? 2 : (f.xp>=3 ? 1 : 0));
    if(Math.random() < 0.012) dropItem(f.x, f.y, 'heart');
    if(Math.random() < 0.006) dropItem(f.x, f.y, 'magnet');
    if(Math.random() < 0.008) dropItem(f.x, f.y, 'soul');
  }
  gainXp(f.xp);
}
const MAX_DROPS = 90;
const GEM_XP = [1, 2, 5];   // 처치가 세 배로 늘어난 만큼 한 알의 값은 낮춘다

/* 구슬을 지우면 경험치가 통째로 증발한다. 레벨업이 멈추고 곧 할 일이 없어진다.
   그래서 지우지 않고, 멀리 있는 둘을 하나로 합쳐 굵게 만든다. 총량은 그대로다. */
function trimDrops(){
  let guard = 0;
  while(drops.length > MAX_DROPS && guard++ < 80){
    let a = -1, b = -1, da = -1, db = -1;
    for(let i=0;i<drops.length;i++){
      const d = drops[i];
      if(d.t !== 'gem' || d.pull) continue;
      const dd = (d.x-P.x)*(d.x-P.x) + (d.y-P.y)*(d.y-P.y);
      if(dd > da){ db = da; b = a; da = dd; a = i; }
      else if(dd > db){ db = dd; b = i; }
    }
    if(a < 0 || b < 0) break;
    const keep = Math.min(a,b), gone = Math.max(a,b);
    drops[keep].xp += drops[gone].xp;
    drops.splice(gone, 1);
  }
}
function gemTier(xp){ return xp >= 9 ? 2 : (xp >= 3 ? 1 : 0); }
function dropGem(x, y, tier){
  drops.push({ x, y, t:'gem', xp:GEM_XP[clamp(tier|0,0,2)],
    vx:rnd(-40,40), vy:rnd(-40,40), pull:0 });
  trimDrops();
}
function dropItem(x,y,kind){ drops.push({ x, y, t:kind, vx:rnd(-30,30), vy:rnd(-30,30), pull:0 }); }

function updateFoes(dt){
  const cull = Math.max(vw(), vh()) * 1.6;
  for(let i=foes.length-1;i>=0;i--){
    const f = foes[i];
    if(!f.boss && (f.x-cam.x)**2 + (f.y-cam.y)**2 > cull*cull){
      foes.splice(i,1); continue;      // 조용히 사라지고 다시 앞쪽에서 나온다
    }
    f.flash = Math.max(0, f.flash - dt);
    f.bob += dt*7;
    const dx = P.x - f.x, dy = P.y - f.y;
    const d = Math.hypot(dx,dy) || 1;
    let speed = f.spd, hold = false;

    if(f.ai === 'charge'){
      // 도깨비: 잠깐 몸을 웅크렸다가 튀어나온다
      f.act -= dt;
      if(f.dash > 0){
        f.dash -= dt; speed = f.spd * 3.4;
        if(parts.length < MAX_PARTS && Math.random() < 0.5)
          burst(f.x, f.y, 1, '#ff8b5c', 40, 0.22);
      } else if(f.tele > 0){
        f.tele -= dt; hold = true; f.flash = Math.max(f.flash, 0.06);
        if(f.tele <= 0){ f.dash = 0.42; f.act = 2.6; sfx.hit(0); }
      } else if(d < 230 && f.act <= 0){
        f.tele = 0.45;
        ring(f.x, f.y, 3, 30, 0.45, '#ff8b5c', 2);
      }
    } else if(f.ai === 'shoot'){
      // 불여우: 사거리 안에서는 멈춰 서서 불덩이를 뱉는다
      f.act -= dt;
      if(d < 250){
        hold = d < 190;
        if(f.act <= 0){
          f.act = 1.9;
          const a2 = Math.atan2(dy,dx);
          bullets.push({ foe:true, x:f.x, y:f.y, vx:Math.cos(a2)*180, vy:Math.sin(a2)*180,
            dmg:f.dmg*0.55, life:2.6, r:7, spr:'bulssi', glow:'#ff7a2e' });
          burst(f.x, f.y, 4, '#ffb85c', 90, 0.25);
        }
      }
    }

    if(!hold){
      const ang = Math.atan2(dy,dx) + f.orbit * clamp(d/260, 0, 1);
      f.x += Math.cos(ang)*speed*dt;
      f.y += Math.sin(ang)*speed*dt;
    }
    f.x += f.kx*dt; f.y += f.ky*dt;
    f.kx *= 0.86; f.ky *= 0.86;

    // 보스는 가끔 탄막을 뿌린다
    if(f.boss){
      // 최종보스는 오래 끌수록 사나워진다. 버티기로는 못 이긴다.
      if(f.final){
        f.rage += dt;
        if(f.rage > 14){ f.rage = 0; f.enr = (f.enr||0) + 1;
          if(f.enr <= 6){ f.dmg *= 1.07; f.ring = Math.min(f.ring, 1.0); } }
      }
      f.ring -= dt;
      if(f.ring <= 0){
        f.ring = 3.4;
        const n = 12;
        for(let k=0;k<n;k++){
          const a = k/n*6.283 + Math.random()*0.2;
          bullets.push({ foe:true, x:f.x, y:f.y, vx:Math.cos(a)*118, vy:Math.sin(a)*118,
            dmg:f.dmg*0.6, life:3.4, r:7, spr:'sajabolt', glow:'#ff4a3c' });
        }
      }
    }

    // 접촉 피해
    if(d < f.r + 9 && P.iframe <= 0){
      P.hp -= f.dmg; P.iframe = 0.50; P.hurtT = 0.34;
      shake = Math.max(shake, 6 + f.dmg*0.22);
      freeze(0.055);
      ring(P.x, P.y, 6, 44, 0.22, '#ff5252', 3);
      combo = 0;
      sfx.hurt();
      popText(P.x, P.y-26, '-'+Math.round(f.dmg), '#ff6b6b', true);
      if(P.hp/P.maxHp < 0.28) slowmo(0.3, 0.45);
      if(P.hp <= 0){ gameOver(); return; }
    }
  }

  // 겹침 밀어내기는 눈에 보이는 것들끼리만 한다. 430마리 전수 비교는 프레임을
  // 무너뜨리고, 화면 밖에서 겹치는 건 어차피 보이지 않는다.
  near.length = 0;
  const nx0 = vw()*0.75, ny0 = vh()*0.75;
  for(let i=0;i<foes.length;i++){
    const f = foes[i];
    if(Math.abs(f.x-cam.x) < nx0 && Math.abs(f.y-cam.y) < ny0) near.push(f);
  }
  for(let i=near.length-1;i>=0;i--){
    const f = near[i];
    for(let j=i-1;j>=0;j--){
      const o = near[j];
      const ax = o.x-f.x, ay = o.y-f.y;
      const rr = f.r + o.r;
      const dd = ax*ax + ay*ay;
      if(dd > 0.01 && dd < rr*rr){
        const dl = Math.sqrt(dd), push = (rr-dl)*0.5;
        const px = ax/dl*push, py = ay/dl*push;
        o.x += px; o.y += py; f.x -= px; f.y -= py;
      }
    }
  }
}

/* ── 무기 ── */
function nearestFoe(x, y, maxD){
  let best = null, bd = (maxD||1e9)**2;
  for(const f of near){
    if(f.hp <= 0) continue;
    const dx = f.x-x, dy = f.y-y, d = dx*dx+dy*dy;
    if(d < bd){ bd = d; best = f; }
  }
  return best;
}
function armDmg(key, lv){ return ARMS[key].dmg[lv-1] * P.dmgMul; }

function updateArms(dt){
  for(const key in P.arms){
    const lv = P.arms[key], A = ARMS[key];
    armT[key] -= dt;
    if(key === 'bell'){ continue; }   // 방울은 상시 회전
    if(armT[key] > 0) continue;
    armT[key] = A.cd[lv-1] * P.cdMul;
    fireArm(key, lv);
  }
  // 방울: 궤도 위에서 접촉 판정
  if(P.arms.bell){
    const lv = P.arms.bell, A = ARMS.bell, E = P.evo.bell;
    const n = A.cnt[lv-1] + (E ? 3 : 0), rad = A.rad[lv-1] * (E ? 1.3 : 1);
    bellAng += dt*(E ? 4.2 : 3.1);
    for(let i=0;i<n;i++){
      const a = bellAng + i/n*6.283;
      const bx = P.x + Math.cos(a)*rad, by = P.y + Math.sin(a)*rad;
      for(let j=near.length-1;j>=0;j--){
        const f = near[j];
        if(f.hp <= 0 || f.bellCd > 0) continue;
        const dx = f.x-bx, dy = f.y-by;
        if(dx*dx+dy*dy < (f.r+9)**2){
          f.bellCd = E ? 0.3 : 0.45;
          hurtFoe(f, armDmg('bell',lv)*(E?2:1), (f.x-P.x)*1.2, (f.y-P.y)*1.2);
          if(E) ring(bx, by, 3, 40, 0.22, '#f5c542', 2.5);
        }
      }
    }
  }
  for(let i=near.length-1;i>=0;i--) if(near[i].bellCd > 0) near[i].bellCd -= dt;
}
let bellAng = 0;

function explode(x, y, dmg, rad){
  ring(x, y, 6, rad, 0.3, '#ff9a3c', 4);
  burst(x, y, 10, '#ffb85c', 200, 0.35);
  for(let i=near.length-1;i>=0;i--){
    const g = near[i];
    if(g.hp <= 0) continue;
    const dx = g.x-x, dy = g.y-y;
    if(dx*dx+dy*dy < rad*rad) hurtFoe(g, dmg, dx*1.4, dy*1.4);
  }
}

function fireArm(key, lv){
  const A = ARMS[key];
  const E = P.evo[key];
  if(key === 'geom'){
    const rng = A.rng[lv-1] * (E ? 1.5 : 1), arc = E ? 6.29 : A.arc[lv-1];
    const t = nearestFoe(P.x, P.y, rng*1.7);
    const base = t ? Math.atan2(t.y-P.y, t.x-P.x) : Math.atan2(P.fy, P.fx);
    const mul = E ? 2.4 : 1;
    slashes.push({ x:P.x, y:P.y, a:base, arc, rng, life:0.18, max:0.18 });
    for(let i=near.length-1;i>=0;i--){
      const f = near[i];
      if(f.hp <= 0) continue;
      const dx = f.x-P.x, dy = f.y-P.y;
      const d = Math.hypot(dx,dy);
      if(d > rng + f.r) continue;
      let da = Math.atan2(dy,dx) - base;
      while(da > Math.PI) da -= 6.283;
      while(da < -Math.PI) da += 6.283;
      if(Math.abs(da) > arc/2) continue;
      hurtFoe(f, armDmg('geom',lv)*mul, dx/d*(E?300:180), dy/d*(E?300:180));
    }
    shake = Math.max(shake, E ? 3.5 : 1.6);
  }
  else if(key === 'bujeok'){
    const n = E ? A.cnt[lv-1] + 5 : A.cnt[lv-1];
    ring(P.x, P.y, 3, E ? 40 : 20, 0.16, '#ffe9a8', 2);
    for(let i=0;i<n;i++){
      let a;
      if(E) a = i/n*6.283 + time*2.2;                  // 만다라 — 소용돌이로 퍼진다
      else {
        const t = nearestFoe(P.x, P.y, 460);
        a = t ? Math.atan2(t.y-P.y, t.x-P.x) + rnd(-0.25,0.25) : Math.random()*6.283;
      }
      bullets.push({ x:P.x, y:P.y, vx:Math.cos(a)*(E?260:210), vy:Math.sin(a)*(E?260:210),
        dmg:armDmg('bujeok',lv)*(E?1.8:1), life:E?3.2:2.6, r:7, spr:'bujeok', home:1, spin:0,
        pierce: E?2:0, hitSet: E?new Set():null });
    }
  }
  else if(key === 'byeorak'){
    const n = A.cnt[lv-1];
    for(let i=0;i<n;i++){
      const cand = foes.filter(f => Math.abs(f.x-cam.x) < vw()*0.6 && Math.abs(f.y-cam.y) < vh()*0.6);
      if(!cand.length) break;
      const f = cand[rint(0,cand.length-1)];
      // 뇌신이면 가장 가까운 적으로 세 번 튄다
      let src = f, hops = E ? 3 : 1;
      const struck = new Set();
      while(hops-- > 0 && src){
        struck.add(src);
        bolts.push({ x:src.x, y:src.y, life:0.24 });
        ring(src.x, src.y, 6, 64, 0.3, '#bfe3ff', 4);
        const rad = E ? 56 : 42;
        for(let j=near.length-1;j>=0;j--){
          const g = near[j];
          if(g.hp <= 0) continue;
          const dx = g.x-src.x, dy = g.y-src.y;
          if(dx*dx+dy*dy < rad*rad) hurtFoe(g, armDmg('byeorak',lv)*(E?1.5:1), dx*2, dy*2);
        }
        burst(src.x, src.y, 12, '#bfe3ff', 170, 0.35);
        if(!hops) break;
        let best = null, bd = 200*200;
        for(const g of foes){
          if(struck.has(g)) continue;
          const dx = g.x-src.x, dy = g.y-src.y, dd = dx*dx+dy*dy;
          if(dd < bd){ bd = dd; best = g; }
        }
        src = best;
      }
    }
    shake = Math.max(shake, 3);
  }
  else if(key === 'bulti'){
    const rad = A.rad[lv-1] * (E ? 1.5 : 1);
    const dm = armDmg('bulti',lv) * (E ? 1.8 : 1);
    for(let i=near.length-1;i>=0;i--){
      const f = near[i];
      if(f.hp <= 0) continue;
      const dx = f.x-P.x, dy = f.y-P.y;
      if(dx*dx+dy*dy >= (rad+f.r)**2) continue;
      const alive = f.hp > 0;
      hurtFoe(f, dm, dx*0.6, dy*0.6);
      // 겁화 — 태워 죽이면 터진다
      if(E && alive && f.hp <= 0) explode(f.x, f.y, dm*1.4, 74);
    }
  }
  else if(key === 'hwasal'){
    const n = E ? 8 : A.cnt[lv-1];
    ring(P.x, P.y, 3, E ? 44 : 24, 0.15, '#e8e2cf', 2);
    const t = nearestFoe(P.x, P.y, 700);
    const base = t ? Math.atan2(t.y-P.y, t.x-P.x) : Math.atan2(P.fy,P.fx);
    for(let i=0;i<n;i++){
      const a = E ? base + i/n*6.283 : base + (i - (n-1)/2)*0.18;
      bullets.push({ x:P.x, y:P.y, vx:Math.cos(a)*430, vy:Math.sin(a)*430,
        dmg:armDmg('hwasal',lv)*(E?1.7:1), life:E?1.9:1.4, r:6, spr:'hwasal',
        pierce:99, hitSet:new Set(), ang:a });
    }
  }
}
const bolts = [];

function updateBullets(dt){
  for(let i=bullets.length-1;i>=0;i--){
    const b = bullets[i];
    b.life -= dt;
    if(b.home){
      const t = nearestFoe(b.x, b.y, 300);
      if(t){
        const a = Math.atan2(t.y-b.y, t.x-b.x);
        const sp = Math.hypot(b.vx,b.vy);
        b.vx += (Math.cos(a)*sp - b.vx) * Math.min(1, dt*4.5);
        b.vy += (Math.sin(a)*sp - b.vy) * Math.min(1, dt*4.5);
      }
      b.spin += dt*9;
    }
    b.x += b.vx*dt; b.y += b.vy*dt;
    if(b.foe && parts.length < MAX_PARTS && Math.random() < 0.6){
      burst(b.x, b.y, 1, b.glow, 22, 0.26);
    }
    if(b.life <= 0){ bullets.splice(i,1); continue; }

    if(b.foe){
      const dx = P.x-b.x, dy = P.y-b.y;
      if(dx*dx+dy*dy < (b.r+9)**2 && P.iframe <= 0){
        P.hp -= b.dmg; P.iframe = 0.50; P.hurtT = 0.34;
        shake = Math.max(shake, 6 + b.dmg*0.2);
        freeze(0.05); combo = 0; sfx.hurt();
        ring(P.x, P.y, 6, 40, 0.2, '#ff5252', 3);
        popText(P.x, P.y-26, '-'+Math.round(b.dmg), '#ff6b6b', true);
        bullets.splice(i,1);
        if(P.hp <= 0){ gameOver(); return; }
      }
      continue;
    }
    for(const f of near){
      if(f.hp <= 0) continue;
      if(b.hitSet && b.hitSet.has(f)) continue;
      const dx = f.x-b.x, dy = f.y-b.y;
      if(dx*dx+dy*dy > (f.r+b.r)**2) continue;
      hurtFoe(f, b.dmg, b.vx*0.45, b.vy*0.45);
      burst(b.x, b.y, 4, '#ffd98a', 90, 0.22);
      if(b.pierce){ b.hitSet.add(f); b.pierce--; if(b.pierce<=0){ bullets.splice(i,1); } }
      else { bullets.splice(i,1); }
      break;
    }
  }
  for(let i=bolts.length-1;i>=0;i--){ bolts[i].life -= dt; if(bolts[i].life<=0) bolts.splice(i,1); }
  for(let i=slashes.length-1;i>=0;i--){ slashes[i].life -= dt; if(slashes[i].life<=0) slashes.splice(i,1); }
}

function updateDrops(dt){
  for(let i=drops.length-1;i>=0;i--){
    const d = drops[i];
    d.x += d.vx*dt; d.y += d.vy*dt;
    d.vx *= 0.90; d.vy *= 0.90;
    const dx = P.x-d.x, dy = P.y-d.y;
    const dist = Math.hypot(dx,dy);
    // 구슬은 언제나 주인을 향해 흘러온다. 안 그러면 도망치는 사이 제 경험치를
    // 뒤에 버리게 되고, 레벨업이 멎어 할 일이 없어진다.
    if(d.t === 'gem' && dist > P.pickR){
      const pull = Math.min(340, 130 + Math.max(0, dist - 240) * 0.7);
      d.x += dx/dist*pull*dt; d.y += dy/dist*pull*dt;
    }
    if(dist < P.pickR || d.pull){
      d.pull = 1;
      const s = 260 + (P.pickR - dist)*3;
      d.x += dx/dist*s*dt; d.y += dy/dist*s*dt;
    }
    if(dist < 14){
      drops.splice(i,1);
      if(d.t === 'gem'){
        gemStreak++; gemT = 0.5;
        gainXp(d.xp);
        sfx.gem(gemStreak);
      } else if(d.t === 'heart'){
        P.hp = Math.min(P.maxHp, P.hp + 28);
        popText(P.x, P.y-30, '+28', '#7bffa0', true);
        ring(P.x, P.y, 4, 40, 0.3, '#7bffa0', 3);
        sfx.heal();
      } else if(d.t === 'magnet'){
        for(const o of drops) o.pull = 1;
        popText(P.x, P.y-30, '전부 끌어당김', '#8fd0ff', true);
        ring(P.x, P.y, 8, 300, 0.5, '#8fd0ff', 4);
        sfx.heal();
      } else if(d.t === 'soul'){
        runSouls++; gemStreak++; gemT = 0.5;
        popText(P.x, P.y-30, '넋 +1', '#8fd0ff');
        sfx.gem(gemStreak);
      } else if(d.t === 'chest'){
        openChest();
        return;
      }
    }
  }
}

function gainXp(n){
  P.xp += n;
  while(P.xp >= P.xpNext){
    P.xp -= P.xpNext; P.lv++;
    P.xpNext = P.lv < 10 ? 5 + P.lv*4 : Math.round(P.xpNext*1.155) + 8;
    offerCards();
  }
}

function updateFx(dt){
  for(let i=rings.length-1;i>=0;i--){
    rings[i].life -= dt;
    if(rings[i].life <= 0) rings.splice(i,1);
  }
  for(let i=pops.length-1;i>=0;i--){
    pops[i].life -= dt;
    if(pops[i].life <= 0) pops.splice(i,1);
  }
}

function updateParts(dt){
  for(let i=parts.length-1;i>=0;i--){
    const p = parts[i];
    p.life -= dt;
    if(p.life <= 0){ parts.splice(i,1); continue; }
    p.x += p.vx*dt; p.y += p.vy*dt;
    p.vx *= 0.93; p.vy *= 0.93;
  }
  for(let i=texts.length-1;i>=0;i--){
    const t = texts[i];
    t.life -= dt; t.y += t.vy*dt; t.vy *= 0.92;
    if(t.life <= 0) texts.splice(i,1);
  }
}

/* ═══════════════ 레벨업 카드 ═══════════════ */
const EVO_ARM = 4, EVO_PASS = 2;      // 6분 안에 실제로 닿아야 보상이 된다
function evoReady(k){
  return P.arms[k] >= EVO_ARM && !P.evo[k] && (P.pass[EVO[k].need]||0) >= EVO_PASS;
}

/* 카드는 균등 추첨이 아니다. 키우던 것과 그 진화 재료가 더 자주 나온다.
   안 그러면 '부채가 끝까지 안 떠서' 무너지는 판이 절반이 된다. */
function buildOptions(){
  const opts = [];
  const owned = Object.keys(P.arms).length;
  const wanted = {};                       // 내 무기의 진화 재료
  for(const k in P.arms) if(!P.evo[k]) wanted[EVO[k].need] = 1;

  for(const k in P.arms) if(P.arms[k] < 5)
    opts.push({ kind:'arm', k, w: P.arms[k] < EVO_ARM ? 3.2 : 1.2 });
  if(owned < 6)
    for(const k in ARMS) if(!P.arms[k])
      opts.push({ kind:'new', k, w: owned < 3 ? 1.8 : 0.7 });
  for(const k in PASSIVES) if((P.pass[k]||0) < 5)
    opts.push({ kind:'pass', k,
      w: (wanted[k] && (P.pass[k]||0) < EVO_PASS) ? 3.2 : 1 });
  return opts;
}

/* 가중치대로 하나 뽑아 목록에서 빼낸다 */
function drawOption(opts){
  let total = 0;
  for(const o of opts) total += o.w;
  let r = Math.random() * total;
  for(let i=0;i<opts.length;i++){
    r -= opts[i].w;
    if(r <= 0) return opts.splice(i,1)[0];
  }
  return opts.pop();
}

function offerCards(){
  const opts = buildOptions();
  const pick = [];
  // 진화가 준비됐으면 무조건 한 장 띄운다 — 이게 이 게임의 보상이다
  for(const k in P.arms) if(evoReady(k)){ pick.push({ kind:'evo', k }); break; }
  while(pick.length < 3 && opts.length) pick.push(drawOption(opts));
  if(!pick.length){ P.hp = Math.min(P.maxHp, P.hp+30); return; }

  paused = true;
  flashScreen = 0.55;
  ring(P.x, P.y, 8, 190, 0.6, '#ffd35c', 5);
  burst(P.x, P.y, 26, '#ffe9a8', 240, 0.6);
  shake = Math.max(shake, 6);
  sfx.level();
  const box = el('cards');
  box.innerHTML = '';
  for(const o of pick) box.appendChild(makeCard(o, () => takeCard(o)));
  el('levelup').hidden = false;
  el('lvnum').textContent = P.lv;
}

function cardInfo(o){
  if(o.kind === 'evo'){
    const e = EVO[o.k];
    return { ic:e.ic, nm:e.nm, tag:'진화', desc:e.desc, cls:' evo' };
  }
  if(o.kind === 'pass'){
    const q = PASSIVES[o.k];
    let hint = '';
    for(const w2 in EVO) if(EVO[w2].need === o.k && P.arms[w2] && !P.evo[w2])
      hint = ` · <i>${ARMS[w2].nm}</i> 진화 재료`;
    return { ic:q.ic, nm:q.nm, tag:'Lv '+((P.pass[o.k]||0)+1), desc:q.desc + hint, cls:'' };
  }
  const w = ARMS[o.k], e = EVO[o.k];
  const need = `<i>${PASSIVES[e.need].nm} ${EVO_PASS}단</i>과 ${EVO_ARM}단이면 진화`;
  if(o.kind === 'new')
    return { ic:w.ic, nm:w.nm, tag:'새 무기', desc:`${w.desc} · ${need}`, cls:' fresh' };
  const lv = P.arms[o.k]+1;
  const tail = P.evo[o.k] ? '' : ` · ${need}`;
  return { ic:w.ic, nm:w.nm, tag:'Lv '+lv, desc:w.desc + tail, cls:'' };
}
function makeCard(o, onPick){
  const c = cardInfo(o);
  const card = document.createElement('button');
  card.className = 'card' + c.cls;
  card.innerHTML =
    `<span class="ci">${c.ic}</span>` +
    `<span class="cn">${c.nm}</span>` +
    `<span class="cl">${c.tag}</span>` +
    `<span class="cd">${c.desc}</span>`;
  if(onPick) card.addEventListener('click', onPick);
  else card.disabled = true;
  return card;
}

/* 정예를 잡으면 나오는 상자 — 고르는 게 아니라 한꺼번에 받는다 */
function openChest(){
  paused = true;
  const opts = buildOptions();
  const got = [];
  for(let i=0;i<3 && opts.length;i++){
    const o = drawOption(opts);
    got.push(cardInfo(o));
    applyPick(o);
  }
  if(!got.length){ P.hp = Math.min(P.maxHp, P.hp + 40); paused = false; return; }
  const box = el('chestList');
  box.innerHTML = '';
  for(const c of got){
    const row = document.createElement('div');
    row.className = 'loot';
    row.innerHTML = `<span class="ci">${c.ic}</span><span class="cn">${c.nm}</span>` +
                    `<span class="cl">${c.tag}</span>`;
    box.appendChild(row);
  }
  el('chest').hidden = false;
  flashScreen = 0.7;
  ring(P.x, P.y, 8, 220, 0.7, '#f0b23c', 6);
  sfx.level();
  syncHud();
}

let evoAnnounced = {};
function checkEvoHint(){
  for(const k in P.arms){
    if(evoReady(k) && !evoAnnounced[k]){
      evoAnnounced[k] = 1;
      announce(EVO[k].nm + ' 준비');
      ring(P.x, P.y, 8, 150, 0.7, '#ffd35c', 5);
    }
  }
}

function applyPick(o){
  if(o.kind === 'evo'){
    P.evo[o.k] = true;
    popText(P.x, P.y-40, EVO[o.k].nm, '#ffd35c', true);
  }
  else if(o.kind === 'arm') P.arms[o.k]++;
  else if(o.kind === 'new'){ P.arms[o.k] = 1; armT[o.k] = 0; }
  else {
    P.pass[o.k] = (P.pass[o.k]||0) + 1;
    if(o.k === 'jipsin') P.spd *= 1.12;
    if(o.k === 'tugu'){ P.maxHp += 22; P.hp += 22; }
    if(o.k === 'sutdol') P.dmgMul *= 1.15;
    if(o.k === 'buchae') P.cdMul *= 0.88;
    if(o.k === 'jaseok') P.pickR *= 1.35;
    if(o.k === 'sansam') P.regen += 0.45;
  }
}

function takeCard(o){
  applyPick(o);
  if(o.kind === 'evo'){
    flashScreen = 0.9; shake = 12; slowmo(0.7, 0.4);
    ring(P.x, P.y, 10, 260, 0.9, '#ffd35c', 8);
    burst(P.x, P.y, 40, '#ffe9a8', 300, 0.8);
    sfx.win();
  }
  el('levelup').hidden = true;
  paused = false;
  checkEvoHint();
  syncHud(true);
}

/* ═══════════════ 끝 ═══════════════ */
function endRun(title, sub, cls, lines){
  running = false; finished = true; paused = false;
  SND.stopMusic();
  if(lines) el('overStory').innerHTML = lines.map(l => `<p>${l}</p>`).join('');
  const score = Math.round(time*10) + kills*3 + P.lv*40;
  runSouls += Math.floor(kills/25);
  souls += runSouls;
  best = Math.max(best, score);
  saveMeta();
  el('rSoul').textContent = '+' + runSouls;
  el('overTitle').textContent = title;
  el('overTitle').className = cls;
  el('overSub').textContent = sub;
  el('rTime').textContent = fmtTime(time);
  el('rKill').textContent = kills;
  el('rLv').textContent = P.lv;
  el('rScore').textContent = score.toLocaleString('ko-KR');
  el('rBest').textContent = best.toLocaleString('ko-KR');
  el('over').hidden = false;
  el('bossbar').hidden = true;
}
function gameOver(){
  burst(P.x, P.y, 54, '#ff6b6b', 240);
  ring(P.x, P.y, 6, 220, 0.8, '#ff5252', 6);
  shake = 18; slowmo(1.2, 0.25); sfx.dead();
  endRun('쓰러졌다', '', 'bad', STORY.lose);
}
function victory(){
  sfx.win();
  endRun('밤을 넘겼다', '', 'good', STORY.win);
}

/* ═══════════════ HUD ═══════════════ */
/* 매 프레임 DOM을 건드리면 스타일 재계산이 프레임을 잡아먹는다.
   값이 바뀔 때만, 그리고 초당 20번만 쓴다. */
const hud = {};
let hudT = 0;
function put(id, v){
  if(hud[id] === v) return;
  hud[id] = v;
  el(id).textContent = v;
}
function putStyle(id, prop, v){
  const k = id + prop;
  if(hud[k] === v) return;
  hud[k] = v;
  el(id).style.setProperty(prop, v);
}

function syncHud(force){
  if(!force){
    hudT -= 1;
    if(hudT > 0) return;
  }
  hudT = 3;

  putStyle('hpfill', 'width', clamp(P.hp/P.maxHp*100,0,100).toFixed(1) + '%');
  put('hptext', Math.max(0,Math.ceil(P.hp)) + ' / ' + Math.round(P.maxHp));
  putStyle('xpfill', 'width', clamp(P.xp/P.xpNext*100,0,100).toFixed(1) + '%');
  put('lv', P.lv);
  put('clock', fmtTime(time));
  put('kills', kills);
  put('soulnow', runSouls);
  putStyle('dash', '--cd', (P.dashCd/2.4).toFixed(2));
  putStyle('ult', '--fill', (ult/ULT_MAX).toFixed(2));

  const ready = ult >= ULT_MAX;
  if(hud.ultReady !== ready){ hud.ultReady = ready; el('ult').classList.toggle('ready', ready); }

  let html = '';
  for(const k in P.arms){
    if(P.evo[k]) html += `<span class="pip evo">${EVO[k].ic}<b>★</b></span>`;
    else html += `<span class="pip${evoReady(k)?' ready':''}">${ARMS[k].ic}<b>${P.arms[k]}</b></span>`;
  }
  for(const k in P.pass) html += `<span class="pip dim">${PASSIVES[k].ic}<b>${P.pass[k]}</b></span>`;
  if(hud.arms !== html){ hud.arms = html; el('arms').innerHTML = html; }

  const bn = el('banner');
  const showB = bannerT > 0;
  if(hud.bannerOn !== showB){ hud.bannerOn = showB; bn.hidden = !showB; }
  if(showB){
    if(hud.bannerTxt !== banner){ hud.bannerTxt = banner; bn.textContent = banner; }
    putStyle('banner', 'opacity', Math.min(1, bannerT/0.5).toFixed(2));
  }

  const cb = el('combo');
  const showC = combo >= 5;
  if(hud.comboOn !== showC){ hud.comboOn = showC; cb.hidden = !showC; }
  if(showC){
    put('combo', combo + ' 연속');
    putStyle('combo', 'font-size', (0.82 + Math.min(combo,60)*0.012).toFixed(2) + 'rem');
    const hot = combo >= 25;
    if(hud.comboHot !== hot){ hud.comboHot = hot; cb.classList.toggle('hot', hot); }
  }

  SND.setIntensity(bossAlive ? 2 : (time > 200 ? 1 : 0));
}

/* ═══════════════ 그리기 ═══════════════ */
let ground = null, vig = null;
const order = [];
const byY = (a,b) => a.y - b.y;
function makeVig(){
  vig = document.createElement('canvas');
  vig.width = 64; vig.height = 64;
  const g = vig.getContext('2d');
  const rg = g.createRadialGradient(32,32,14,32,32,42);
  rg.addColorStop(0,'rgba(0,0,0,0)');
  rg.addColorStop(1,'rgba(0,0,0,0.55)');
  g.fillStyle = rg; g.fillRect(0,0,64,64);
}
function makeGround(){
  // 타일에는 고운 결만 넣는다. 큰 무늬를 넣으면 반복이 눈에 보인다.
  const A = ACTS[Math.max(0, actIdx)];
  const n = 128, c = document.createElement('canvas');
  c.width = c.height = n;
  const g = c.getContext('2d');
  g.fillStyle = A.ground; g.fillRect(0,0,n,n);
  for(let i=0;i<900;i++){
    g.fillStyle = Math.random()<0.5 ? A.dot1 : A.dot2;
    g.fillRect(Math.floor(Math.random()*n), Math.floor(Math.random()*n), 2, 2);
  }
  ground = ctx.createPattern(c, 'repeat');
}

/* 장식은 월드 좌표를 해시해 흩뿌린다 — 반복 무늬가 생기지 않는다 */
function hash2(x, y){
  let h = (x|0)*374761393 + (y|0)*668265263;
  h = (h ^ (h >> 13)) * 1274126177;
  return ((h ^ (h >> 16)) >>> 0) / 4294967296;
}
const DECAL = 104;
function drawDecals(){
  const x0 = Math.floor((cam.x - vw()/2) / DECAL) - 1;
  const x1 = Math.floor((cam.x + vw()/2) / DECAL) + 1;
  const y0 = Math.floor((cam.y - vh()/2) / DECAL) - 1;
  const y1 = Math.floor((cam.y + vh()/2) / DECAL) + 1;
  for(let cy=y0; cy<=y1; cy++){
    for(let cx=x0; cx<=x1; cx++){
      const h = hash2(cx, cy);
      if(h > 0.55) continue;
      const px = cx*DECAL + hash2(cx+91, cy)*DECAL;
      const py = cy*DECAL + hash2(cx, cy+57)*DECAL;
      const kind = Math.floor(h*3);
      const dc = ACTS[Math.max(0, actIdx)].decal;
      if(kind === 0){            // 풀포기
        ctx.fillStyle = dc;
        ctx.fillRect(px, py, 2, 7); ctx.fillRect(px+4, py+2, 2, 5); ctx.fillRect(px-4, py+3, 2, 4);
      } else if(kind === 1){     // 돌
        ctx.fillStyle = '#2a2350'; ctx.fillRect(px, py, 9, 6);
        ctx.fillStyle = '#382e63'; ctx.fillRect(px, py, 9, 2);
      } else {                   // 마른 나뭇가지
        ctx.fillStyle = '#241d3f';
        ctx.fillRect(px, py, 13, 2); ctx.fillRect(px+4, py-3, 2, 4);
      }
    }
  }
}

function draw(){
  ctx.setTransform(dpr*ZOOM,0,0,dpr*ZOOM,0,0);
  if(!ground){ makeGround(); vig = null; }

  let sx = 0, sy = 0;
  if(shake > 0.1){ sx = rnd(-shake,shake); sy = rnd(-shake,shake); shake *= 0.86; } else shake = 0;

  const W = vw(), H = vh();
  const ox = Math.round(W/2 - cam.x + sx), oy = Math.round(H/2 - cam.y + sy);

  // 바닥
  ctx.save();
  ctx.translate(ox % 128, oy % 128);
  ctx.fillStyle = ground;
  ctx.fillRect(-128, -128, W+256, H+256);
  ctx.restore();
  // 가장자리를 눌러 화면 중앙으로 시선을 모은다
  if(!vig) makeVig();
  ctx.drawImage(vig, 0, 0, W, H);

  ctx.save();
  ctx.translate(ox, oy);
  drawDecals();
  ctx.restore();

  ctx.save();
  ctx.translate(ox, oy);

  // 불티 장막
  if(P.arms.bulti){
    const rad = ARMS.bulti.rad[P.arms.bulti-1];
    const pulse = 1 + Math.sin(time*7)*0.04;
    const grd = ctx.createRadialGradient(P.x,P.y,rad*0.30,P.x,P.y,rad*pulse);
    grd.addColorStop(0,'rgba(255,150,50,0.04)');
    grd.addColorStop(0.62,'rgba(255,120,30,0.26)');
    grd.addColorStop(0.92,'rgba(255,80,20,0.34)');
    grd.addColorStop(1,'rgba(255,60,10,0)');
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(P.x,P.y,rad*pulse,0,6.283); ctx.fill();
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = '#ff9a3c'; ctx.lineWidth = 2.5;
    ctx.setLineDash([9, 7]); ctx.lineDashOffset = -time*70;
    ctx.beginPath(); ctx.arc(P.x,P.y,rad*pulse,0,6.283); ctx.stroke();
    ctx.restore();
    if(parts.length < MAX_PARTS && Math.random() < 0.5){
      const a2 = Math.random()*6.283;
      burst(P.x+Math.cos(a2)*rad, P.y+Math.sin(a2)*rad, 1, '#ff9a3c', 30, 0.4);
    }
  }

  // 습득물
  for(const d of drops){
    const s = d.t === 'gem' ? sprite(['gem1','gem2','gem3'][gemTier(d.xp)], 2)
            : sprite(d.t, d.t === 'chest' ? 4 : (d.t === 'soul' ? 3 : 2));
    if(d.t === 'chest' || d.t === 'soul'){
      ctx.save();
      ctx.globalAlpha = 0.35 + Math.sin(time*5 + d.x)*0.15;
      ctx.fillStyle = d.t === 'chest' ? '#f0b23c' : '#8fd0ff';
      ctx.beginPath(); ctx.arc(d.x, d.y, d.t === 'chest' ? 22 : 13, 0, 6.283); ctx.fill();
      ctx.restore();
    }
    ctx.drawImage(s.img, Math.round(d.x-s.w/2), Math.round(d.y-s.h/2));
  }

  // 그림자 — 하나의 경로로 묶어 한 번에 칠한다 (개별 fill은 마리 수만큼 비싸다)
  ctx.fillStyle = 'rgba(0,0,0,0.30)';
  ctx.beginPath();
  for(let i=0;i<near.length;i++){
    const f = near[i];
    ctx.moveTo(f.x + f.r*0.8, f.y + f.r*0.75);
    ctx.ellipse(f.x, f.y+f.r*0.75, f.r*0.8, f.r*0.32, 0, 0, 6.283);
  }
  ctx.moveTo(P.x+9, P.y+10);
  ctx.ellipse(P.x, P.y+10, 9, 3.6, 0, 0, 6.283);
  ctx.fill();

  // 검기 자국
  for(const s of slashes){
    const k = s.life/s.max;
    ctx.save();
    // shadowBlur는 캔버스에서 가장 비싼 연산이다. 넓고 옅은 획을 한 번 더 긋는 편이 싸다.
    const rr = s.rng*(1.06-0.16*k);
    ctx.lineCap = 'round';
    ctx.globalAlpha = k*0.28;
    ctx.strokeStyle = '#9fd4ff';
    ctx.lineWidth = 34*k + 10;
    ctx.beginPath(); ctx.arc(s.x, s.y, rr, s.a - s.arc/2, s.a + s.arc/2); ctx.stroke();
    ctx.globalAlpha = k*0.95;
    ctx.strokeStyle = '#f2f9ff';
    ctx.lineWidth = 18*k + 4;
    ctx.beginPath(); ctx.arc(s.x, s.y, rr, s.a - s.arc/2, s.a + s.arc/2); ctx.stroke();
    ctx.restore();
  }

  // 방울
  if(P.arms.bell){
    const lv = P.arms.bell, E = P.evo.bell;
    const n = ARMS.bell.cnt[lv-1] + (E ? 3 : 0), rad = ARMS.bell.rad[lv-1] * (E ? 1.3 : 1);
    const s = sprite('bell', E ? 3 : 2);
    for(let i=0;i<n;i++){
      const a = bellAng + i/n*6.283;
      ctx.drawImage(s.img, Math.round(P.x+Math.cos(a)*rad-s.w/2), Math.round(P.y+Math.sin(a)*rad-s.h/2));
    }
  }

  // 적 + 나 (y 순 정렬)
  order.length = 0;
  for(let i=0;i<near.length;i++) order.push(near[i]);
  order.push(P);
  order.sort(byY);
  for(const o of order){
    if(o === P) drawPlayer();
    else drawFoe(o);
  }

  // 스러지는 몸 — 하얗게 부풀며 사라진다
  for(const q of pops){
    const k = 1 - q.life/q.max;
    const sp = sprite(q.spr, q.scale);
    ctx.save();
    ctx.globalAlpha = (1-k)*0.85;
    ctx.translate(q.x, q.y);
    if(q.flip) ctx.scale(-1,1);
    const sc = 1 + k*0.75;
    ctx.scale(sc, sc);
    ctx.drawImage(sp.lit, -sp.w/2, -sp.h/2);
    ctx.restore();
  }
  ctx.globalAlpha = 1;

  // 충격파
  for(const r of rings){
    const k = 1 - r.life/r.max;
    const rad = r.r0 + (r.r1 - r.r0) * (1 - (1-k)*(1-k));
    ctx.globalAlpha = (r.life/r.max) * 0.85;
    ctx.strokeStyle = r.col;
    ctx.lineWidth = Math.max(0.6, r.w * (1 - k*0.75));
    ctx.beginPath(); ctx.arc(r.x, r.y, rad, 0, 6.283); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // 번개
  for(const b of bolts){
    const k = b.life/0.24;
    ctx.save();
    ctx.globalAlpha = k;
    ctx.strokeStyle = '#eaf6ff'; ctx.lineWidth = 7;
    ctx.beginPath();
    let yy = b.y - 220, xx = b.x + rnd(-14,14);
    ctx.moveTo(xx, yy);
    while(yy < b.y){ yy += 30; xx = b.x + rnd(-16,16)*(1-(b.y-yy)/220); ctx.lineTo(xx, yy); }
    ctx.stroke();
    ctx.globalAlpha = k*0.5;
    ctx.fillStyle = '#bfe3ff';
    ctx.beginPath(); ctx.arc(b.x, b.y, 62*(1-k)+18, 0, 6.283); ctx.fill();
    ctx.restore();
  }

  // 탄
  for(const b of bullets){
    if(b.foe){
      const s = sprite(b.spr, 3);
      const a = Math.atan2(b.vy, b.vx);
      const puls = 1 + Math.sin(time*22 + b.x*0.05)*0.10;
      ctx.save();
      ctx.translate(b.x, b.y);
      // 위험하다는 걸 먼저 알리는 붉은 무리
      ctx.globalAlpha = 0.34;
      ctx.fillStyle = b.glow;
      ctx.beginPath(); ctx.arc(0, 0, b.r*2.4*puls, 0, 6.283); ctx.fill();
      ctx.globalAlpha = 1;
      ctx.rotate(a);
      ctx.scale(puls, puls);
      ctx.drawImage(s.img, -s.w/2, -s.h/2);
      ctx.restore();
      continue;
    }
    const s = sprite(b.spr, 2);
    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(b.spr === 'hwasal' ? b.ang : b.spin);
    ctx.drawImage(s.img, -s.w/2, -s.h/2);
    ctx.restore();
  }

  // 파편
  for(const p of parts){
    ctx.globalAlpha = clamp(p.life/p.max,0,1);
    ctx.fillStyle = p.col;
    ctx.fillRect(p.x-p.r/2, p.y-p.r/2, p.r, p.r);
  }
  ctx.globalAlpha = 1;

  // 숫자
  ctx.textAlign = 'center';
  for(const t of texts){
    ctx.globalAlpha = clamp(t.life/0.7,0,1);
    ctx.font = (t.big?'800 21px':'800 15px') + ' system-ui, sans-serif';
    ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.85)';
    ctx.strokeText(t.txt, t.x, t.y);
    ctx.fillStyle = t.col;
    ctx.fillText(t.txt, t.x, t.y);
  }
  ctx.globalAlpha = 1;
  ctx.restore();

  ctx.setTransform(dpr,0,0,dpr,0,0);        // 여기서부터 화면 좌표

  // 화면 밖 보스 방향 표시
  if(bossAlive){
    const dx = bossAlive.x - cam.x, dy = bossAlive.y - cam.y;
    if(Math.abs(dx) > VW/2-30 || Math.abs(dy) > VH/2-30){
      const a = Math.atan2(dy,dx);
      const rx = VW/2 + Math.cos(a)*(Math.min(VW,VH)/2-26);
      const ry = VH/2 + Math.sin(a)*(Math.min(VW,VH)/2-26);
      ctx.save(); ctx.translate(rx,ry); ctx.rotate(a);
      ctx.fillStyle = '#ff5d5d';
      ctx.beginPath(); ctx.moveTo(11,0); ctx.lineTo(-8,-7); ctx.lineTo(-8,7); ctx.closePath(); ctx.fill();
      ctx.restore();
    }
  }

  // 조이스틱
  if(stick.on){
    const len = Math.hypot(stick.dx, stick.dy);
    const n = Math.min(len, STICK_R);
    const kx = len ? stick.dx/len*n : 0, ky = len ? stick.dy/len*n : 0;
    ctx.save();
    ctx.globalAlpha = 0.30;
    ctx.strokeStyle = '#dfe6ff'; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(stick.ox, stick.oy, STICK_R, 0, 6.283); ctx.stroke();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = '#dfe6ff';
    ctx.beginPath(); ctx.arc(stick.ox+kx, stick.oy+ky, 19, 0, 6.283); ctx.fill();
    ctx.restore();
  }

  // 피격 붉은 테두리
  if(P.hurtT > 0 || P.hp/P.maxHp < 0.3){
    const a = P.hurtT>0 ? P.hurtT/0.3*0.55 : (0.3 - P.hp/P.maxHp)*0.9;
    const g2 = ctx.createRadialGradient(VW/2,VH/2,Math.min(VW,VH)*0.32,VW/2,VH/2,Math.max(VW,VH)*0.62);
    g2.addColorStop(0,'rgba(255,0,0,0)');
    g2.addColorStop(1,'rgba(255,20,20,'+clamp(a,0,0.6)+')');
    ctx.fillStyle = g2; ctx.fillRect(0,0,VW,VH);
  }
  if(flashScreen > 0){
    ctx.fillStyle = 'rgba(255,255,255,'+ (flashScreen*0.5) +')';
    ctx.fillRect(0,0,VW,VH);
    flashScreen -= 0.05;
  }
}

function drawPlayer(){
  const s = sprite(charKey, 3);
  const bobY = Math.sin(P.bob)*1.6;
  ctx.save();
  ctx.translate(P.x, P.y + bobY);
  if(P.fx < 0) ctx.scale(-1,1);
  const blink = P.iframe > 0 && Math.floor(P.iframe*24) % 2 === 0;
  ctx.globalAlpha = blink ? 0.45 : 1;
  ctx.drawImage(P.hurtT>0 ? s.lit : s.img, -s.w/2, -s.h/2 - 4);
  ctx.restore();
}
function drawFoe(f){
  const s = sprite(f.spr, f.scale);
  const bobY = Math.sin(f.bob)*(f.boss?2.5:1.8);
  const sq = f.flash > 0 ? 1.18 : 1;
  // 흔한 경우(안 뒤집고 안 맞은 잡것)는 변환 없이 바로 찍는다
  if(sq === 1 && !f.elite && f.tele <= 0 && P.x >= f.x){
    ctx.drawImage(s.img, Math.round(f.x - s.w/2), Math.round(f.y + bobY - s.h/2 - f.r*0.3));
    if(f.hp < f.maxHp && (f.elite || f.maxHp > 60)){
      const w = f.r*1.7;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(f.x-w/2, f.y-f.r-9, w, 3.4);
      ctx.fillStyle = '#ff5d5d';
      ctx.fillRect(f.x-w/2, f.y-f.r-9, w*clamp(f.hp/f.maxHp,0,1), 3.4);
    }
    return;
  }
  if(f.elite){
    ctx.save();
    ctx.globalAlpha = 0.55 + Math.sin(time*6)*0.18;
    ctx.strokeStyle = '#f0b23c'; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.ellipse(f.x, f.y + f.r*0.8, f.r*1.15, f.r*0.45, 0, 0, 6.283); ctx.stroke();
    ctx.restore();
  }
  if(f.tele > 0){
    ctx.save();
    ctx.globalAlpha = 0.75;
    ctx.strokeStyle = '#ff8b5c'; ctx.lineWidth = 2;
    ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(f.x, f.y); ctx.lineTo(P.x, P.y); ctx.stroke();
    ctx.restore();
  }
  ctx.save();
  ctx.translate(f.x, f.y + bobY);
  if(P.x < f.x) ctx.scale(-1,1);
  ctx.scale(sq, 2-sq);
  ctx.drawImage(f.flash>0 ? s.lit : s.img, -s.w/2, -s.h/2 - f.r*0.3);
  ctx.restore();
  // 한 방에 죽는 잡것에게까지 체력바를 그리면 fillRect가 마리 수의 두 배로 늘어난다
  if(!f.boss && f.hp < f.maxHp && (f.elite || f.maxHp > 60)){
    const w = f.r*1.7;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(f.x-w/2, f.y-f.r-9, w, 3.4);
    ctx.fillStyle = '#ff5d5d';
    ctx.fillRect(f.x-w/2, f.y-f.r-9, w*clamp(f.hp/f.maxHp,0,1), 3.4);
  }
}

/* ═══════════════ 루프 ═══════════════ */
let last = 0, acc = 0, turbo = 1;
const STEP = 1/60;

function frame(t){
  requestAnimationFrame(frame);
  const nowS = t/1000;
  let dt = Math.min(0.1, nowS - last);
  last = nowS;

  SND.tick();
  hsCd = Math.max(0, hsCd - dt);
  if(slowT > 0){ slowT -= dt; dt *= slowScale; }
  updateFx(dt);

  if(running && !paused){
    comboT -= dt; if(comboT <= 0) combo = 0;
    gemT -= dt;   if(gemT <= 0) gemStreak = 0;
    if(hitStop > 0){ hitStop -= dt; }
    else {
      acc += dt * turbo;
      let guard = 0;
      while(acc >= STEP && guard++ < 8*turbo){
        acc -= STEP;
        time += STEP;
        updatePlayer(STEP);
        updateSpawns(STEP);
        updateFoes(STEP);
        if(!running) break;
        updateArms(STEP);
        updateBullets(STEP);
        if(!running) break;
        updateDrops(STEP);
        updateParts(STEP);
      }
      syncHud();
    }
  } else {
    updateParts(dt);
  }
  draw();
}

/* ═══════════════ 크기 ═══════════════ */
function resize(){
  const r = cv.getBoundingClientRect();
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  VW = Math.max(1, Math.round(r.width));
  VH = Math.max(1, Math.round(r.height));
  cv.width = Math.round(VW*dpr); cv.height = Math.round(VH*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.imageSmoothingEnabled = false;
}

/* ═══════════════ 인물 고르기 ═══════════════ */
function renderChars(){
  const box = el('chars');
  box.innerHTML = '';
  for(const c of CHARS){
    const btn = document.createElement('button');
    btn.className = 'chr' + (c.k === charKey ? ' on' : '');
    btn.innerHTML =
      `<span class="ci">${c.ic}</span><span class="cn">${c.nm}</span>` +
      `<span class="cd">${c.desc}</span>`;
    btn.addEventListener('click', () => {
      charKey = c.k;
      try{ localStorage.setItem('baekgwi.char', c.k); }catch(e){}
      SND.init(); sfx.ui(); renderChars();
    });
    box.appendChild(btn);
  }
}

/* ═══════════════ 상점 ═══════════════ */
function renderShop(){
  el('soulbank').textContent = souls;
  const box = el('shop');
  box.innerHTML = '';
  for(const k in META){
    const m = META[k], lv = metaLv[k] || 0;
    const maxed = lv >= m.cost.length;
    const cost = maxed ? 0 : m.cost[lv];
    const btn = document.createElement('button');
    btn.className = 'buy' + (maxed ? ' maxed' : (souls >= cost ? '' : ' poor'));
    btn.disabled = maxed || souls < cost;
    btn.innerHTML =
      `<span class="bi">${m.ic}</span>` +
      `<span class="bn">${m.nm} <i>${lv}/${m.cost.length}</i></span>` +
      `<span class="bd">${m.desc}</span>` +
      `<span class="bc">${maxed ? '완성' : cost + ' 넋'}</span>`;
    btn.addEventListener('click', () => {
      if(souls < cost) return;
      souls -= cost; metaLv[k] = lv + 1; saveMeta();
      SND.init(); sfx.heal();
      renderShop();
    });
    box.appendChild(btn);
  }
}

/* ═══════════════ 시작 배선 ═══════════════ */
function boot(){
  resize();
  bindInput();
  loadMeta();
  try{ const c = localStorage.getItem('baekgwi.char'); if(c && CHARS.some(x=>x.k===c)) charKey = c; }catch(e){}
  if(best) el('bestHint').textContent = '최고 ' + best.toLocaleString('ko-KR') + '점';
  renderShop();

  // 오디오는 사용자가 누른 다음에야 열 수 있다
  const begin = () => {
    SND.init(); SND.resume(); SND.setMuted(muted);
    el('title').hidden = true; el('over').hidden = true;
    sfx.ui();
    reset();
    paused = true;                       // 서막을 먼저 보여 준다
    showStory('백귀야행', STORY.intro, '나 선 다', () => {
      paused = false; actIdx = -1;       // 곧바로 1막이 열린다
    });
  };
  el('start').addEventListener('click', begin);
  el('storyOk').addEventListener('click', () => { sfx.ui(); closeStory(); });
  el('ult').addEventListener('pointerdown', e => { e.preventDefault(); e.stopPropagation(); fireUlt(); });
  renderChars();
  el('again').addEventListener('click', () => {
    el('over').hidden = true; el('title').hidden = false;
    renderShop(); renderChars();
    if(best) el('bestHint').textContent = '최고 ' + best.toLocaleString('ko-KR') + '점';
  });
  el('chestOk').addEventListener('click', () => {
    el('chest').hidden = true; paused = false;
  });
  el('mute').addEventListener('click', e => {
    muted = !muted;
    SND.setMuted(muted);
    e.currentTarget.setAttribute('aria-pressed', String(!muted));
    e.currentTarget.textContent = muted ? '🔇' : '🔊';
  });

  window.addEventListener('resize', resize);
  window.addEventListener('orientationchange', () => setTimeout(resize, 150));
  requestAnimationFrame(t => { last = t/1000; requestAnimationFrame(frame); });
}

window.__BG = { P, foes, drops, bullets, get time(){return time}, get kills(){return kills},
  get running(){return running}, get paused(){return paused}, reset, takeCard, dash,
  get bossAlive(){return bossAlive}, ARMS, PASSIVES, FOES, EVO, SPR,
  get souls(){return souls}, get runSouls(){return runSouls}, META,
  openChest, spawnElite, get waveIdx(){return waveIdx},
  get ult(){return ult}, fireUlt, get actIdx(){return actIdx},
  setChar(k){ charKey = k; }, CHARS, ACTS,
  setTurbo(n){ turbo = Math.max(1, n|0); },
  setStick(x,y){ stick.on = !!(x||y); stick.ox=0; stick.oy=0; stick.dx=x*STICK_R; stick.dy=y*STICK_R; },
  killAll(){ for(let i=foes.length-1;i>=0;i--) killFoe(foes[i]); },
  warp(t){ time = t; } };

boot();
})();
