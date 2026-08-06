/* 백귀야행 — 게임 본체
   조작은 이동과 회피 둘뿐. 공격은 전부 자동으로 나간다. */
'use strict';
(function(){

/* ═══════════════ 자료 ═══════════════ */

const RUN_TIME = 360;                 // 6분
const MAX_FOES = 260, MAX_PARTS = 300, MAX_TEXTS = 44;

const FOES = {
  honbul:  { spr:'honbul',  nm:'혼불',    hp:9,   spd:74, dmg:6,  xp:1, r:8,  from:0   },
  heosu:   { spr:'heosu',   nm:'허수아비', hp:28, spd:58, dmg:9,  xp:2, r:10, from:12  },
  dokkae:  { spr:'dokkae',  nm:'도깨비',  hp:48,  spd:66, dmg:12, xp:3, r:11, from:35  },
  haegol:  { spr:'haegol',  nm:'해골',    hp:36,  spd:88, dmg:11, xp:3, r:10, from:70  },
  yeou:    { spr:'yeou',    nm:'불여우',  hp:80,  spd:97, dmg:15, xp:6, r:11, from:110 },
  geuseun: { spr:'geuseun', nm:'그슨대',  hp:190, spd:52, dmg:22, xp:14, r:15, from:160 },
};
const BOSSES = [
  { spr:'gumiho', nm:'구미호',    hp:2600, spd:38, dmg:24, xp:150, r:22, at:180, scale:5 },
  { spr:'saja',   nm:'저승사자',  hp:7800, spd:33, dmg:32, xp:600, r:26, at:360, scale:6 },
];

// 무기 — lv 1..5. 값은 레벨별 배열.
const ARMS = {
  geom: { nm:'검기', ic:'⚔', desc:'앞쪽을 부채꼴로 벤다',
    cd:[0.80,0.70,0.62,0.55,0.48], dmg:[16,23,31,41,53], arc:[2.1,2.3,2.5,2.7,3.0], rng:[76,84,92,100,110] },
  bujeok:{ nm:'부적', ic:'📜', desc:'적을 쫓아가는 부적',
    cd:[0.90,0.80,0.70,0.62,0.54], dmg:[14,20,27,36,47], cnt:[2,2,3,3,4] },
  byeorak:{ nm:'벼락', ic:'⚡', desc:'하늘에서 내리꽂힌다',
    cd:[2.40,2.10,1.85,1.60,1.35], dmg:[34,46,60,78,100], cnt:[1,2,2,3,4] },
  bulti:{ nm:'불티', ic:'🔥', desc:'몸을 두른 불꽃이 태운다',
    cd:[0.45,0.42,0.39,0.36,0.32], dmg:[7,10,13,17,22], rad:[52,60,68,78,90] },
  hwasal:{ nm:'화살', ic:'🏹', desc:'적을 꿰뚫고 지나간다',
    cd:[1.40,1.22,1.06,0.90,0.75], dmg:[20,27,35,45,58], cnt:[1,2,2,3,3] },
  bell: { nm:'방울', ic:'🔔', desc:'몸 주위를 도는 방울',
    cd:[0.30,0.30,0.30,0.30,0.30], dmg:[11,15,20,26,34], cnt:[2,2,3,3,4], rad:[54,58,62,66,72] },
};
const PASSIVES = {
  jipsin:{ nm:'짚신', ic:'👣', desc:'이동 속도 +12%' },
  tugu:  { nm:'투구', ic:'🪖', desc:'최대 체력 +22' },
  sutdol:{ nm:'숫돌', ic:'🪨', desc:'모든 피해 +15%' },
  buchae:{ nm:'부채', ic:'🪭', desc:'공격 속도 +12%' },
  jaseok:{ nm:'자석', ic:'🧲', desc:'획득 범위 +35%' },
  sansam:{ nm:'산삼', ic:'🌿', desc:'초당 체력 회복 +0.7' },
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
  arms:{}, pass:{},
};

const foes = [], bullets = [], parts = [], drops = [], texts = [], slashes = [];
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
  if(rings.length > 40) rings.shift();
  rings.push({ x, y, r0, r1, life, max:life, col, w });
}
function popSprite(sprName, scale, x, y, flip){
  if(pops.length > 40) pops.shift();
  pops.push({ spr:sprName, scale, x, y, flip, life:0.17, max:0.17 });
}

/* ═══════════════ 효과 ═══════════════ */
function burst(x, y, n, col, spd, life){
  for(let i=0;i<n && parts.length<MAX_PARTS;i++){
    const a = Math.random()*6.283, s = rnd(spd*0.4, spd);
    parts.push({ x, y, vx:Math.cos(a)*s, vy:Math.sin(a)*s,
      life:life||rnd(0.25,0.5), max:life||0.45, col, r:rnd(1.6,3.4) });
  }
}
function popText(x, y, txt, col, big){
  if(texts.length >= MAX_TEXTS) texts.shift();
  texts.push({ x:x+rnd(-9,9), y, txt, col, life:0.7, vy:-46, big:!!big });
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
  P.maxHp = 100; P.hp = 100; P.spd = 108;
  P.lv = 1; P.xp = 0; P.xpNext = 5;
  P.iframe = 0; P.dashCd = 0; P.dashT = 0; P.hurtT = 0;
  P.dmgMul = 1; P.cdMul = 1; P.pickR = 52; P.regen = 0;
  P.arms = { bujeok:1 };               // 부적 한 장으로 시작
  P.pass = {};
  cam.x = 0; cam.y = 0;
  for(const k in ARMS) armT[k] = 0;
  syncHud();
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
    if(/^[wasd]$/i.test(e.key) || e.key.startsWith('Arrow')) e.preventDefault();
  });
  window.addEventListener('keyup', e => { keys[e.key] = false; });
}

function dash(){
  if(!running || paused || P.dashCd > 0) return;
  const [dx,dy] = stickVec();
  const ux = dx || P.fx, uy = dy || P.fy;
  const l = Math.hypot(ux,uy) || 1;
  P.dashT = 0.16; P.dashCd = 2.4; P.iframe = Math.max(P.iframe, 0.32);
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

function spawnRate(){
  const t = time;
  return 1.2 + t*0.028 + (t>200 ? (t-200)*0.05 : 0);   // 초당 마리
}
function foePool(){
  const out = [];
  for(const k in FOES) if(time >= FOES[k].from) out.push(k);
  return out;
}
function spawnFoe(key, ang, dist){
  if(foes.length >= MAX_FOES) return;
  const d = FOES[key];
  const mul = 1 + time/95;
  const a = ang !== undefined ? ang : Math.random()*6.283;
  // 화면이 세로로 길어서 원으로 뿌리면 좌우가 너무 멀다. 화면 모양대로 타원에 놓는다.
  const m = dist || 1;
  const rx = (VW*0.60 + 46) * m, ry = (VH*0.58 + 46) * m;
  foes.push({
    k:key, spr:d.spr, x:cam.x + Math.cos(a)*rx, y:cam.y + Math.sin(a)*ry,
    hp:d.hp*mul, maxHp:d.hp*mul, spd:d.spd*rnd(0.9,1.1), dmg:d.dmg*(1+time/240),
    xp:d.xp, r:d.r, kx:0, ky:0, flash:0, bellCd:0, bob:Math.random()*6.283, boss:false, scale:3,
    orbit:rnd(-0.85,0.85),
  });
}
function spawnBoss(b){
  const a = Math.random()*6.283;
  const f = {
    k:'boss', spr:b.spr, x:cam.x+Math.cos(a)*(VW*0.55+40), y:cam.y+Math.sin(a)*(VH*0.5+40),
    hp:b.hp, maxHp:b.hp, spd:b.spd, dmg:b.dmg, xp:b.xp, r:b.r,
    kx:0, ky:0, flash:0, bellCd:0, bob:0, orbit:0, boss:true, nm:b.nm, scale:b.scale, ring:2.5, final:b.at>=RUN_TIME,
  };
  foes.push(f); bossAlive = f;
  el('bossbar').hidden = false;
  el('bossname').textContent = b.nm;
  shake = 14; flashScreen = 0.6; slowmo(0.5, 0.5);
  ring(f.x, f.y, 12, 200, 0.8, '#ff5d5d', 6);
  sfx.boss();
  popText(P.x, P.y-40, b.nm + ' 등장', '#ff5d5d', true);
}

function updateSpawns(dt){
  if(nextBoss < BOSSES.length && time >= BOSSES[nextBoss].at && !bossAlive){
    spawnBoss(BOSSES[nextBoss]); nextBoss++;
  }
  spawnAcc += spawnRate()*dt;
  const pool = foePool();
  while(spawnAcc >= 1){
    // 가끔 한쪽에서 무리로 몰려온다 (등장 예산에서 인원수만큼 뺀다)
    if(spawnAcc >= 5 && Math.random() < 0.16){
      spawnAcc -= 5;
      const a = Math.random()*6.283;
      for(let i=0;i<5;i++) spawnFoe(pool[rint(0,pool.length-1)], a + rnd(-0.30,0.30));
    } else {
      spawnAcc -= 1;
      spawnFoe(pool[rint(0,pool.length-1)]);
    }
  }
}

const CRIT_CHANCE = 0.13, CRIT_MUL = 2.3;

function hurtFoe(f, dmg, kx, ky){
  const crit = Math.random() < CRIT_CHANCE;
  if(crit) dmg *= CRIT_MUL;
  f.hp -= dmg; f.flash = crit ? 0.20 : 0.12;
  const kb = crit ? 1.9 : 1;
  f.kx += (kx||0)*kb; f.ky += (ky||0)*kb;
  popText(f.x, f.y - f.r - 6, String(Math.round(dmg)), crit ? '#ffd35c' : '#fff3c4', crit);
  if(f.boss) el('bossfill').style.width = clamp(f.hp/f.maxHp*100,0,100) + '%';

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
  popSprite(f.spr, f.scale, f.x, f.y, P.x < f.x);
  burst(f.x, f.y, f.boss?60:9, f.boss?'#ffd35c':'#ff8b5c', f.boss?300:130);
  if(f.boss){
    sfx.bigKill();
  } else if(f.maxHp > 90){
    ring(f.x, f.y, 4, 46, 0.28, '#ff9a5c', 3);
    shake = Math.max(shake, 3.5);
    freeze(0.035);
    sfx.bigKill();
  } else {
    ring(f.x, f.y, 2, 22, 0.16, '#ffb07a', 2);
    sfx.kill(combo);
  }
  if(f.boss){
    bossAlive = null; el('bossbar').hidden = true;
    shake = 22; flashScreen = 0.85; hitStop = 0.30; slowmo(1.1, 0.32);
    ring(f.x, f.y, 10, 260, 0.9, '#ffd35c', 8);
    ring(f.x, f.y, 10, 170, 0.6, '#ffffff', 4);
    for(let i2=0;i2<14;i2++) dropGem(f.x+rnd(-40,40), f.y+rnd(-40,40), 3);
    dropItem(f.x, f.y, 'heart'); dropItem(f.x+18, f.y, 'magnet');
    if(f.final){ victory(); return; }
  } else {
    dropGem(f.x, f.y, f.xp>=6 ? 2 : (f.xp>=3 ? 1 : 0));
    if(Math.random() < 0.012) dropItem(f.x, f.y, 'heart');
    if(Math.random() < 0.006) dropItem(f.x, f.y, 'magnet');
  }
  gainXp(f.xp);
}
function dropGem(x,y,tier){ drops.push({ x, y, t:'gem', tier:clamp(tier|0,0,2), vx:rnd(-40,40), vy:rnd(-40,40), pull:0 }); }
function dropItem(x,y,kind){ drops.push({ x, y, t:kind, vx:rnd(-30,30), vy:rnd(-30,30), pull:0 }); }

function updateFoes(dt){
  const cull = Math.max(VW, VH) * 1.5;
  for(let i=foes.length-1;i>=0;i--){
    const f = foes[i];
    if(!f.boss && (f.x-cam.x)**2 + (f.y-cam.y)**2 > cull*cull){
      foes.splice(i,1); continue;      // 조용히 사라지고 다시 앞쪽에서 나온다
    }
    f.flash = Math.max(0, f.flash - dt);
    f.bob += dt*7;
    const dx = P.x - f.x, dy = P.y - f.y;
    const d = Math.hypot(dx,dy) || 1;
    // 멀리 있을수록 크게 휘어 들어오고, 가까워지면 곧장 달려든다
    const ang = Math.atan2(dy,dx) + f.orbit * clamp(d/260, 0, 1);
    f.x += Math.cos(ang)*f.spd*dt + f.kx*dt;
    f.y += Math.sin(ang)*f.spd*dt + f.ky*dt;
    f.kx *= 0.86; f.ky *= 0.86;

    // 보스는 가끔 탄막을 뿌린다
    if(f.boss){
      f.ring -= dt;
      if(f.ring <= 0){
        f.ring = 3.4;
        const n = 12;
        for(let k=0;k<n;k++){
          const a = k/n*6.283 + Math.random()*0.2;
          bullets.push({ foe:true, x:f.x, y:f.y, vx:Math.cos(a)*118, vy:Math.sin(a)*118,
            dmg:f.dmg*0.6, life:3.4, r:6, col:'#ff6a5c' });
        }
      }
    }

    // 접촉 피해
    if(d < f.r + 9 && P.iframe <= 0){
      P.hp -= f.dmg; P.iframe = 0.42; P.hurtT = 0.34;
      shake = Math.max(shake, 6 + f.dmg*0.22);
      freeze(0.055);
      ring(P.x, P.y, 6, 44, 0.22, '#ff5252', 3);
      combo = 0;
      sfx.hurt();
      popText(P.x, P.y-26, '-'+Math.round(f.dmg), '#ff6b6b', true);
      if(P.hp/P.maxHp < 0.28) slowmo(0.3, 0.45);
      if(P.hp <= 0){ gameOver(); return; }
    }
    // 겹침 방지
    for(let j=i-1;j>=0;j--){
      const o = foes[j];
      const ox = o.x-f.x, oy = o.y-f.y;
      const rr = f.r + o.r;
      const dd = ox*ox + oy*oy;
      if(dd > 0.01 && dd < rr*rr){
        const dl = Math.sqrt(dd), push = (rr-dl)*0.5;
        const nx = ox/dl*push, ny = oy/dl*push;
        o.x += nx; o.y += ny; f.x -= nx; f.y -= ny;
      }
    }
  }
}

/* ── 무기 ── */
function nearestFoe(x, y, maxD){
  let best = null, bd = (maxD||1e9)**2;
  for(const f of foes){
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
    const lv = P.arms.bell, A = ARMS.bell;
    const n = A.cnt[lv-1], rad = A.rad[lv-1];
    bellAng += dt*3.1;
    for(let i=0;i<n;i++){
      const a = bellAng + i/n*6.283;
      const bx = P.x + Math.cos(a)*rad, by = P.y + Math.sin(a)*rad;
      for(let j=foes.length-1;j>=0;j--){
        const f = foes[j];
        if(f.bellCd > 0) continue;
        const dx = f.x-bx, dy = f.y-by;
        if(dx*dx+dy*dy < (f.r+9)**2){
          f.bellCd = 0.45;
          hurtFoe(f, armDmg('bell',lv), (f.x-P.x)*1.2, (f.y-P.y)*1.2);
        }
      }
    }
  }
  for(const f of foes) if(f.bellCd > 0) f.bellCd -= dt;
}
let bellAng = 0;

function fireArm(key, lv){
  const A = ARMS[key];
  if(key === 'geom'){
    const rng = A.rng[lv-1], arc = A.arc[lv-1];
    const t = nearestFoe(P.x, P.y, rng*1.7);
    const base = t ? Math.atan2(t.y-P.y, t.x-P.x) : Math.atan2(P.fy, P.fx);
    slashes.push({ x:P.x, y:P.y, a:base, arc, rng, life:0.18, max:0.18 });
    for(let i=foes.length-1;i>=0;i--){
      const f = foes[i];
      const dx = f.x-P.x, dy = f.y-P.y;
      const d = Math.hypot(dx,dy);
      if(d > rng + f.r) continue;
      let da = Math.atan2(dy,dx) - base;
      while(da > Math.PI) da -= 6.283;
      while(da < -Math.PI) da += 6.283;
      if(Math.abs(da) > arc/2) continue;
      hurtFoe(f, armDmg('geom',lv), dx/d*180, dy/d*180);
    }
    shake = Math.max(shake, 1.6);
  }
  else if(key === 'bujeok'){
    const n = A.cnt[lv-1];
    ring(P.x, P.y, 3, 20, 0.14, '#ffe9a8', 2);
    for(let i=0;i<n;i++){
      const t = nearestFoe(P.x, P.y, 460);
      const a = t ? Math.atan2(t.y-P.y, t.x-P.x) + rnd(-0.25,0.25) : Math.random()*6.283;
      bullets.push({ x:P.x, y:P.y, vx:Math.cos(a)*210, vy:Math.sin(a)*210,
        dmg:armDmg('bujeok',lv), life:2.6, r:7, spr:'bujeok', home:1, spin:0 });
    }
  }
  else if(key === 'byeorak'){
    const n = A.cnt[lv-1];
    for(let i=0;i<n;i++){
      const cand = foes.filter(f => Math.abs(f.x-cam.x) < VW*0.6 && Math.abs(f.y-cam.y) < VH*0.6);
      if(!cand.length) break;
      const f = cand[rint(0,cand.length-1)];
      bolts.push({ x:f.x, y:f.y, life:0.24 });
      ring(f.x, f.y, 6, 64, 0.3, '#bfe3ff', 4);
      for(let j=foes.length-1;j>=0;j--){
        const g = foes[j];
        const dx = g.x-f.x, dy = g.y-f.y;
        if(dx*dx+dy*dy < 42*42) hurtFoe(g, armDmg('byeorak',lv), dx*2, dy*2);
      }
      burst(f.x, f.y, 12, '#bfe3ff', 170, 0.35);
    }
    shake = Math.max(shake, 3);
  }
  else if(key === 'bulti'){
    const rad = A.rad[lv-1];
    for(let i=foes.length-1;i>=0;i--){
      const f = foes[i];
      const dx = f.x-P.x, dy = f.y-P.y;
      if(dx*dx+dy*dy < (rad+f.r)**2) hurtFoe(f, armDmg('bulti',lv), dx*0.6, dy*0.6);
    }
  }
  else if(key === 'hwasal'){
    const n = A.cnt[lv-1];
    ring(P.x, P.y, 3, 24, 0.14, '#e8e2cf', 2);
    const t = nearestFoe(P.x, P.y, 700);
    const base = t ? Math.atan2(t.y-P.y, t.x-P.x) : Math.atan2(P.fy,P.fx);
    for(let i=0;i<n;i++){
      const a = base + (i - (n-1)/2)*0.18;
      bullets.push({ x:P.x, y:P.y, vx:Math.cos(a)*430, vy:Math.sin(a)*430,
        dmg:armDmg('hwasal',lv), life:1.4, r:6, spr:'hwasal', pierce:99, hitSet:new Set(), ang:a });
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
    if(b.life <= 0){ bullets.splice(i,1); continue; }

    if(b.foe){
      const dx = P.x-b.x, dy = P.y-b.y;
      if(dx*dx+dy*dy < (b.r+9)**2 && P.iframe <= 0){
        P.hp -= b.dmg; P.iframe = 0.42; P.hurtT = 0.34;
        shake = Math.max(shake, 6 + b.dmg*0.2);
        freeze(0.05); combo = 0; sfx.hurt();
        ring(P.x, P.y, 6, 40, 0.2, '#ff5252', 3);
        popText(P.x, P.y-26, '-'+Math.round(b.dmg), '#ff6b6b', true);
        bullets.splice(i,1);
        if(P.hp <= 0){ gameOver(); return; }
      }
      continue;
    }
    for(const f of foes){
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
    if(dist < P.pickR || d.pull){
      d.pull = 1;
      const s = 260 + (P.pickR - dist)*3;
      d.x += dx/dist*s*dt; d.y += dy/dist*s*dt;
    }
    if(dist < 14){
      drops.splice(i,1);
      if(d.t === 'gem'){
        gemStreak++; gemT = 0.5;
        gainXp([1,3,9][d.tier]);
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
      }
    }
  }
}

function gainXp(n){
  P.xp += n;
  while(P.xp >= P.xpNext){
    P.xp -= P.xpNext; P.lv++;
    P.xpNext = P.lv < 12 ? 5 + P.lv*4 : Math.round(P.xpNext*1.16) + 6;
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
function offerCards(){
  const opts = [];
  for(const k in P.arms) if(P.arms[k] < 5) opts.push({ kind:'arm', k });
  if(Object.keys(P.arms).length < 6)
    for(const k in ARMS) if(!P.arms[k]) opts.push({ kind:'new', k });
  for(const k in PASSIVES) if((P.pass[k]||0) < 5) opts.push({ kind:'pass', k });

  const pick = [];
  while(pick.length < 3 && opts.length){
    pick.push(opts.splice(rint(0,opts.length-1),1)[0]);
  }
  if(!pick.length){ P.hp = Math.min(P.maxHp, P.hp+30); return; }

  paused = true;
  flashScreen = 0.55;
  ring(P.x, P.y, 8, 190, 0.6, '#ffd35c', 5);
  burst(P.x, P.y, 26, '#ffe9a8', 240, 0.6);
  shake = Math.max(shake, 6);
  sfx.level();
  const box = el('cards');
  box.innerHTML = '';
  for(const o of pick){
    const src = o.kind === 'pass' ? PASSIVES[o.k] : ARMS[o.k];
    const lv = o.kind === 'arm' ? P.arms[o.k]+1 : (o.kind === 'pass' ? (P.pass[o.k]||0)+1 : 1);
    const card = document.createElement('button');
    card.className = 'card' + (o.kind === 'new' ? ' fresh' : '');
    card.innerHTML =
      `<span class="ci">${src.ic}</span>` +
      `<span class="cn">${src.nm}</span>` +
      `<span class="cl">${o.kind==='new' ? '새 무기' : 'Lv '+lv}</span>` +
      `<span class="cd">${src.desc}</span>`;
    card.addEventListener('click', () => takeCard(o));
    box.appendChild(card);
  }
  el('levelup').hidden = false;
  el('lvnum').textContent = P.lv;
}

function takeCard(o){
  if(o.kind === 'arm') P.arms[o.k]++;
  else if(o.kind === 'new'){ P.arms[o.k] = 1; armT[o.k] = 0; }
  else {
    P.pass[o.k] = (P.pass[o.k]||0) + 1;
    if(o.k === 'jipsin') P.spd *= 1.12;
    if(o.k === 'tugu'){ P.maxHp += 22; P.hp += 22; }
    if(o.k === 'sutdol') P.dmgMul *= 1.15;
    if(o.k === 'buchae') P.cdMul *= 0.88;
    if(o.k === 'jaseok') P.pickR *= 1.35;
    if(o.k === 'sansam') P.regen += 0.7;
  }
  el('levelup').hidden = true;
  paused = false;
  syncHud();
}

/* ═══════════════ 끝 ═══════════════ */
function endRun(title, sub, cls){
  running = false; finished = true; paused = false;
  const score = Math.round(time*10) + kills*3 + P.lv*40;
  try{
    best = Math.max(best, score, parseInt(localStorage.getItem('baekgwi.best')||'0',10)||0);
    localStorage.setItem('baekgwi.best', String(best));
  }catch(e){ best = Math.max(best, score); }
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
  shake = 18; slowmo(1.2, 0.25); SND.stopMusic(); sfx.dead();
  endRun('쓰러졌다', '백귀의 밤은 아직 끝나지 않았다.', 'bad');
}
function victory(){
  SND.stopMusic(); sfx.win();
  endRun('밤을 넘겼다', '저승사자를 베고 동이 텄다.', 'good');
}

/* ═══════════════ HUD ═══════════════ */
function syncHud(){
  el('hpfill').style.width = clamp(P.hp/P.maxHp*100,0,100) + '%';
  el('hptext').textContent = Math.max(0,Math.ceil(P.hp)) + ' / ' + Math.round(P.maxHp);
  el('xpfill').style.width = clamp(P.xp/P.xpNext*100,0,100) + '%';
  el('lv').textContent = P.lv;
  el('clock').textContent = fmtTime(time);
  el('dash').style.setProperty('--cd', (P.dashCd/2.4).toFixed(3));

  const cb = el('combo');
  if(combo >= 5){
    cb.hidden = false;
    cb.textContent = combo + ' 연속';
    cb.style.fontSize = (0.82 + Math.min(combo,60)*0.012).toFixed(2) + 'rem';
    cb.classList.toggle('hot', combo >= 25);
  } else if(!cb.hidden) cb.hidden = true;

  SND.setIntensity(bossAlive ? 2 : (time > 200 ? 1 : 0));
  el('kills').textContent = kills;

  const row = el('arms');
  let html = '';
  for(const k in P.arms) html += `<span class="pip">${ARMS[k].ic}<b>${P.arms[k]}</b></span>`;
  for(const k in P.pass) html += `<span class="pip dim">${PASSIVES[k].ic}<b>${P.pass[k]}</b></span>`;
  if(row.innerHTML !== html) row.innerHTML = html;
}

/* ═══════════════ 그리기 ═══════════════ */
let ground = null, vig = null;
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
  const n = 128, c = document.createElement('canvas');
  c.width = c.height = n;
  const g = c.getContext('2d');
  g.fillStyle = '#151129'; g.fillRect(0,0,n,n);
  for(let i=0;i<900;i++){
    g.fillStyle = Math.random()<0.5 ? '#191539' : '#110e24';
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
  const x0 = Math.floor((cam.x - VW/2) / DECAL) - 1;
  const x1 = Math.floor((cam.x + VW/2) / DECAL) + 1;
  const y0 = Math.floor((cam.y - VH/2) / DECAL) - 1;
  const y1 = Math.floor((cam.y + VH/2) / DECAL) + 1;
  for(let cy=y0; cy<=y1; cy++){
    for(let cx=x0; cx<=x1; cx++){
      const h = hash2(cx, cy);
      if(h > 0.55) continue;
      const px = cx*DECAL + hash2(cx+91, cy)*DECAL;
      const py = cy*DECAL + hash2(cx, cy+57)*DECAL;
      const kind = Math.floor(h*3);
      if(kind === 0){            // 풀포기
        ctx.fillStyle = '#22376a';
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
  ctx.setTransform(dpr,0,0,dpr,0,0);
  if(!ground){ makeGround(); vig = null; }

  let sx = 0, sy = 0;
  if(shake > 0.1){ sx = rnd(-shake,shake); sy = rnd(-shake,shake); shake *= 0.86; } else shake = 0;

  const ox = Math.round(VW/2 - cam.x + sx), oy = Math.round(VH/2 - cam.y + sy);

  // 바닥
  ctx.save();
  ctx.translate(ox % 128, oy % 128);
  ctx.fillStyle = ground;
  ctx.fillRect(-128, -128, VW+256, VH+256);
  ctx.restore();
  // 가장자리를 눌러 화면 중앙으로 시선을 모은다
  if(!vig) makeVig();
  ctx.drawImage(vig, 0, 0, VW, VH);

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
    const grd = ctx.createRadialGradient(P.x,P.y,rad*0.35,P.x,P.y,rad*pulse);
    grd.addColorStop(0,'rgba(255,140,40,0.02)');
    grd.addColorStop(0.75,'rgba(255,120,30,0.16)');
    grd.addColorStop(1,'rgba(255,80,20,0)');
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(P.x,P.y,rad*pulse,0,6.283); ctx.fill();
  }

  // 습득물
  for(const d of drops){
    const s = d.t === 'gem' ? sprite(['gem1','gem2','gem3'][d.tier], 2) : sprite(d.t, 2);
    ctx.drawImage(s.img, Math.round(d.x-s.w/2), Math.round(d.y-s.h/2));
  }

  // 그림자
  ctx.fillStyle = 'rgba(0,0,0,0.30)';
  for(const f of foes){
    ctx.beginPath(); ctx.ellipse(f.x, f.y+f.r*0.75, f.r*0.8, f.r*0.32, 0,0,6.283); ctx.fill();
  }
  ctx.beginPath(); ctx.ellipse(P.x, P.y+10, 9, 3.6, 0,0,6.283); ctx.fill();

  // 검기 자국
  for(const s of slashes){
    const k = s.life/s.max;
    ctx.save();
    ctx.globalAlpha = k*0.95;
    ctx.strokeStyle = '#f2f9ff';
    ctx.lineWidth = 11*k + 3;
    ctx.shadowColor = '#9fd4ff'; ctx.shadowBlur = 12*k;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.rng*(1.06-0.16*k), s.a - s.arc/2, s.a + s.arc/2);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.restore();
  }

  // 방울
  if(P.arms.bell){
    const lv = P.arms.bell, n = ARMS.bell.cnt[lv-1], rad = ARMS.bell.rad[lv-1];
    const s = sprite('bell', 2);
    for(let i=0;i<n;i++){
      const a = bellAng + i/n*6.283;
      ctx.drawImage(s.img, Math.round(P.x+Math.cos(a)*rad-s.w/2), Math.round(P.y+Math.sin(a)*rad-s.h/2));
    }
  }

  // 적 + 나 (y 순 정렬)
  const order = foes.slice();
  order.push(P);
  order.sort((a,b) => a.y - b.y);
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
    ctx.strokeStyle = '#eaf6ff'; ctx.lineWidth = 4;
    ctx.beginPath();
    let yy = b.y - 220, xx = b.x + rnd(-14,14);
    ctx.moveTo(xx, yy);
    while(yy < b.y){ yy += 30; xx = b.x + rnd(-16,16)*(1-(b.y-yy)/220); ctx.lineTo(xx, yy); }
    ctx.stroke();
    ctx.globalAlpha = k*0.5;
    ctx.fillStyle = '#bfe3ff';
    ctx.beginPath(); ctx.arc(b.x, b.y, 40*(1-k)+14, 0, 6.283); ctx.fill();
    ctx.restore();
  }

  // 탄
  for(const b of bullets){
    if(b.foe){
      ctx.fillStyle = b.col;
      ctx.beginPath(); ctx.arc(b.x,b.y,b.r,0,6.283); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.beginPath(); ctx.arc(b.x-1,b.y-1,b.r*0.4,0,6.283); ctx.fill();
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
  const s = sprite('musa', 3);
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
  ctx.save();
  ctx.translate(f.x, f.y + bobY);
  if(P.x < f.x) ctx.scale(-1,1);
  ctx.scale(sq, 2-sq);
  ctx.drawImage(f.flash>0 ? s.lit : s.img, -s.w/2, -s.h/2 - f.r*0.3);
  ctx.restore();
  if(!f.boss && f.hp < f.maxHp){
    const w = f.r*1.7;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(f.x-w/2, f.y-f.r-9, w, 3.4);
    ctx.fillStyle = '#ff5d5d';
    ctx.fillRect(f.x-w/2, f.y-f.r-9, w*clamp(f.hp/f.maxHp,0,1), 3.4);
  }
}

/* ═══════════════ 루프 ═══════════════ */
let last = 0, acc = 0;
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
      acc += dt;
      let guard = 0;
      while(acc >= STEP && guard++ < 5){
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

/* ═══════════════ 시작 배선 ═══════════════ */
function boot(){
  resize();
  bindInput();
  try{ best = parseInt(localStorage.getItem('baekgwi.best')||'0',10)||0; }catch(e){}
  if(best) el('bestHint').textContent = '최고 ' + best.toLocaleString('ko-KR') + '점';

  // 오디오는 사용자가 누른 다음에야 열 수 있다
  const begin = () => {
    SND.init(); SND.resume(); SND.setMuted(muted);
    el('title').hidden = true; el('over').hidden = true;
    sfx.ui(); reset();
  };
  el('start').addEventListener('click', begin);
  el('again').addEventListener('click', begin);
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
  get bossAlive(){return bossAlive}, ARMS, PASSIVES, FOES, SPR,
  setStick(x,y){ stick.on = !!(x||y); stick.ox=0; stick.oy=0; stick.dx=x*STICK_R; stick.dy=y*STICK_R; },
  killAll(){ for(let i=foes.length-1;i>=0;i--) killFoe(foes[i]); },
  warp(t){ time = t; } };

boot();
})();
