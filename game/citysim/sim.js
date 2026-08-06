/* 포켓시티 — 시뮬레이션 엔진 (렌더링/UI와 완전히 분리) */
(function (global) {
  'use strict';

  var T = GD.T, BAL = GD.BAL, CATALOG = GD.CATALOG;

  // 코드 -> 카탈로그 항목 역인덱스
  var BY_CODE = {};
  Object.keys(CATALOG).forEach(function (k) { BY_CODE[CATALOG[k].code] = CATALOG[k]; });

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

  // 타일 좌표 기반 안정 해시 (0~1). 매 틱 값이 흔들리지 않도록 결정적으로 만든다.
  function hash2(x, y, salt) {
    var h = (x | 0) * 374761393 + (y | 0) * 668265263 + (salt | 0) * 2246822519;
    h = (h ^ (h >>> 13)) * 1274126177;
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  }

  function isZone(c) { return c === T.R || c === T.C || c === T.I; }
  function isBuilding(c) { return c >= 10; }

  // ---------------------------------------------------------------- 생성
  function create() {
    var w = BAL.W, h = BAL.H, n = w * h;
    var S = {
      w: w, h: h,
      tile: new Uint8Array(n),
      level: new Uint8Array(n),
      streak: new Int8Array(n),      // 연속 성장/쇠퇴 카운터
      money: BAL.START_MONEY,
      tax: BAL.START_TAX,
      month: 0,
      pop: 0, jobsC: 0, jobsI: 0,
      happy: 55,
      powerCap: 0, powerUse: 0, waterCap: 0, waterUse: 0,
      demand: { R: 1, C: 0, I: 0 },
      income: 0, expense: 0,
      unemployment: 0,
      avgLand: 0, avgPollution: 0,
      event: null,
      tierIdx: 0,
      hist: [],
      log: [],
      terrainPark: new Float32Array(n),
      terrainDirty: true,
      fields: {
        land: new Float32Array(n),
        pollution: new Float32Array(n),
        park: new Float32Array(n),
        police: new Float32Array(n),
        fire: new Float32Array(n),
        edu: new Float32Array(n),
        health: new Float32Array(n),
        power: new Uint8Array(n),
        water: new Uint8Array(n)
      }
    };
    generateTerrain(S);
    return S;
  }

  // 강과 숲이 있는 초기 지형
  function generateTerrain(S) {
    var w = S.w, h = S.h, x, y;

    // 굽이치는 강 (세로로 흐름)
    var cx = w * 0.30;
    for (y = 0; y < h; y++) {
      cx += Math.sin(y * 0.19) * 0.85 + Math.sin(y * 0.061) * 0.6;
      cx = clamp(cx, 3, w - 4);
      var width = 2 + (y % 11 === 0 ? 1 : 0);
      for (x = Math.round(cx - width / 2); x <= Math.round(cx + width / 2); x++) {
        if (x >= 0 && x < w) S.tile[y * w + x] = T.WATER;
      }
    }

    // 숲 군락
    for (var i = 0; i < 26; i++) {
      var gx = Math.floor(hash2(i, 7, 1) * w);
      var gy = Math.floor(hash2(i, 13, 2) * h);
      var r = 2 + Math.floor(hash2(i, 3, 3) * 4);
      for (y = gy - r; y <= gy + r; y++) {
        for (x = gx - r; x <= gx + r; x++) {
          if (x < 0 || y < 0 || x >= w || y >= h) continue;
          var d = Math.hypot(x - gx, y - gy);
          if (d <= r && hash2(x, y, 4) > d / (r + 1) * 0.85) {
            var k = y * w + x;
            if (S.tile[k] === T.EMPTY) S.tile[k] = T.TREE;
          }
        }
      }
    }

    // 시작 도로 (맵 중앙 십자로 조금)
    var mx = Math.floor(w * 0.62), my = Math.floor(h / 2);
    for (x = mx - 5; x <= mx + 5; x++) setStart(S, x, my);
    for (y = my - 4; y <= my + 4; y++) setStart(S, mx, y);
  }

  function setStart(S, x, y) {
    if (x < 0 || y < 0 || x >= S.w || y >= S.h) return;
    var k = y * S.w + x;
    if (S.tile[k] !== T.WATER) S.tile[k] = T.ROAD;
  }

  // ---------------------------------------------------------------- 건설/철거
  function canBuild(S, key, x, y) {
    if (x < 0 || y < 0 || x >= S.w || y >= S.h) return { ok: false, why: '맵 밖입니다' };
    var k = y * S.w + x, cur = S.tile[k];
    if (cur === T.WATER) return { ok: false, why: '물 위에는 지을 수 없습니다' };

    if (key === 'bulldoze') {
      if (cur === T.EMPTY) return { ok: false, why: null };
      // 철거는 자금이 없어도 허용한다 — 부채에 빠진 도시가 유지비를 줄일 유일한 탈출구다
      return { ok: true, cost: GD.TOOLS.bulldoze.cost };
    }

    var item = CATALOG[key];
    if (!item) return { ok: false, why: null };
    if (item.unlockPop && S.pop < item.unlockPop) {
      return { ok: false, why: '인구 ' + item.unlockPop.toLocaleString() + '명부터 해금됩니다' };
    }
    if (cur === item.code && S.level[k] === 0) return { ok: false, why: null }; // 이미 같은 것
    if (cur === item.code && isZone(cur)) return { ok: false, why: null };

    // 기존 것을 밀고 짓는 경우 철거비 추가
    var extra = (cur === T.EMPTY) ? 0 : (cur === T.TREE ? 2 : GD.TOOLS.bulldoze.cost);
    var cost = item.cost + extra;
    if (S.money < cost) return { ok: false, why: '자금이 부족합니다', cost: cost };
    return { ok: true, cost: cost };
  }

  function build(S, key, x, y) {
    var chk = canBuild(S, key, x, y);
    if (!chk.ok) return chk;
    var k = y * S.w + x;

    if (S.tile[k] === T.TREE) S.terrainDirty = true;   // 숲이 사라지면 쾌적도 재계산

    S.tile[k] = (key === 'bulldoze') ? T.EMPTY : CATALOG[key].code;
    S.level[k] = 0;
    S.streak[k] = 0;
    S.money -= chk.cost;
    return { ok: true, cost: chk.cost };
  }

  // ---------------------------------------------------------------- 보조 조회
  function hasRoadAdj(S, x, y) {
    var w = S.w, h = S.h;
    if (x > 0 && S.tile[y * w + x - 1] === T.ROAD) return true;
    if (x < w - 1 && S.tile[y * w + x + 1] === T.ROAD) return true;
    if (y > 0 && S.tile[(y - 1) * w + x] === T.ROAD) return true;
    if (y < h - 1 && S.tile[(y + 1) * w + x] === T.ROAD) return true;
    return false;
  }

  // 반경 안에 선형 감쇠로 값을 더한다. 매 틱 수만 번 도는 자리라 Math.hypot 대신
  // 제곱거리 비교 + sqrt 한 번만 쓴다.
  function stamp(field, S, x, y, radius, strength) {
    var w = S.w, h = S.h, r2 = radius * radius, inv = 1 / (radius + 0.5);
    var x0 = x - radius; if (x0 < 0) x0 = 0;
    var x1 = x + radius; if (x1 > w - 1) x1 = w - 1;
    var y0 = y - radius; if (y0 < 0) y0 = 0;
    var y1 = y + radius; if (y1 > h - 1) y1 = h - 1;
    for (var j = y0; j <= y1; j++) {
      var dy = j - y, dy2 = dy * dy, row = j * w;
      for (var i = x0; i <= x1; i++) {
        var dx = i - x, d2 = dx * dx + dy2;
        if (d2 > r2) continue;
        field[row + i] += strength * (1 - Math.sqrt(d2) * inv);
      }
    }
  }

  // ---------------------------------------------------------------- 필드 계산
  function computeFields(S) {
    var f = S.fields, n = S.w * S.h, i, x, y, k, c;

    // 강·숲이 주는 쾌적도는 지형이 바뀔 때만 다시 계산한다 (매 틱 도는 건 낭비)
    if (S.terrainDirty) {
      S.terrainPark.fill(0);
      for (y = 0; y < S.h; y++) {
        for (x = 0; x < S.w; x++) {
          c = S.tile[y * S.w + x];
          if (c === T.TREE) stamp(S.terrainPark, S, x, y, 2, 5);
          else if (c === T.WATER) stamp(S.terrainPark, S, x, y, 3, 3);
        }
      }
      S.terrainDirty = false;
    }

    f.pollution.fill(0); f.police.fill(0);
    f.fire.fill(0); f.edu.fill(0); f.health.fill(0);
    f.power.fill(0); f.water.fill(0);
    f.park.set(S.terrainPark);

    var powerCap = 0, waterCap = 0;
    var powerSrc = [], waterSrc = [];

    for (y = 0; y < S.h; y++) {
      for (x = 0; x < S.w; x++) {
        k = y * S.w + x;
        c = S.tile[k];
        if (c === T.TREE || c === T.WATER) continue;   // 위에서 terrainPark로 반영
        if (c === T.I && S.level[k] > 0) {
          stamp(f.pollution, S, x, y, BAL.POLLUTION_RADIUS, BAL.POLLUTION_I * S.level[k]);
          continue;
        }
        if (!isBuilding(c)) continue;

        var item = BY_CODE[c];
        if (!item) continue;
        var working = hasRoadAdj(S, x, y);
        if (!working) continue;

        if (item.power) { powerCap += item.power; powerSrc.push([x, y, item.radius]); }
        if (item.water) { waterCap += item.water; waterSrc.push([x, y, item.radius]); }
        if (item.pollution) stamp(f.pollution, S, x, y, item.radius >> 1, item.pollution);
        if (item.land) stamp(f.park, S, x, y, item.radius, item.land);
        if (item.service === 'police') stamp(f.police, S, x, y, item.radius, 24);
        if (item.service === 'fire') stamp(f.fire, S, x, y, item.radius, 24);
        if (item.service === 'edu') stamp(f.edu, S, x, y, item.radius, 22);
        if (item.service === 'health') stamp(f.health, S, x, y, item.radius, 22);
      }
    }

    // 전력/수도 커버리지 (공급망은 시설 반경으로 단순화)
    var s;
    for (i = 0; i < powerSrc.length; i++) {
      s = powerSrc[i];
      markCircle(f.power, S, s[0], s[1], s[2]);
    }
    for (i = 0; i < waterSrc.length; i++) {
      s = waterSrc[i];
      markCircle(f.water, S, s[0], s[1], s[2]);
    }

    S.powerCap = powerCap;
    S.waterCap = waterCap;

    // 지가
    var sumLand = 0, sumPol = 0, cnt = 0;
    for (k = 0; k < n; k++) {
      var v = 30;
      v += Math.min(38, f.park[k] * 0.85);
      v += Math.min(9, f.police[k] * 0.35);
      v += Math.min(6, f.fire[k] * 0.22);
      v += Math.min(11, f.edu[k] * 0.45);
      v += Math.min(9, f.health[k] * 0.38);
      v -= Math.min(34, f.pollution[k] * 1.15);
      v -= S.crime || 0;
      f.land[k] = clamp(v, 0, 100);
      if (isZone(S.tile[k])) { sumLand += f.land[k]; sumPol += f.pollution[k]; cnt++; }
    }
    S.avgLand = cnt ? sumLand / cnt : 0;
    S.avgPollution = cnt ? sumPol / cnt : 0;
  }

  function markCircle(field, S, x, y, radius) {
    var w = S.w, r2 = radius * radius;
    var x0 = x - radius; if (x0 < 0) x0 = 0;
    var x1 = x + radius; if (x1 > w - 1) x1 = w - 1;
    var y0 = y - radius; if (y0 < 0) y0 = 0;
    var y1 = y + radius; if (y1 > S.h - 1) y1 = S.h - 1;
    for (var j = y0; j <= y1; j++) {
      var dy = j - y, dy2 = dy * dy, row = j * w;
      for (var i = x0; i <= x1; i++) {
        var dx = i - x;
        if (dx * dx + dy2 <= r2) field[row + i] = 1;
      }
    }
  }

  // ---------------------------------------------------------------- 한 달 진행
  function tick(S) {
    var events = [];
    computeFields(S);

    var f = S.fields, w = S.w, h = S.h;
    var x, y, k, c, lv;

    // 1) 수요 = 이번 달 시작 시점의 인구/일자리 기준
    var workers = S.pop * BAL.WORK_RATIO;
    var jobs = S.jobsC + S.jobsI;
    var boom = S.event ? S.event.demand : 0;

    var dR = clamp(((jobs + BAL.COMMUTE_JOBS) - workers) / (jobs + workers + 45) * 1.4 + boom, -1, 1);
    var needC = S.pop * BAL.C_JOB_DEMAND;
    var needI = S.pop * BAL.I_JOB_DEMAND + 6;
    var dC = clamp((needC - S.jobsC) / (needC + 22) + boom, -1, 1);
    var dI = clamp((needI - S.jobsI) / (needI + 22) + boom, -1, 1);
    // 불행한 도시는 사람이 떠난다. 다만 표본이 작은 초기 도시에는 적용하지 않는다.
    if (S.happy < 30 && S.pop > 250) { dR -= 0.4; dC -= 0.2; }
    S.demand = { R: clamp(dR, -1, 1), C: clamp(dC, -1, 1), I: clamp(dI, -1, 1) };

    // 2) 전력/수도 수요 집계 + 부족분 판정
    var powerUse = 0, waterUse = 0;
    for (k = 0; k < w * h; k++) {
      c = S.tile[k];
      if (isZone(c)) {
        lv = S.level[k];
        powerUse += BAL.POWER_PER[lv];
        waterUse += BAL.WATER_PER[lv];
      } else if (isBuilding(c) && c !== T.POWER_COAL && c !== T.POWER_SOLAR) {
        powerUse += 12; waterUse += 8;
      }
    }
    S.powerUse = powerUse;
    S.waterUse = waterUse;
    var pRatio = powerUse > 0 ? Math.min(1, S.powerCap / powerUse) : 1;
    var wRatio = waterUse > 0 ? Math.min(1, S.waterCap / waterUse) : 1;

    // 3) 구역 성장/쇠퇴
    var pop = 0, jobsC = 0, jobsI = 0;
    var zoneCount = 0, blackout = 0, dry = 0, noRoad = 0;

    for (y = 0; y < h; y++) {
      for (x = 0; x < w; x++) {
        k = y * w + x;
        c = S.tile[k];
        if (!isZone(c)) continue;
        zoneCount++;
        lv = S.level[k];

        var road = hasRoadAdj(S, x, y);
        // 공급 부족 시, 타일별 안정 해시로 일부 칸만 단전/단수시킨다
        var powered = f.power[k] === 1 && hash2(x, y, 11) < pRatio;
        var watered = f.water[k] === 1 && hash2(x, y, 22) < wRatio;

        if (!road) noRoad++;
        if (!powered && lv > 0) blackout++;
        if (!watered && lv > 0) dry++;

        var d = c === T.R ? S.demand.R : c === T.C ? S.demand.C : S.demand.I;
        var land = f.land[k];
        var serviced = road && powered && watered;

        if (!serviced) {
          S.streak[k] = Math.max(-9, S.streak[k] - 1);
          if (S.streak[k] <= -3 && lv > 0) { S.level[k] = lv - 1; S.streak[k] = 0; }
        } else {
          var need = BAL.LEVEL_REQ[Math.min(3, lv + 1)];
          var gate = true;
          if (c === T.C && lv + 1 >= 3) gate = f.edu[k] > 7;      // 고급 상업은 교육 필요
          if (c === T.R && lv + 1 >= 3) gate = f.health[k] > 6;   // 고급 주거는 의료 필요

          if (lv < 3 && d > 0.06 && land >= need && gate) {
            S.streak[k] = Math.min(9, S.streak[k] + 1);
            var chance = 0.16 + d * 0.30 + (land - need) * 0.006;
            if (hash2(x, y, S.month) < chance) { S.level[k] = lv + 1; S.streak[k] = 0; }
          } else if (d < -0.28 && lv > 0) {
            S.streak[k] = Math.max(-9, S.streak[k] - 1);
            if (S.streak[k] <= -4) { S.level[k] = lv - 1; S.streak[k] = 0; }
          } else {
            S.streak[k] = 0;
          }
        }

        lv = S.level[k];
        if (c === T.R) pop += BAL.R_CAP[lv];
        else if (c === T.C) jobsC += BAL.C_JOBS[lv];
        else jobsI += BAL.I_JOBS[lv];
      }
    }

    S.pop = pop; S.jobsC = jobsC; S.jobsI = jobsI;

    // 4) 실업률 / 범죄 / 행복도
    // 도시 밖으로 통근하는 인구가 있다고 보고 일자리에 여유분을 더한다.
    // (이게 없으면 일자리가 0인 초기 도시의 실업률이 100%가 되어 행복도가 즉사한다)
    workers = pop * BAL.WORK_RATIO;
    jobs = jobsC + jobsI;
    var jobsEff = jobs + BAL.COMMUTE_JOBS;
    S.unemployment = workers > 0 ? clamp((workers - jobsEff) / workers, 0, 1) : 0;

    var polSum = 0, policeSum = 0, cnt = 0;
    for (k = 0; k < w * h; k++) {
      if (!isZone(S.tile[k]) || S.level[k] === 0) continue;
      polSum += f.pollution[k]; policeSum += f.police[k]; cnt++;
    }
    var avgPolice = cnt ? policeSum / cnt : 0;
    S.crime = clamp(4 + S.unemployment * 22 - avgPolice * 0.5, 0, 30);

    var blackoutRatio = cnt ? blackout / cnt : 0;
    var dryRatio = cnt ? dry / cnt : 0;
    var target = 55
      + (S.avgLand - 42) * 0.55
      - S.unemployment * 55
      - Math.max(0, S.tax - 9) * 3.2
      + Math.max(0, 9 - S.tax) * 1.4
      - blackoutRatio * 30
      - dryRatio * 24
      - S.crime * 0.6
      - (S.money < 0 ? 10 : 0);
    target = clamp(target, 0, 100);
    S.happy = S.happy + (target - S.happy) * 0.34;   // 급변 방지

    // 5) 재정
    var eff = 0.55 + (S.happy / 100) * 0.55;         // 행복도가 낮으면 징수 효율 하락
    var income = (pop * BAL.TAX_R + jobsC * BAL.TAX_C + jobsI * BAL.TAX_I) * S.tax * eff;
    var expense = 0;
    for (k = 0; k < w * h; k++) {
      c = S.tile[k];
      if (c === T.ROAD) expense += CATALOG.road.upkeep;
      else if (isBuilding(c)) { var it = BY_CODE[c]; if (it) expense += it.upkeep; }
    }
    if (S.event && S.event.income) income *= S.event.income;

    S.income = income; S.expense = expense;
    S.money += income - expense;
    // 적자는 부채로 쌓이고 이자가 붙는다. 시설을 강제로 없애지는 않는다 —
    // 발전소가 사라지면 도시 전체가 정전되어 회복이 불가능해지기 때문이다.
    if (S.money < 0) S.money -= -S.money * 0.008;

    // 6) 사건
    S.month++;
    if (S.event) { S.event.left--; if (S.event.left <= 0) { events.push({ t: 'info', m: S.event.end }); S.event = null; } }
    else if (S.month > 24 && hash2(S.month, 99, 5) < 0.035) {
      var good = hash2(S.month, 7, 6) > 0.45;
      S.event = good
        ? { name: '호황', demand: 0.22, income: 1.12, left: 10 + Math.floor(hash2(S.month, 1, 7) * 8), end: '호황이 끝났습니다.' }
        : { name: '불황', demand: -0.26, income: 0.86, left: 8 + Math.floor(hash2(S.month, 2, 8) * 8), end: '불황에서 벗어났습니다.' };
      events.push({ t: good ? 'good' : 'warn', m: (good ? '📈 호황 시작 — 수요와 세수가 늘어납니다.' : '📉 불황 시작 — 수요와 세수가 줄어듭니다.') });
    }

    // 화재: 소방 사각지대에 있는 건물이 드물게 불탄다.
    // 확률은 사각지대 건물 수에 비례한다 — 작은 마을이 대도시만큼 자주 불타면 안 된다.
    if (S.month > 18) {
      var victims = [];
      for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
        k = y * w + x;
        if (isZone(S.tile[k]) && S.level[k] > 0 && f.fire[k] < 3) victims.push(k);
      }
      var fireP = Math.min(0.07, victims.length * 0.0016);
      if (victims.length && hash2(S.month, 41, 9) < fireP) {
        var vk = victims[Math.floor(hash2(S.month, 17, 10) * victims.length)];
        S.tile[vk] = T.RUBBLE; S.level[vk] = 0; S.streak[vk] = 0;
        events.push({ t: 'bad', m: '🔥 화재 발생! 소방 사각지대의 건물이 전소했습니다.' });
      }
    }

    // 부채 경고 (3개월마다 한 번만)
    if (S.money < 0 && S.month % 3 === 0) {
      events.push({
        t: 'bad',
        m: S.money < -8000
          ? '💀 부채 ' + Math.round(-S.money).toLocaleString() + '원. 이자가 불어납니다 — 도구▸철거로 시설을 줄이세요.'
          : '⚠️ 적자입니다. 세율을 올리거나 도로·시설을 줄이세요.'
      });
    }

    // 단계 승급
    var ti = 0;
    for (var i = 0; i < GD.TIERS.length; i++) if (pop >= GD.TIERS[i].pop) ti = i;
    if (ti > S.tierIdx) {
      S.tierIdx = ti;
      events.push({ t: 'good', m: '🎉 ' + GD.TIERS[ti].name + ' 승격! 새 시설이 해금되었습니다.' });
    } else if (ti < S.tierIdx) {
      S.tierIdx = ti;
    }

    // 기록 (24개월마다 1점, 최대 240점)
    if (S.month % 2 === 0) {
      S.hist.push({ m: S.month, pop: pop, money: Math.round(S.money), inc: Math.round(income), exp: Math.round(expense), happy: Math.round(S.happy) });
      if (S.hist.length > 260) S.hist.shift();
    }

    return events;
  }

  // ---------------------------------------------------------------- 조회
  function inspect(S, x, y) {
    if (x < 0 || y < 0 || x >= S.w || y >= S.h) return null;
    var k = y * S.w + x, c = S.tile[k], f = S.fields;
    var name = c === T.EMPTY ? '빈 땅' : c === T.WATER ? '강' : c === T.TREE ? '숲'
      : c === T.ROAD ? '도로' : c === T.RUBBLE ? '잔해'
      : c === T.R ? '주거 구역' : c === T.C ? '상업 구역' : c === T.I ? '공업 구역'
      : (BY_CODE[c] ? BY_CODE[c].name : '?');
    return {
      x: x, y: y, code: c, name: name, level: S.level[k],
      road: hasRoadAdj(S, x, y),
      power: f.power[k] === 1, water: f.water[k] === 1,
      land: Math.round(f.land[k]), pollution: Math.round(f.pollution[k]),
      police: Math.round(f.police[k]), fire: Math.round(f.fire[k]),
      edu: Math.round(f.edu[k]), health: Math.round(f.health[k]),
      capacity: c === T.R ? BAL.R_CAP[S.level[k]] : c === T.C ? BAL.C_JOBS[S.level[k]] : c === T.I ? BAL.I_JOBS[S.level[k]] : 0
    };
  }

  // ---------------------------------------------------------------- 저장
  function serialize(S) {
    return JSON.stringify({
      v: 1, w: S.w, h: S.h,
      tile: Array.from(S.tile), level: Array.from(S.level),
      money: S.money, tax: S.tax, month: S.month, happy: S.happy,
      tierIdx: S.tierIdx, hist: S.hist, event: S.event
    });
  }

  function deserialize(json) {
    var d = JSON.parse(json);
    if (!d || d.w !== BAL.W || d.h !== BAL.H) return null;
    var S = create();
    S.tile.set(d.tile); S.level.set(d.level);
    S.money = d.money; S.tax = d.tax; S.month = d.month; S.happy = d.happy;
    S.tierIdx = d.tierIdx || 0; S.hist = d.hist || []; S.event = d.event || null;
    computeFields(S);
    // 인구/일자리 즉시 복원
    var pop = 0, jc = 0, ji = 0;
    for (var k = 0; k < S.w * S.h; k++) {
      var c = S.tile[k], lv = S.level[k];
      if (c === T.R) pop += BAL.R_CAP[lv];
      else if (c === T.C) jc += BAL.C_JOBS[lv];
      else if (c === T.I) ji += BAL.I_JOBS[lv];
    }
    S.pop = pop; S.jobsC = jc; S.jobsI = ji;
    return S;
  }

  global.Sim = {
    create: create, tick: tick, build: build, canBuild: canBuild,
    inspect: inspect, computeFields: computeFields,
    serialize: serialize, deserialize: deserialize,
    isZone: isZone, isBuilding: isBuilding, byCode: BY_CODE, hasRoadAdj: hasRoadAdj
  };
})(window);
