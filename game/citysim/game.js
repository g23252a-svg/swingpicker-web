/* 포켓시티 — 입력 처리 · 게임 루프 · 저장 */
(function (global) {
  'use strict';

  var BAL = GD.BAL;
  var SAVE_KEY = 'pocketcity.save.v1';
  var SEEN_KEY = 'pocketcity.seen.v1';

  var S = null;
  var speed = 1;
  var overlay = 'none';
  var tool = null;
  var cursor = null;
  var acc = 0, last = 0, sinceSave = 0;
  var cv;

  // ------------------------------------------------------------ 초기화
  function boot() {
    cv = document.getElementById('map');
    Render.init(cv);

    var loaded = null;
    try {
      var raw = localStorage.getItem(SAVE_KEY);
      if (raw) loaded = Sim.deserialize(raw);
    } catch (e) { loaded = null; }

    S = loaded || Sim.create();
    Sim.computeFields(S);
    Render.centerOn(Math.floor(BAL.W * 0.62), Math.floor(BAL.H / 2));

    UI.init(API);
    UI.updateHUD();
    bindInput();

    if (!loaded) {
      try { localStorage.setItem(SAVE_KEY, Sim.serialize(S)); } catch (e) {}
    }
    if (!localStorage.getItem(SEEN_KEY)) {
      try { localStorage.setItem(SEEN_KEY, '1'); } catch (e) {}
      UI.openHelp();
    } else {
      UI.toast(loaded ? '저장된 도시를 불러왔습니다' : '새 도시를 시작합니다', 'good');
    }

    global.addEventListener('resize', function () { Render.resize(); });
    global.addEventListener('orientationchange', function () { setTimeout(function () { Render.resize(); }, 120); });
    document.addEventListener('visibilitychange', function () { if (document.hidden) save(); });
    global.addEventListener('pagehide', save);

    last = performance.now();
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------ 루프
  function frame(t) {
    var dt = Math.min(250, t - last);
    last = t;

    if (speed > 0) {
      acc += dt * speed;
      var step = BAL.TICK_MS;
      var guard = 0;
      while (acc >= step && guard++ < 4) {
        acc -= step;
        var events = Sim.tick(S);
        events.forEach(function (e) { UI.toast(e.m, e.t); });
        sinceSave++;
        if (sinceSave >= 12) { save(); sinceSave = 0; }
      }
      UI.updateHUD();
      UI.refreshPaletteState();
    }

    Render.clampCam(S);
    Render.draw(S, { overlay: overlay, cursor: cursor });
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------ 입력
  var pointers = new Map();
  var mode = null;            // 'pan' | 'paint' | 'pinch'
  var panStart = null, pinchStart = null;
  var lastTile = null, moved = 0, downAt = 0;

  function bindInput() {
    cv.addEventListener('pointerdown', onDown, { passive: false });
    cv.addEventListener('pointermove', onMove, { passive: false });
    cv.addEventListener('pointerup', onUp, { passive: false });
    cv.addEventListener('pointercancel', onUp, { passive: false });
    cv.addEventListener('wheel', onWheel, { passive: false });
    cv.addEventListener('contextmenu', function (e) { e.preventDefault(); });
  }

  function onDown(e) {
    e.preventDefault();
    cv.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved = 0; downAt = performance.now();

    if (pointers.size === 2) {
      startPinch();
      cursor = null;
      return;
    }
    if (pointers.size > 2) return;

    if (tool && tool !== 'inspect') {
      mode = 'paint';
      var t = Render.screenToTile(e.clientX, e.clientY);
      lastTile = t;
      applyAt(t.x, t.y);
    } else {
      mode = 'pan';
      panStart = { cx: Render.cam.x, cy: Render.cam.y, px: e.clientX, py: e.clientY };
    }
  }

  function onMove(e) {
    if (!pointers.has(e.pointerId)) return;
    e.preventDefault();
    var prev = pointers.get(e.pointerId);
    moved += Math.abs(e.clientX - prev.x) + Math.abs(e.clientY - prev.y);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (pointers.size >= 2) { doPinch(); return; }

    if (mode === 'pan' && panStart) {
      Render.cam.x = panStart.cx - (e.clientX - panStart.px) / Render.cam.zoom;
      Render.cam.y = panStart.cy - (e.clientY - panStart.py) / Render.cam.zoom;
    } else if (mode === 'paint') {
      var t = Render.screenToTile(e.clientX, e.clientY);
      if (!lastTile || t.x !== lastTile.x || t.y !== lastTile.y) {
        line(lastTile, t).forEach(function (p) { applyAt(p.x, p.y); });
        lastTile = t;
      }
    }
  }

  function onUp(e) {
    pointers.delete(e.pointerId);
    if (pointers.size === 1) {
      // 두 손가락 중 하나를 뗐다 — 남은 손가락으로 패닝 이어가기
      var only = pointers.values().next().value;
      mode = 'pan';
      panStart = { cx: Render.cam.x, cy: Render.cam.y, px: only.x, py: only.y };
      pinchStart = null;
      return;
    }
    if (pointers.size > 0) return;

    // 탭 판정
    if (mode === 'pan' && moved < 10 && performance.now() - downAt < 320 && tool === 'inspect') {
      var t = Render.screenToTile(e.clientX, e.clientY);
      UI.showInspect(Sim.inspect(S, t.x, t.y));
    }

    mode = null; panStart = null; pinchStart = null; lastTile = null; cursor = null;
    UI.updateHUD();
    UI.refreshPaletteState();
  }

  function startPinch() {
    var p = Array.from(pointers.values());
    pinchStart = {
      dist: Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y) || 1,
      zoom: Render.cam.zoom,
      mx: (p[0].x + p[1].x) / 2,
      my: (p[0].y + p[1].y) / 2,
      cx: Render.cam.x,
      cy: Render.cam.y
    };
    mode = 'pinch';
  }

  function doPinch() {
    if (!pinchStart) { startPinch(); return; }
    var p = Array.from(pointers.values());
    var dist = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y) || 1;
    var mx = (p[0].x + p[1].x) / 2, my = (p[0].y + p[1].y) / 2;

    var z0 = Render.cam.zoom;
    var z1 = Math.max(Render.cam.min, Math.min(Render.cam.max, pinchStart.zoom * (dist / pinchStart.dist)));

    // 두 손가락 중점이 가리키던 월드 좌표를 고정한 채 확대
    var wx = (pinchStart.mx - Render.w / 2) / z0 + Render.cam.x;
    var wy = (pinchStart.my - Render.h / 2) / z0 + Render.cam.y;
    Render.cam.zoom = z1;
    Render.cam.x = wx - (mx - Render.w / 2) / z1;
    Render.cam.y = wy - (my - Render.h / 2) / z1;
    pinchStart.mx = mx; pinchStart.my = my;
    pinchStart.dist = dist; pinchStart.zoom = z1;
  }

  function onWheel(e) {
    e.preventDefault();
    var z0 = Render.cam.zoom;
    var z1 = Math.max(Render.cam.min, Math.min(Render.cam.max, z0 * (e.deltaY < 0 ? 1.12 : 0.89)));
    var wx = (e.clientX - Render.w / 2) / z0 + Render.cam.x;
    var wy = (e.clientY - Render.h / 2) / z0 + Render.cam.y;
    Render.cam.zoom = z1;
    Render.cam.x = wx - (e.clientX - Render.w / 2) / z1;
    Render.cam.y = wy - (e.clientY - Render.h / 2) / z1;
  }

  // 두 타일 사이를 잇는 칸 목록 (Bresenham)
  function line(a, b) {
    if (!a) return [b];
    var pts = [], x0 = a.x, y0 = a.y, x1 = b.x, y1 = b.y;
    var dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0);
    var sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    var err = dx + dy, guard = 0;
    while (guard++ < 400) {
      if (!(x0 === a.x && y0 === a.y)) pts.push({ x: x0, y: y0 });
      if (x0 === x1 && y0 === y1) break;
      var e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
    return pts;
  }

  // ------------------------------------------------------------ 건설 적용
  function applyAt(x, y) {
    if (!tool) return;
    var res = Sim.build(S, tool, x, y);
    cursor = [{ x: x, y: y, ok: res.ok }];
    if (res.ok) {
      Sim.computeFields(S);
      UI.updateHUD();
    } else if (res.why) {
      UI.toast(res.why, 'warn');
      if (res.why.indexOf('자금') >= 0) selectToolInternal(null);
    }
  }

  // ------------------------------------------------------------ 저장
  function save() {
    if (!S) return;
    try { localStorage.setItem(SAVE_KEY, Sim.serialize(S)); } catch (e) {}
  }

  function selectToolInternal(k) {
    tool = k;
    UI.showToolTip(k);
    UI.refreshPaletteState();
  }

  // ------------------------------------------------------------ 외부 API (ui.js가 사용)
  var API = {
    state: function () { return S; },
    tool: function () { return tool; },
    selectTool: selectToolInternal,
    setSpeed: function (v) { speed = v; },
    setOverlay: function (v) { overlay = v; },
    setTax: function (v) { S.tax = v; },
    save: save,
    reset: function () {
      S = Sim.create();
      Sim.computeFields(S);
      speed = 1; overlay = 'none'; selectToolInternal(null);
      Render.cam.zoom = 1;
      Render.centerOn(Math.floor(BAL.W * 0.62), Math.floor(BAL.H / 2));
      save();
      UI.closeModal();
      UI.updateHUD();
      UI.renderPalette();
      UI.toast('새 도시를 시작합니다', 'good');
    }
  };

  global.PocketCity = API;   // 디버깅/자동화용

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();

  if ('serviceWorker' in navigator) {
    global.addEventListener('load', function () {
      navigator.serviceWorker.register('sw.js').catch(function () {});
    });
  }
})(window);
