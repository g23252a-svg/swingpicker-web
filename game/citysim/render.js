/* 포켓시티 — 캔버스 렌더러 */
(function (global) {
  'use strict';

  var T = GD.T;
  // 기본 타일 크기(px). 폰 세로 화면(≈390px)에서 한 번에 20칸 남짓 보이는 값.
  var TS = 20;

  var cv, ctx, dpr = 1, cw = 0, ch = 0;
  var cam = { x: 0, y: 0, zoom: 1, min: 0.55, max: 3.4 };

  var GRASS = ['#2b4531', '#2f4a35', '#33513a', '#2d4833'];
  var ZONE_COL = {};
  ZONE_COL[T.R] = { edge: '#56c271', body: ['#3f7a55', '#4f9d68', '#63c07f'], roof: '#8fe0a8' };
  ZONE_COL[T.C] = { edge: '#4da3ff', body: ['#2f5f96', '#3c7cc4', '#4f9ae8'], roof: '#93cbff' };
  ZONE_COL[T.I] = { edge: '#e8b23a', body: ['#8a6a2a', '#b08a35', '#d0a542'], roof: '#f2d287' };

  function init(canvas) {
    cv = canvas;
    ctx = cv.getContext('2d', { alpha: false });
    resize();
  }

  function resize() {
    dpr = Math.min(global.devicePixelRatio || 1, 2.5);
    cw = cv.clientWidth; ch = cv.clientHeight;
    cv.width = Math.round(cw * dpr);
    cv.height = Math.round(ch * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
  }

  function centerOn(tx, ty) {
    cam.x = tx * TS;
    cam.y = ty * TS;
  }

  function clampCam(S) {
    var pad = 200 / cam.zoom;
    cam.x = Math.max(-pad, Math.min(S.w * TS + pad, cam.x));
    cam.y = Math.max(-pad, Math.min(S.h * TS + pad, cam.y));
    cam.zoom = Math.max(cam.min, Math.min(cam.max, cam.zoom));
  }

  // 화면 좌표 -> 타일 좌표
  function screenToTile(sx, sy) {
    var wx = (sx - cw / 2) / cam.zoom + cam.x;
    var wy = (sy - ch / 2) / cam.zoom + cam.y;
    return { x: Math.floor(wx / TS), y: Math.floor(wy / TS) };
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ------------------------------------------------------------ 메인 그리기
  function draw(S, view) {
    if (!ctx) return;
    var z = cam.zoom, size = TS * z;

    ctx.fillStyle = '#131a17';
    ctx.fillRect(0, 0, cw, ch);

    var left = cam.x - cw / 2 / z, top = cam.y - ch / 2 / z;
    var x0 = Math.max(0, Math.floor(left / TS));
    var y0 = Math.max(0, Math.floor(top / TS));
    var x1 = Math.min(S.w - 1, Math.ceil((left + cw / z) / TS));
    var y1 = Math.min(S.h - 1, Math.ceil((top + ch / z) / TS));

    var ox = cw / 2 - cam.x * z, oy = ch / 2 - cam.y * z;
    var detail = size > 13;   // 너무 축소하면 디테일 생략

    var x, y, k, c, px, py;

    // 1) 바닥
    for (y = y0; y <= y1; y++) {
      for (x = x0; x <= x1; x++) {
        k = y * S.w + x;
        c = S.tile[k];
        px = ox + x * size; py = oy + y * size;
        if (c === T.WATER) {
          ctx.fillStyle = '#1d3c5e';
          ctx.fillRect(px, py, size + 1, size + 1);
        } else {
          ctx.fillStyle = GRASS[(x * 7 + y * 13) & 3];
          ctx.fillRect(px, py, size + 1, size + 1);
        }
      }
    }

    // 2) 타일 내용
    ctx.font = Math.round(size * 0.6) + 'px system-ui, "Apple Color Emoji", "Segoe UI Emoji"';
    for (y = y0; y <= y1; y++) {
      for (x = x0; x <= x1; x++) {
        k = y * S.w + x;
        c = S.tile[k];
        if (c === T.EMPTY || c === T.WATER) continue;
        px = ox + x * size; py = oy + y * size;

        if (c === T.TREE) { drawTree(px, py, size, x, y); continue; }
        if (c === T.ROAD) { drawRoad(S, x, y, px, py, size, detail); continue; }
        if (c === T.RUBBLE) { drawRubble(px, py, size); continue; }
        if (Sim.isZone(c)) { drawZone(S, k, c, px, py, size, detail, x, y); continue; }
        drawFacility(S, k, c, px, py, size, x, y, detail);
      }
    }

    // 3) 오버레이 히트맵
    if (view.overlay && view.overlay !== 'none') {
      drawOverlay(S, view.overlay, x0, y0, x1, y1, ox, oy, size);
    }

    // 4) 커서 / 드래그 프리뷰
    if (view.cursor) drawCursor(S, view, ox, oy, size);

    // 5) 맵 경계
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.lineWidth = 2;
    ctx.strokeRect(ox, oy, S.w * size, S.h * size);
  }

  function drawTree(px, py, s, x, y) {
    var cx = px + s / 2, cy = py + s / 2;
    ctx.fillStyle = '#1f3a26';
    ctx.beginPath(); ctx.arc(cx, cy + s * 0.10, s * 0.30, 0, 6.284); ctx.fill();
    ctx.fillStyle = ((x + y) & 1) ? '#3d7a48' : '#356b40';
    ctx.beginPath(); ctx.arc(cx, cy - s * 0.04, s * 0.28, 0, 6.284); ctx.fill();
  }

  function drawRubble(px, py, s) {
    ctx.fillStyle = '#4a4136';
    ctx.fillRect(px + s * 0.14, py + s * 0.14, s * 0.72, s * 0.72);
    ctx.fillStyle = '#6b5c49';
    ctx.fillRect(px + s * 0.26, py + s * 0.30, s * 0.20, s * 0.18);
    ctx.fillRect(px + s * 0.54, py + s * 0.52, s * 0.22, s * 0.16);
  }

  function drawRoad(S, x, y, px, py, s, detail) {
    ctx.fillStyle = '#33373d';
    ctx.fillRect(px, py, s + 1, s + 1);
    if (!detail) return;
    var w = S.w, h = S.h;
    var n = y > 0 && S.tile[(y - 1) * w + x] === T.ROAD;
    var so = y < h - 1 && S.tile[(y + 1) * w + x] === T.ROAD;
    var we = x > 0 && S.tile[y * w + x - 1] === T.ROAD;
    var ea = x < w - 1 && S.tile[y * w + x + 1] === T.ROAD;
    var cx = px + s / 2, cy = py + s / 2, lw = Math.max(1, s * 0.045);

    ctx.strokeStyle = 'rgba(232,222,180,.55)';
    ctx.lineWidth = lw;
    ctx.setLineDash([s * 0.16, s * 0.14]);
    ctx.beginPath();
    if (n) { ctx.moveTo(cx, py); ctx.lineTo(cx, cy); }
    if (so) { ctx.moveTo(cx, cy); ctx.lineTo(cx, py + s); }
    if (we) { ctx.moveTo(px, cy); ctx.lineTo(cx, cy); }
    if (ea) { ctx.moveTo(cx, cy); ctx.lineTo(px + s, cy); }
    if (!n && !so && !we && !ea) { ctx.moveTo(px + s * 0.3, cy); ctx.lineTo(px + s * 0.7, cy); }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawZone(S, k, c, px, py, s, detail, x, y) {
    var lv = S.level[k], col = ZONE_COL[c];

    if (lv === 0) {
      // 빈 구역: 색 점선 + 옅은 채움
      ctx.fillStyle = c === T.R ? 'rgba(86,194,113,.16)' : c === T.C ? 'rgba(77,163,255,.16)' : 'rgba(232,178,58,.16)';
      ctx.fillRect(px + 1, py + 1, s - 2, s - 2);
      if (detail) {
        ctx.strokeStyle = col.edge;
        ctx.lineWidth = Math.max(1, s * 0.04);
        ctx.setLineDash([s * 0.13, s * 0.11]);
        ctx.strokeRect(px + s * 0.12, py + s * 0.12, s * 0.76, s * 0.76);
        ctx.setLineDash([]);
      }
      return;
    }

    // 건물: 레벨이 높을수록 크고 높다
    var inset = s * (lv === 1 ? 0.22 : lv === 2 ? 0.15 : 0.10);
    var bw = s - inset * 2;
    var lift = s * (lv === 1 ? 0.05 : lv === 2 ? 0.10 : 0.17);   // 위로 솟은 느낌

    ctx.fillStyle = 'rgba(0,0,0,.35)';
    roundRect(px + inset + s * 0.04, py + inset + s * 0.06, bw, bw, s * 0.09); ctx.fill();

    ctx.fillStyle = col.body[lv - 1];
    roundRect(px + inset, py + inset - lift, bw, bw, s * 0.09); ctx.fill();

    if (!detail) return;

    // 옥상 하이라이트
    ctx.fillStyle = col.roof;
    ctx.globalAlpha = 0.35;
    roundRect(px + inset + bw * 0.12, py + inset - lift + bw * 0.08, bw * 0.76, bw * 0.20, s * 0.05);
    ctx.fill();
    ctx.globalAlpha = 1;

    // 창문 (확대했을 때만)
    if (s > 26) {
      var n = lv + 1, gap = bw / (n + 1), r = Math.max(0.8, s * 0.045);
      ctx.fillStyle = 'rgba(255,246,205,.82)';
      for (var i = 1; i <= n; i++) {
        for (var j = 1; j <= n; j++) {
          if (((x * 3 + y * 5 + i * 7 + j * 11) % 5) === 0) continue;
          ctx.beginPath();
          ctx.arc(px + inset + gap * i, py + inset - lift + bw * 0.42 + gap * (j - 1) * 0.7, r, 0, 6.284);
          ctx.fill();
        }
      }
    }
  }

  function drawFacility(S, k, c, px, py, s, x, y, detail) {
    var item = Sim.byCode[c];
    if (!item) return;
    var working = Sim.hasRoadAdj(S, x, y);

    ctx.fillStyle = 'rgba(0,0,0,.35)';
    roundRect(px + s * 0.12, py + s * 0.16, s * 0.78, s * 0.78, s * 0.14); ctx.fill();

    ctx.fillStyle = working ? '#dfe6ef' : '#7c7f86';
    roundRect(px + s * 0.09, py + s * 0.09, s * 0.82, s * 0.82, s * 0.14); ctx.fill();

    if (detail) {
      ctx.fillText(item.icon, px + s / 2, py + s / 2 + s * 0.03);
      if (!working) {
        ctx.fillStyle = 'rgba(255,107,107,.92)';
        ctx.beginPath(); ctx.arc(px + s * 0.83, py + s * 0.17, s * 0.13, 0, 6.284); ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = Math.round(s * 0.20) + 'px system-ui';
        ctx.fillText('!', px + s * 0.83, py + s * 0.18);
        ctx.font = Math.round(s * 0.6) + 'px system-ui, "Apple Color Emoji", "Segoe UI Emoji"';
      }
    }
  }

  function drawOverlay(S, mode, x0, y0, x1, y1, ox, oy, size) {
    var f = S.fields;
    for (var y = y0; y <= y1; y++) {
      for (var x = x0; x <= x1; x++) {
        var k = y * S.w + x, col = null;
        if (mode === 'power') col = f.power[k] ? 'rgba(255,214,64,.34)' : 'rgba(10,10,20,.48)';
        else if (mode === 'water') col = f.water[k] ? 'rgba(64,190,255,.32)' : 'rgba(10,10,20,.48)';
        else if (mode === 'pollution') {
          var p = Math.min(1, f.pollution[k] / 22);
          col = p > 0.02 ? 'rgba(180,60,40,' + (p * 0.62).toFixed(3) + ')' : null;
        } else if (mode === 'land') {
          var v = f.land[k] / 100;
          col = 'rgba(' + Math.round(255 - v * 190) + ',' + Math.round(70 + v * 165) + ',90,.40)';
        }
        if (col) {
          ctx.fillStyle = col;
          ctx.fillRect(ox + x * size, oy + y * size, size + 1, size + 1);
        }
      }
    }
  }

  function drawCursor(S, view, ox, oy, size) {
    var cells = view.cursor;
    ctx.lineWidth = Math.max(1.5, size * 0.07);
    for (var i = 0; i < cells.length; i++) {
      var t = cells[i];
      if (t.x < 0 || t.y < 0 || t.x >= S.w || t.y >= S.h) continue;
      ctx.strokeStyle = t.ok ? 'rgba(255,255,255,.92)' : 'rgba(255,107,107,.95)';
      ctx.fillStyle = t.ok ? 'rgba(255,255,255,.16)' : 'rgba(255,107,107,.20)';
      var px = ox + t.x * size, py = oy + t.y * size;
      ctx.fillRect(px, py, size, size);
      ctx.strokeRect(px + 1, py + 1, size - 2, size - 2);
    }
  }

  global.Render = {
    init: init, resize: resize, draw: draw, cam: cam,
    screenToTile: screenToTile, centerOn: centerOn, clampCam: clampCam, TS: TS,
    get w() { return cw; }, get h() { return ch; }
  };
})(window);
