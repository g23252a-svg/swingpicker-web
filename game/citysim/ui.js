/* 포켓시티 — HUD / 팔레트 / 모달 / 토스트 */
(function (global) {
  'use strict';

  var CATALOG = GD.CATALOG, TOOLS = GD.TOOLS, TIERS = GD.TIERS;
  var G = null;   // game.js가 주입

  var el = {};
  function $(id) { return document.getElementById(id); }

  function init(game) {
    G = game;
    el.money = $('stat-money').querySelector('b');
    el.moneyBox = $('stat-money');
    el.pop = $('stat-pop').querySelector('b');
    el.happy = $('stat-happy').querySelector('b');
    el.date = $('date');
    el.tier = $('tier');
    el.power = $('meter-power');
    el.water = $('meter-water');
    el.tools = $('tools');
    el.toolTip = $('tool-tip');
    el.toolTipText = $('tool-tip-text');
    el.toasts = $('toasts');
    el.modal = $('modal');
    el.modalTitle = $('modal-title');
    el.modalBody = $('modal-body');
    el.rci = document.querySelectorAll('.rci-bar');

    // 카테고리 탭
    $('cats').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      $('cats').querySelectorAll('button').forEach(function (x) { x.classList.toggle('on', x === b); });
      renderPalette(b.dataset.cat);
    });

    // 속도
    $('speed').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      G.setSpeed(parseInt(b.dataset.speed, 10));
      $('speed').querySelectorAll('button').forEach(function (x) { x.classList.toggle('on', x === b); });
    });

    // 오버레이
    $('overlay-switch').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      G.setOverlay(b.dataset.ov);
      $('overlay-switch').querySelectorAll('button').forEach(function (x) { x.classList.toggle('on', x === b); });
    });

    $('btn-menu').addEventListener('click', openMenu);
    $('tool-cancel').addEventListener('click', function () { G.selectTool(null); });
    $('modal-close').addEventListener('click', closeModal);
    el.modal.addEventListener('click', function (e) { if (e.target === el.modal) closeModal(); });

    renderPalette('zone');
  }

  // ------------------------------------------------------------ 포맷
  function won(v) {
    var neg = v < 0; v = Math.abs(Math.round(v));
    var s;
    if (v >= 100000000) s = (v / 100000000).toFixed(2).replace(/\.?0+$/, '') + '억';
    else if (v >= 10000) s = (v / 10000).toFixed(1).replace(/\.0$/, '') + '만';
    else s = v.toLocaleString('ko-KR');
    return (neg ? '-₩' : '₩') + s;
  }
  function num(v) { return Math.round(v).toLocaleString('ko-KR'); }

  // ------------------------------------------------------------ 팔레트
  var curCat = 'zone';
  function renderPalette(cat) {
    curCat = cat || curCat;
    var S = G.state();
    var items = [];
    Object.keys(CATALOG).forEach(function (k) {
      if (CATALOG[k].cat === curCat) items.push({ key: k, it: CATALOG[k] });
    });
    if (curCat === 'tool') {
      Object.keys(TOOLS).forEach(function (k) { items.push({ key: k, it: TOOLS[k] }); });
    }

    el.tools.innerHTML = '';
    items.forEach(function (o) {
      var locked = o.it.unlockPop && S.pop < o.it.unlockPop;
      var b = document.createElement('button');
      b.className = 'tool' + (G.tool() === o.key ? ' on' : '') + (locked ? ' locked' : '') +
        (!locked && o.it.cost > S.money ? ' poor' : '');
      b.dataset.key = o.key;
      b.innerHTML =
        '<span class="ico">' + o.it.icon + '</span>' +
        '<span class="nm">' + o.it.name + '</span>' +
        '<span class="cost">' + (o.it.cost ? won(o.it.cost) : '—') + '</span>' +
        (locked ? '<span class="lock">🔒</span>' : '');
      b.addEventListener('click', function () {
        if (locked) {
          toast(o.it.name + '은(는) 인구 ' + num(o.it.unlockPop) + '명부터 해금됩니다', 'warn');
          return;
        }
        G.selectTool(G.tool() === o.key ? null : o.key);
      });
      el.tools.appendChild(b);
    });
  }

  function refreshPaletteState() {
    var S = G.state();
    el.tools.querySelectorAll('.tool').forEach(function (b) {
      var k = b.dataset.key, it = CATALOG[k] || TOOLS[k];
      if (!it) return;
      var locked = it.unlockPop && S.pop < it.unlockPop;
      b.classList.toggle('on', G.tool() === k);
      b.classList.toggle('locked', !!locked);
      b.classList.toggle('poor', !locked && it.cost > S.money);
      var lock = b.querySelector('.lock');
      if (locked && !lock) {
        var sp = document.createElement('span'); sp.className = 'lock'; sp.textContent = '🔒'; b.appendChild(sp);
      } else if (!locked && lock) lock.remove();
    });
  }

  function showToolTip(key) {
    if (!key) { el.toolTip.hidden = true; return; }
    var it = CATALOG[key] || TOOLS[key];
    el.toolTipText.textContent = it.icon + ' ' + it.name + ' · ' + it.desc;
    el.toolTip.hidden = false;
  }

  // ------------------------------------------------------------ HUD
  function updateHUD() {
    var S = G.state();
    el.money.textContent = won(S.money);
    el.moneyBox.classList.toggle('neg', S.money < 0);
    el.pop.textContent = num(S.pop) + '명';
    el.happy.textContent = Math.round(S.happy) + '%';
    el.date.textContent = (Math.floor(S.month / 12) + 1) + '년 ' + (S.month % 12 + 1) + '월';
    el.tier.textContent = TIERS[S.tierIdx].name + (S.event ? ' · ' + S.event.name : '');

    setMeter(el.power, S.powerCap, S.powerUse);
    setMeter(el.water, S.waterCap, S.waterUse);

    el.rci.forEach(function (bar) {
      var d = S.demand[bar.dataset.z] || 0;
      var i = bar.querySelector('i');
      var pct = Math.min(50, Math.abs(d) * 50);
      i.style.width = pct + '%';
      i.style.transform = d < 0 ? 'translateX(-100%)' : 'translateX(0)';
    });
  }

  function setMeter(m, cap, use) {
    var ratio = use > 0 ? cap / use : (cap > 0 ? 1 : 0);
    m.classList.toggle('short', use > 0 && cap < use);
    m.querySelector('.bar > span').style.width = Math.min(100, ratio * 100) + '%';
  }

  // ------------------------------------------------------------ 토스트
  var lastToast = '', lastToastAt = 0;
  function toast(msg, type) {
    var now = Date.now();
    if (msg === lastToast && now - lastToastAt < 1500) return;
    lastToast = msg; lastToastAt = now;
    var d = document.createElement('div');
    d.className = 'toast ' + (type || '');
    d.textContent = msg;
    el.toasts.appendChild(d);
    setTimeout(function () { d.remove(); }, 2800);
    if (el.toasts.children.length > 3) el.toasts.firstChild.remove();
  }

  // ------------------------------------------------------------ 모달
  function openModal(title, html) {
    el.modalTitle.textContent = title;
    el.modalBody.innerHTML = html;
    el.modal.hidden = false;
    return el.modalBody;
  }
  function closeModal() { el.modal.hidden = true; el.modalBody.innerHTML = ''; }

  function openMenu() {
    var b = openModal('메뉴', '');
    [['📊 도시 현황', openStats], ['💰 예산·세율', openBudget], ['📖 플레이 방법', openHelp],
     ['💾 지금 저장', function () { G.save(); toast('저장했습니다', 'good'); }],
     ['🗑 새 도시 시작', confirmReset]].forEach(function (o) {
      var btn = document.createElement('button');
      btn.className = 'btn' + (o[0][0] === '🗑' ? ' danger' : '');
      btn.textContent = o[0];
      btn.onclick = o[1];
      b.appendChild(btn);
    });
  }

  function row(k, v, cls) {
    return '<div class="row"><span>' + k + '</span><span class="v ' + (cls || '') + '">' + v + '</span></div>';
  }

  function openStats() {
    var S = G.state();
    var jobs = S.jobsC + S.jobsI;
    var net = S.income - S.expense;
    var h = '';
    h += '<h3 class="sec">인구</h3>';
    h += row('총 인구', num(S.pop) + '명');
    h += row('경제활동 인구', num(S.pop * GD.BAL.WORK_RATIO) + '명');
    h += row('일자리 (상업/공업)', num(S.jobsC) + ' / ' + num(S.jobsI));
    h += row('실업률', (S.unemployment * 100).toFixed(1) + '%', S.unemployment > 0.15 ? 'neg' : 'pos');
    h += '<h3 class="sec">삶의 질</h3>';
    h += row('행복도', Math.round(S.happy) + '%', S.happy >= 60 ? 'pos' : S.happy < 40 ? 'neg' : '');
    h += row('평균 지가', Math.round(S.avgLand) + ' / 100');
    h += row('평균 오염', Math.round(S.avgPollution), S.avgPollution > 8 ? 'neg' : '');
    h += row('범죄 지수', Math.round(S.crime || 0), (S.crime || 0) > 14 ? 'neg' : '');
    h += '<h3 class="sec">기반 시설</h3>';
    h += row('전력', num(S.powerUse) + ' / ' + num(S.powerCap), S.powerUse > S.powerCap ? 'neg' : 'pos');
    h += row('수도', num(S.waterUse) + ' / ' + num(S.waterCap), S.waterUse > S.waterCap ? 'neg' : 'pos');
    h += '<h3 class="sec">이번 달 재정</h3>';
    h += row('세수', won(S.income), 'pos');
    h += row('유지비', '-' + won(S.expense), 'neg');
    h += row('수지', (net >= 0 ? '+' : '') + won(net), net >= 0 ? 'pos' : 'neg');
    h += sparkline(S.hist, 'pop', '인구 추이');
    openModal('도시 현황', h);
  }

  function sparkline(hist, key, label) {
    if (!hist || hist.length < 3) return '';
    var vals = hist.map(function (p) { return p[key]; });
    var max = Math.max.apply(null, vals) || 1, min = Math.min.apply(null, vals);
    var span = Math.max(1, max - min);
    var pts = vals.map(function (v, i) {
      return (i / (vals.length - 1) * 100).toFixed(2) + ',' + (36 - (v - min) / span * 32).toFixed(2);
    }).join(' ');
    return '<h3 class="sec">' + label + '</h3>' +
      '<svg viewBox="0 0 100 40" preserveAspectRatio="none" style="width:100%;height:70px;overflow:visible">' +
      '<polyline points="' + pts + '" fill="none" stroke="#4da3ff" stroke-width="1.4" vector-effect="non-scaling-stroke"/>' +
      '</svg>' +
      '<p class="hint">최고 ' + num(max) + ' · 현재 ' + num(vals[vals.length - 1]) + '</p>';
  }

  function openBudget() {
    var S = G.state();
    var b = openModal('예산 · 세율', '');
    var wrap = document.createElement('div');
    wrap.innerHTML =
      '<div class="slider-head"><span>세율</span><b id="tax-val">' + S.tax.toFixed(1) + '%</b></div>' +
      '<input type="range" id="tax" min="0" max="20" step="0.5" value="' + S.tax + '">' +
      '<p class="hint" id="tax-hint"></p>';
    b.appendChild(wrap);

    var info = document.createElement('div');
    b.appendChild(info);

    function refresh() {
      var s = G.state();
      info.innerHTML =
        '<h3 class="sec">이번 달</h3>' +
        row('세수', won(s.income), 'pos') +
        row('유지비', '-' + won(s.expense), 'neg') +
        row('수지', (s.income - s.expense >= 0 ? '+' : '') + won(s.income - s.expense), s.income >= s.expense ? 'pos' : 'neg') +
        row('보유 자금', won(s.money), s.money < 0 ? 'neg' : '');
      var t = s.tax;
      document.getElementById('tax-hint').innerHTML =
        t <= 6 ? '<span class="ok">낮은 세율</span> — 시민이 좋아하고 도시가 빨리 자랍니다. 대신 적자를 조심하세요.'
        : t <= 11 ? '<span class="mid">보통 세율</span> — 성장과 재정의 균형점입니다.'
        : '<span class="no">높은 세율</span> — 세수는 늘지만 행복도가 떨어져 도시가 쇠퇴합니다.';
    }
    wrap.querySelector('#tax').addEventListener('input', function (e) {
      G.setTax(parseFloat(e.target.value));
      document.getElementById('tax-val').textContent = parseFloat(e.target.value).toFixed(1) + '%';
      refresh();
    });
    refresh();
  }

  function openHelp() {
    openModal('플레이 방법',
      '<h3 class="sec">조작</h3>' +
      '<p class="hint">· 한 손가락 드래그 — 지도 이동 (도구를 고른 상태면 <b>연속 건설</b>)<br>' +
      '· 두 손가락 — 확대·축소 및 이동<br>' +
      '· 도구를 고르고 칸을 탭하면 건설, <b>도구 → 철거</b>로 제거<br>' +
      '· 우측 <b>전력·수도·오염·지가</b> 버튼으로 상태를 지도 위에 겹쳐볼 수 있습니다.</p>' +
      '<h3 class="sec">도시가 자라는 조건</h3>' +
      '<p class="hint">구역 한 칸이 건물로 자라려면 <b>① 도로에 접해 있고 ② 전기가 들어오고 ③ 물이 나와야</b> 합니다. ' +
      '셋 중 하나라도 빠지면 몇 달 뒤 쇠퇴합니다.<br>' +
      '더 높은 레벨로 크려면 그 칸의 <b>지가</b>가 충분해야 합니다. 지가는 공원·광장·치안·교육·의료가 올리고, 오염과 범죄가 내립니다.</p>' +
      '<h3 class="sec">RCI 수요 막대</h3>' +
      '<p class="hint">상단의 <b>주거·상업·공업</b> 막대는 지금 무엇이 부족한지 알려줍니다. ' +
      '막대가 오른쪽으로 차 있으면 그 구역을 더 지정할 때이고, 왼쪽으로 차 있으면 이미 과잉이라 지어도 비어 있습니다.<br>' +
      '일자리가 부족하면 주거 수요가 멈추고, 인구가 부족하면 상업 수요가 생기지 않습니다.</p>' +
      '<h3 class="sec">막힐 때 확인할 것</h3>' +
      '<p class="hint">· 인구가 안 늘어난다 → 전력·수도 오버레이로 사각지대 확인, 일자리(상업·공업) 부족 확인<br>' +
      '· 건물이 무너진다 → 단전·단수 또는 수요 붕괴<br>' +
      '· 화재가 난다 → 소방서 반경 밖. 소방서를 늘리세요<br>' +
      '· 적자다 → 세율을 올리거나, 도로·시설을 줄이세요. 도로도 유지비를 먹습니다</p>' +
      '<h3 class="sec">추천 시작 순서</h3>' +
      '<p class="hint">화력발전 1 + 급수탑 1 → 도로를 격자로 깔고 → 주거 여러 칸 + 공업 몇 칸 → 인구가 붙으면 상업 → 공원 → 소방서 → 학교·병원 순으로.</p>');
  }

  function confirmReset() {
    var b = openModal('새 도시 시작', '<p class="hint">지금 도시는 사라지고 처음부터 다시 시작합니다. 되돌릴 수 없습니다.</p>');
    var yes = document.createElement('button');
    yes.className = 'btn danger'; yes.textContent = '정말 새로 시작';
    yes.onclick = function () { closeModal(); G.reset(); };
    var no = document.createElement('button');
    no.className = 'btn'; no.textContent = '취소';
    no.onclick = closeModal;
    b.appendChild(yes); b.appendChild(no);
  }

  function showInspect(info) {
    if (!info) return;
    var h = '';
    h += row('종류', info.name + (info.level ? ' · Lv.' + info.level : ''));
    if (info.capacity) h += row(info.code === GD.T.R ? '거주 인구' : '일자리', num(info.capacity) + (info.code === GD.T.R ? '명' : '개'));
    h += '<h3 class="sec">공급</h3>';
    h += row('도로 연결', info.road ? '연결됨' : '없음', info.road ? 'pos' : 'neg');
    h += row('전기', info.power ? '들어옴' : '안 들어옴', info.power ? 'pos' : 'neg');
    h += row('수도', info.water ? '나옴' : '안 나옴', info.water ? 'pos' : 'neg');
    h += '<h3 class="sec">환경</h3>';
    h += row('지가', info.land + ' / 100');
    h += row('오염', info.pollution, info.pollution > 8 ? 'neg' : '');
    h += row('치안 / 소방', info.police + ' / ' + info.fire);
    h += row('교육 / 의료', info.edu + ' / ' + info.health);
    h += '<p class="hint">좌표 (' + info.x + ', ' + info.y + ')</p>';
    openModal('칸 살펴보기', h);
  }

  global.UI = {
    init: init, updateHUD: updateHUD, toast: toast, showToolTip: showToolTip,
    renderPalette: renderPalette, refreshPaletteState: refreshPaletteState,
    openModal: openModal, closeModal: closeModal, showInspect: showInspect,
    openHelp: openHelp, won: won, num: num
  };
})(window);
