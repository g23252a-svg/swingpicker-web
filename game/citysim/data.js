/* 포켓시티 — 정적 데이터 (타일 종류, 건물 카탈로그, 밸런스 상수) */
(function (global) {
  'use strict';

  // 타일 코드
  var T = {
    EMPTY: 0,
    WATER: 1,   // 강 — 건설 불가, 지가 보너스
    TREE: 2,    // 나무 — 철거하면 건설 가능, 지가 보너스
    ROAD: 3,
    R: 4,       // 주거 구역
    C: 5,       // 상업 구역
    I: 6,       // 공업 구역
    RUBBLE: 7,  // 화재 잔해
    POWER_COAL: 10,
    POWER_SOLAR: 11,
    WATER_TOWER: 12,
    PARK: 13,
    PLAZA: 14,
    POLICE: 15,
    FIRE: 16,
    HOSPITAL: 17,
    SCHOOL: 18
  };

  var ZONES = [T.R, T.C, T.I];

  // 건설 카탈로그. cost=건설비, upkeep=월 유지비
  var CATALOG = {
    road:        { code: T.ROAD,        cat: 'road',    name: '도로',     icon: '🛣', cost: 10,   upkeep: 0.4, desc: '모든 건물은 도로에 닿아야 작동합니다.' },

    zone_r:      { code: T.R,           cat: 'zone',    name: '주거',     icon: '🏘', cost: 20,   upkeep: 0,   desc: '시민이 들어와 삽니다. 인구의 원천.' },
    zone_c:      { code: T.C,           cat: 'zone',    name: '상업',     icon: '🏬', cost: 25,   upkeep: 0,   desc: '일자리와 세수를 만듭니다. 인구가 있어야 수요가 생깁니다.' },
    zone_i:      { code: T.I,           cat: 'zone',    name: '공업',     icon: '🏭', cost: 25,   upkeep: 0,   desc: '일자리가 많지만 오염이 심합니다. 주거와 떨어뜨리세요.' },

    power_coal:  { code: T.POWER_COAL,  cat: 'util',    name: '화력발전', icon: '🔌', cost: 900,  upkeep: 45, power: 900,  pollution: 6, radius: 14, desc: '전력 900. 주변 오염이 큽니다.' },
    power_solar: { code: T.POWER_SOLAR, cat: 'util',    name: '태양광',   icon: '☀️', cost: 2400, upkeep: 35, power: 600,  pollution: 0, radius: 14, unlockPop: 3000, desc: '전력 600, 무공해. 인구 3,000명부터.' },
    water_tower: { code: T.WATER_TOWER, cat: 'util',    name: '급수탑',   icon: '💧', cost: 320,  upkeep: 16, water: 700,  radius: 12, desc: '수도 700 공급.' },

    park:        { code: T.PARK,        cat: 'park',    name: '공원',     icon: '🌳', cost: 130,  upkeep: 8,  land: 22, radius: 6, desc: '주변 지가와 행복도를 올립니다.' },
    plaza:       { code: T.PLAZA,       cat: 'park',    name: '광장',     icon: '⛲️', cost: 600,  upkeep: 26, land: 40, radius: 10, unlockPop: 1500, desc: '넓은 범위의 지가를 크게 올립니다. 인구 1,500명부터.' },

    police:      { code: T.POLICE,      cat: 'service', name: '경찰서',   icon: '🚓', cost: 520,  upkeep: 32, service: 'police', radius: 11, desc: '범죄를 낮춰 지가를 지킵니다.' },
    fire:        { code: T.FIRE,        cat: 'service', name: '소방서',   icon: '🚒', cost: 520,  upkeep: 32, service: 'fire',   radius: 11, desc: '화재 발생을 막습니다. 없으면 건물이 불탑니다.' },
    school:      { code: T.SCHOOL,      cat: 'service', name: '학교',     icon: '🏫', cost: 760,  upkeep: 52, service: 'edu',    radius: 12, unlockPop: 400,  desc: '교육 수준이 상업 발전의 조건입니다. 인구 400명부터.' },
    hospital:    { code: T.HOSPITAL,    cat: 'service', name: '병원',     icon: '🏥', cost: 1000, upkeep: 64, service: 'health', radius: 13, unlockPop: 900,  desc: '건강과 행복도를 올립니다. 인구 900명부터.' }
  };

  // 도구(건설이 아닌 조작)
  var TOOLS = {
    bulldoze: { cat: 'tool', name: '철거', icon: '🚧', cost: 4, desc: '건물·구역·나무를 없앱니다. 칸당 ₩4.' },
    inspect:  { cat: 'tool', name: '살펴보기', icon: '🔍', cost: 0, desc: '칸을 눌러 상태를 확인합니다.' }
  };

  // 성장 단계
  var TIERS = [
    { pop: 0,     name: '마을' },
    { pop: 500,   name: '읍' },
    { pop: 2000,  name: '시' },
    { pop: 10000, name: '대도시' },
    { pop: 50000, name: '광역시' }
  ];

  var BAL = {
    W: 64, H: 64,               // 맵 크기(칸)
    START_MONEY: 12000,
    START_TAX: 9,               // %
    TICK_MS: 2600,              // 1개월 = 2.6초 (1배속)

    // 구역 1칸이 레벨별로 담는 용량
    R_CAP: [0, 14, 44, 110],    // 인구
    C_JOBS: [0, 6, 18, 44],     // 상업 일자리
    I_JOBS: [0, 10, 26, 60],    // 공업 일자리

    // 레벨업에 필요한 지가(0~100). 빈 땅의 기본 지가는 30이므로 Lv.1은 여유롭게,
    // Lv.2 이상은 공원·서비스 투자가 있어야 도달한다.
    LEVEL_REQ: [0, 14, 40, 66],

    // 1칸이 쓰는 전력/수도 (레벨 비례)
    POWER_PER: [0, 6, 16, 34],
    WATER_PER: [0, 5, 13, 28],

    WORK_RATIO: 0.52,           // 인구 중 경제활동 비율
    COMMUTE_JOBS: 45,           // 도시 밖으로 통근 가능한 일자리 (초기 도시 보정)
    C_JOB_DEMAND: 0.13,         // 인구 1명당 필요한 상업 일자리
    I_JOB_DEMAND: 0.17,

    TAX_R: 0.070,               // 세율 1%당 인구 1명 세수
    TAX_C: 0.115,               // 세율 1%당 상업 일자리 1개 세수
    TAX_I: 0.100,

    POLLUTION_I: 2.2,           // 공업 1칸 레벨당 오염 (여러 칸이 겹치면 누적된다)
    POLLUTION_RADIUS: 7
  };

  global.GD = { T: T, ZONES: ZONES, CATALOG: CATALOG, TOOLS: TOOLS, TIERS: TIERS, BAL: BAL };
})(window);
