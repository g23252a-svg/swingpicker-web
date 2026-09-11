/* 백귀야행 — 도트 스프라이트 자료
   '.'은 투명. 나머지 글자는 각 스프라이트의 palette 키. */
const SPR = {

/* ── 무사 (플레이어) 12x12 ─────────────────────── */
musa: { p:{ o:'#17131f', h:'#2b2140', s:'#f2c9a0', e:'#17131f',
            b:'#4a7fd4', l:'#7aa9ef', t:'#efe6cc', k:'#2f3b52' }, d:[
'...oooooo...',
'..ohhhhhho..',
'..ohhhhhho..',
'..osssssso..',
'..osesseso..',
'..osssssso..',
'.ottbbbbtto.',
'.obbllllbbo.',
'.obbbbbbbbo.',
'.ottbbbbtto.',
'..okk..kko..',
'...oo..oo...']},

/* ── 무녀 / 역사 (같은 뼈대, 다른 색) ─────────── */
munyeo: { p:{ o:'#1a1218', h:'#241a22', s:'#f7d6b4', e:'#1a1218',
              b:'#ece6ef', l:'#ffffff', t:'#c93a3a', k:'#7d2a2a' }, d:[
'...oooooo...',
'..ohhhhhho..',
'..ohhhhhho..',
'..osssssso..',
'..osesseso..',
'..osssssso..',
'.ottbbbbtto.',
'.obbllllbbo.',
'.obbbbbbbbo.',
'.ottbbbbtto.',
'..okk..kko..',
'...oo..oo...',
]},

yeoksa: { p:{ o:'#141a12', h:'#2a2016', s:'#d9a06a', e:'#141a12',
              b:'#4a6b3a', l:'#6f9455', t:'#c9a24a', k:'#3a2e1c' }, d:[
'...oooooo...',
'..ohhhhhho..',
'..ohhhhhho..',
'..osssssso..',
'..osesseso..',
'..osssssso..',
'.ottbbbbtto.',
'.obbllllbbo.',
'.obbbbbbbbo.',
'.ottbbbbtto.',
'..okk..kko..',
'...oo..oo...',
]},

/* ── 달걀귀신 (1막 보스) 16x16 ────────────────── */
dalgyal: { p:{ o:'#171a1c', w:'#f3f0e6', g:'#b9b3a4' }, d:[
'......oooo......',
'....oowwwwoo....',
'...owwwwwwwwo...',
'..owwwwwwwwwwo..',
'..owwwwwwwwwwo..',
'.owwwwwwwwwwwwo.',
'.owwwwwwwwwwwwo.',
'.owwwwwwwwwwwwo.',
'.owwwwwwwwwwwwo.',
'..owwwwwwwwwwo..',
'..owwwwwwwwwwo..',
'...owwwwwwwwo...',
'...ogwwwwwwgo...',
'....ogwwwwgo....',
'.....og..go.....',
'......o..o......']},

/* ── 혼불 12x12 ─────────────────────────────── */
honbul: { p:{ o:'#0b1a28', f:'#4fc9f5', g:'#c9f6ff', d:'#2477a8' }, d:[
'.....gg.....',
'....gffg....',
'...gffffg...',
'..offffffo..',
'..offggffo..',
'..offggffo..',
'..offffffo..',
'...odffdo...',
'....oddo....',
'.....oo.....',
'............',
'............']},

/* ── 허수아비 12x12 ─────────────────────────── */
heosu: { p:{ o:'#170f05', w:'#cda552', e:'#170f05', c:'#7d4a2a', s:'#a8702f' }, d:[
'....oooo....',
'...occcco...',
'.oooccccooo.',
'..owwwwwwo..',
'..owewwewo..',
'..owwwwwwo..',
'...osssso...',
'.ossccccsso.',
'.osccccccso.',
'..occcccco..',
'...oww.wo...',
'...oo..oo...']},

/* ── 도깨비 12x12 ───────────────────────────── */
dokkae: { p:{ o:'#22060a', r:'#d1402f', l:'#ee7a53', e:'#ffe14d', y:'#8d1f18',
              t:'#fff3c4', c:'#6b4a2a' }, d:[
'..o......o..',
'.oto....oto.',
'..orrrrrro..',
'..orlllrro..',
'..oererero..',
'..orttttro..',
'.oorrrrrroo.',
'orrrrrrrrrro',
'oyrrrrrrrryo',
'.oorrrrrroo.',
'...oyy.yyo..',
'...oo..oo...']},

/* ── 해골 12x12 ─────────────────────────────── */
haegol: { p:{ o:'#191919', b:'#e8e4d6', g:'#9d9788', e:'#141018', r:'#3a3630' }, d:[
'...oooooo...',
'..obbbbbbo..',
'.obbbbbbbbo.',
'.obeebbeebo.',
'.obeebbeebo.',
'.obbbbbbbbo.',
'..obbrrbbo..',
'..obobobbo..',
'...obbbbo...',
'..obbbbbbo..',
'...ogg.ggo..',
'...oo..oo...']},

/* ── 불여우 12x12 ───────────────────────────── */
yeou: { p:{ o:'#2a0c03', r:'#f07a1e', l:'#ffb85c', w:'#fff0d2', e:'#2a0c03',
            y:'#c14e0a' }, d:[
'.o........o.',
'oro......oro',
'orro....orro',
'.orrrrrrrro.',
'.orererrero.',
'.orrwwwwrro.',
'..orwwwwro..',
'.oorrwwrroo.',
'orrrrrrrrrro',
'oyrrrrrrryyo',
'.oyy.rr.yyo.',
'..o..oo..o..']},

/* ── 그슨대 12x12 (덩치) ────────────────────── */
geuseun: { p:{ o:'#0a0610', d:'#3a2555', m:'#5c3b84', e:'#ff5d5d',
               g:'#241338' }, d:[
'..oo....oo..',
'.oddo..oddo.',
'.odddoodddo.',
'oddddddddddo',
'odmdddddmddo',
'odedddddedo.',
'oddddddddddo',
'odddmmmmdddo',
'oddddddddddo',
'.oggddddggo.',
'..ogg..ggo..',
'...o....o...']},

/* ── 구미호 (중간보스) 16x16 ─────────────────── */
gumiho: { p:{ o:'#2b1408', w:'#fdf6e6', g:'#c9573c', y:'#f0b23c',
              e:'#d43a3a' }, d:[
'.o............o.',
'.oyo........oyo.',
'.oywo......oywo.',
'..owwoooooowwo..',
'..owwwwwwwwwwo..',
'..owweewweewwo..',
'..owwwwwwwwwwo..',
'..owwwggggwwwo..',
'...owwwwwwwwo...',
'.o.owwwwwwwwo.o.',
'oyo.owwwwwwo.oyo',
'oyyo.owwwwo.oyyo',
'.oyyo.oooo.oyyo.',
'..oyyo....oyyo..',
'...oyyo..oyyo...',
'....oyo..oyo....']},

/* ── 저승사자 (최종보스) 16x16 ───────────────── */
saja: { p:{ o:'#08060c', k:'#1d1830', m:'#332a52', e:'#ff3b30',
            g:'#7b2233', b:'#c8c2d8' }, d:[
'......oooo......',
'....ookkkkoo....',
'..okkkkkkkkkko..',
'..okkbbbbbbkko..',
'..okbebbbbebko..',
'..okkbbbbbbkko..',
'..okkkkkkkkkko..',
'.okkkmmmmmmkkko.',
'okkkkmmmmmmkkkko',
'okgkkkmmmmkkkgko',
'okgkkkkkkkkkkgko',
'.okkkkkkkkkkkko.',
'..okkkkkkkkkko..',
'...okkkkkkkko...',
'....okk..kko....',
'.....oo..oo.....']},

/* ── 습득물 8x8 ─────────────────────────────── */
gem1: { p:{ o:'#0d2a14', a:'#4be07a', b:'#a9ffc8' }, d:[
'..oooo..',
'.obbbbo.',
'obbaabbo',
'obaaaabo',
'obaaaabo',
'obbaabbo',
'.obbbbo.',
'..oooo..']},

gem2: { p:{ o:'#0b1c33', a:'#3aa0ff', b:'#b6dcff' }, d:[
'..oooo..',
'.obbbbo.',
'obbaabbo',
'obaaaabo',
'obaaaabo',
'obbaabbo',
'.obbbbo.',
'..oooo..']},

gem3: { p:{ o:'#2a0b33', a:'#c060ff', b:'#efc9ff' }, d:[
'..oooo..',
'.obbbbo.',
'obbaabbo',
'obaaaabo',
'obaaaabo',
'obbaabbo',
'.obbbbo.',
'..oooo..']},

heart: { p:{ o:'#3d0810', r:'#ff4d5e', l:'#ffa8b0' }, d:[
'.oo..oo.',
'olroorlo',
'orrrrrro',
'orrrrrro',
'.orrrro.',
'..orro..',
'...oo...',
'........']},

magnet: { p:{ o:'#231a05', r:'#e04a3a', s:'#d8d8e0' }, d:[
'.oo..oo.',
'orroorro',
'orroorro',
'orroorro',
'orrrrrro',
'.orrrro.',
'..osso..',
'...oo...']},

chest: { p:{ o:'#231405', g:'#f0b23c', y:'#ffe9a8', b:'#8a5a24', w:'#e8e2cf' }, d:[
'.oooooo.',
'ogyyyygo',
'oggggggo',
'oooooooo',
'obbbbbbo',
'obbowobo',
'obbbbbbo',
'.oooooo.']},

soul: { p:{ o:'#0d1c33', f:'#8fd0ff', w:'#f2fbff' }, d:[
'...oo...',
'..offo..',
'.offffo.',
'offwwffo',
'offwwffo',
'.offffo.',
'..offo..',
'...oo...']},

/* ── 적 탄환 8x8 (오른쪽이 진행 방향) ──────────
   둥근 구슬처럼 그리면 주우러 가게 된다. 꼬리 달린 불덩이로 그린다. */
bulssi: { p:{ o:'#3a0c02', r:'#c9340f', y:'#ff9a2e', w:'#fff0c4' }, d:[
'.....oo.',
'...oryyo',
'..orywwo',
'oorywwwo',
'oorywwwo',
'..orywwo',
'...oryyo',
'.....oo.']},

sajabolt: { p:{ o:'#050308', m:'#4b2f6e', k:'#2a1840', e:'#ff3b30' }, d:[
'....oo..',
'..oommo.',
'.omkkmmo',
'omkkeemo',
'omkkeemo',
'.omkkmmo',
'..oommo.',
'....oo..']},

/* ── 무기 8x8 ───────────────────────────────── */
bujeok: { p:{ o:'#3a2a0a', y:'#ffe9a8', r:'#d13a2a' }, d:[
'.oooooo.',
'oyyyyyyo',
'oyryryyo',
'oyyryyyo',
'oyryryyo',
'oyyyyyyo',
'oyyryyyo',
'.oooooo.']},

hwasal: { p:{ o:'#2b1a0a', w:'#e8e2cf', t:'#b6873f' }, d:[
'.......o',
'......ow',
'.....oww',
'oooooww.',
'ottttwo.',
'oooowo..',
'...ow...',
'..o.....']},

bell: { p:{ o:'#3a2c05', g:'#f5c542', l:'#ffeaa0', d:'#a67c14' }, d:[
'...oo...',
'..ollo..',
'.ogllgo.',
'.oggggo.',
'oggggggo',
'oddddddo',
'.o.gg.o.',
'..oggo..']},
};
