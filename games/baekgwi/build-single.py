#!/usr/bin/env python3
"""index.html + sprites.js + game.js 를 단일 HTML 한 장으로 합친다.

저장소에서는 세 파일로 나눠 두는 편이 고치기 쉽지만, 어딘가에 올리거나
파일 하나만 건네줄 때는 합쳐진 편이 편하다.

    python3 build-single.py            # baekgwi-single.html 생성
    python3 build-single.py out.html
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
TAG = '<script src="sprites.js"></script>\n<script src="game.js"></script>'


def build() -> str:
    html = (HERE / 'index.html').read_text(encoding='utf-8')
    if TAG not in html:
        raise SystemExit('index.html 안에서 스크립트 태그를 찾지 못했습니다.')
    joined = '<script>\n{}\n{}\n</script>'.format(
        (HERE / 'sprites.js').read_text(encoding='utf-8'),
        (HERE / 'game.js').read_text(encoding='utf-8'),
    )
    return html.replace(TAG, joined)


if __name__ == '__main__':
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'baekgwi-single.html'
    text = build()
    out.write_text(text, encoding='utf-8')
    print('{} — {:,} bytes'.format(out, len(text.encode('utf-8'))))
