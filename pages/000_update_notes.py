# pages/000_update_notes.py
# -*- coding: utf-8 -*-

import streamlit as st
from version_info import APP_VERSION, CHANGELOG

st.set_page_config(page_title="업데이트 노트", page_icon="🧩", layout="wide")

st.title("🧩 LDY Pro Trader 업데이트 노트")
st.caption(f"현재 서비스 버전: v{APP_VERSION}")

if CHANGELOG:
    latest = CHANGELOG[0]
    st.success(
        f"가장 최근 업데이트: v{latest['version']} · {latest['date']} – {latest['title']}"
    )

st.markdown("---")

for entry in CHANGELOG:
    with st.expander(
        f"v{entry['version']} · {entry['date']} – {entry['title']}",
        expanded=(entry["version"] == APP_VERSION),
    ):
        if entry.get("items"):
            for item in entry["items"]:
                st.markdown(f"- {item}")
        else:
            st.write("상세 변경 내역 준비 중입니다.")
