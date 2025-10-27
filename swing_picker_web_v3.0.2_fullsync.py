# =======================================================
# 🔹 Google Analytics 4 (GA4) 접속자 추적 코드
# =======================================================
import streamlit as st
import streamlit.components.v1 as components

GA_MEASUREMENT_ID = "G-3PLRGRT2RL"  # ✅ 이두영님 전용 측정 ID

GA_SCRIPT = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
  window._gtagReady = true;
</script>
"""
st.markdown(GA_SCRIPT, unsafe_allow_html=True)

