import streamlit as st
import streamlit.components.v1 as components

# ✅ 페이지 설정은 반드시 첫 Streamlit 호출로
st.set_page_config(page_title="Swing Picker Web v3.0.2 FullSync", layout="wide")

# --- GA4 ---
GA_MEASUREMENT_ID = "G-3PLRGRT2RL"
GA_SCRIPT = f"""
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

# ✅ 화면 살아있는지 즉시 표시(임시)
st.write("✅ App loaded")
