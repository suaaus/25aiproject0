import streamlit as st
import pandas as pd
from pathlib import Path
import os

@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent

    st.write("📂 base_dir:", base_dir)
    st.write("📄 base_dir 안 파일들:", os.listdir(base_dir))

    hist_name = "온실가스_면적병합_전처리완료.csv"
    pred_name = "XGBoost_예측결과_요약.csv"

    st.write("✅ hist 파일 존재?:", hist_name in os.listdir(base_dir))
    st.write("✅ pred 파일 존재?:", pred_name in os.listdir(base_dir))

    hist_path = base_dir / hist_name
    pred_path = base_dir / pred_name

    st.write("🔗 hist_path:", hist_path)
    st.write("🔗 pred_path:", pred_path)

    # 여기서 실제 에러 메시지도 같이 보기
    try:
        hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    except Exception as e:
        st.write("❌ hist 읽기 에러:", repr(e))
        st.stop()

    try:
        pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    except Exception as e:
        st.write("❌ pred 읽기 에러:", repr(e))
        st.stop()

    return hist, pred

hist, pred = load_data()
