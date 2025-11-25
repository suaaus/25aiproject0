import streamlit as st
import pandas as pd
from pathlib import Path
import os


@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    files = os.listdir(base_dir)

    st.write("📄 파일 목록:", files)

    # CSV 자동 탐색
    hist_file = next((f for f in files if "온실가스" in f and f.endswith(".csv")), None)
    pred_file = next((f for f in files if "XGBoost" in f and f.endswith(".csv")), None)

    st.write("🔍 감지된 hist 파일:", hist_file)
    st.write("🔍 감지된 pred 파일:", pred_file)

    if hist_file is None or pred_file is None:
        st.error("CSV 파일을 찾을 수 없습니다. 파일명을 확인하세요.")
        st.stop()

    # 실제 경로 생성
    hist_path = base_dir / hist_file
    pred_path = base_dir / pred_file

    # 읽기
    hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")

    # 컬럼 정리 (BOM 제거 포함)
    hist.columns = hist.columns.astype(str).str.replace("\ufeff", "").str.strip()
    pred.columns = pred.columns.astype(str).str.replace("\ufeff", "").str.strip()

    return hist, pred
