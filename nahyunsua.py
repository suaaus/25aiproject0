import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# GitHub RAW URL에서 직접 읽기
HIST_URL = "https://raw.githubusercontent.com/suaaus/25aiproject0/main/온실가스_면적병합_전처리완료.csv"
PRED_URL = "https://raw.githubusercontent.com/suaaus/25aiproject0/main/XGBoost_예측결과_요약.csv"

def load_data():
    hist = pd.read_csv(HIST_URL, encoding="utf-8-sig")
    pred = pd.read_csv(PRED_URL, encoding="utf-8-sig")

    hist.columns = hist.columns.astype(str).str.replace("\ufeff", "").str.strip()
    pred.columns = pred.columns.astype(str).str.replace("\ufeff", "").str.strip()

    return hist, pred

st.set_page_config(page_title="2050년 XGBoost 예측 시각화", layout="wide")
st.title("2050년 온실가스 면적당 배출량 XGBoost 예측 시각화")

hist, pred = load_data()
