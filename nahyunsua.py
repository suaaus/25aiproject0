import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import folium
from folium import CircleMarker
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ===========================
# 0. 기본 설정
# ===========================
st.set_page_config(
    page_title="대한민국 도시별 온실가스 배출량 예측 모델",
    layout="wide"
)

st.title("대한민국 도시별 온실가스 배출량 예측 모델")


# ===========================
# 1. 데이터 불러오기
# ===========================
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    hist_path = base_dir / "green_en.csv"
    coord_path = base_dir / "XGBoostData_en.csv"

    df_hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    df_coord = pd.read_csv(coord_path, encoding="utf-8-sig")

    df_hist.columns = df_hist.columns.str.strip()
    df_coord.columns = df_coord.columns.str.strip()

    df_hist["region"] = df_hist["region"].astype(str)
    df_hist["year"] = pd.to_numeric(df_hist["year"], errors="coerce").astype("Int64")
    df_hist["emissions_per_area"] = pd.to_numeric(df_hist.get("emissions_per_area"), errors="coerce")

    df_hist = df_hist.dropna(subset=["region", "year", "emissions_per_area"])
    df_hist["year"] = df_hist["year"].astype(int)

    # 지역 × 연도 평균
    agg_dict = {"emissions_per_area": "mean"}
    if "emissions" in df_hist.columns:
        agg_dict["emissions"] = "mean"
    if "area" in df_hist.columns:
        agg_dict["area"] = "mean"

    df_hist_clean = df_hist.groupby(["region", "year"], as_index=False).agg(agg_dict)

    df_coord["region"] = df_coord["region"].astype(str)
    df_coord["lat"] = pd.to_numeric(df_coord["lat"], errors="coerce")
    df_coord["lon"] = pd.to_numeric(df_coord["lon"], errors="coerce")

    return df_h_
