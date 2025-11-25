import streamlit as st
import pandas as pd
from pathlib import Path
from urllib.parse import quote
import folium
from streamlit_folium import st_folium

# ------------------------------------------------------------
# 1. 파일명 설정
# ------------------------------------------------------------
HIST_NAME = "green.csv"          # 과거 실측 데이터
PRED_NAME = "XGBoostData.csv"    # 2050 예측 결과

REPO_BASE_URL = "https://raw.githubusercontent.com/suaaus/25aiproject0/main/"


# ------------------------------------------------------------
# 2. CSV 읽기 (로컬 → GitHub 순서)
# ------------------------------------------------------------
def read_csv_local_or_github(filename: str) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    local_path = base_dir / filename

    # 1) 로컬에서 먼저 찾기
    if local_path.exists():
        return pd.read_csv(local_path, encoding="utf-8-sig")

    # 2) 없다면 GitHub RAW URL에서 읽기
    url = REPO_BASE_URL + quote(filename)
    return pd.read_csv(url, encoding="utf-8-sig")


@st.cache_data
def load_data():
    hist = read_csv_local_or_github(HIST_NAME)
    pred = read_csv_local_or_github(PRED_NAME)

    # 컬럼 정리
    hist.columns = hist.columns.astype(str).str.replace("\ufeff", "").str.strip()
    pred.columns = pred.columns.astype(str).str.replace("\ufeff", "").str.strip()

    return hist, pred


# ------------------------------------------------------------
# 3. Streamlit 기본 UI
# ------------------------------------------------------------
st.set_page_config(page_title="2050년 XGBoost 예측 시각화", layout="wide")
st.title("2050년 면적당 배출량 XGBoost 예측 시각화")

st.write("""
과거 온실가스 데이터(green.csv)와  
XGBoost 예측 결과(XGBoostData.csv)를 활용한 2050년 시각화 대시보드입니다.
""")


# ------------------------------------------------------------
# 4. 데이터 로딩
# ------------------------------------------------------------
hist, pred = load_data()

regions = sorted(pred["지역"].unique())
selected_region = st.sidebar.selectbox("지역 선택", regions)

tab1, tab2, tab3 = st.tabs(["2050 예측 지도", "지역별 추세 그래프", "데이터 테이블"])


# ------------------------------------------------------------
# 5. 지도 시각화
# ------------------------------------------------------------
with tab1:
    st.subheader("2050년 면적당 배출량 예측 지도")

    center_lat = pred["위도"].mean()
    center_lon = pred["경도"].mean()

    m = folium.Map(location=[center_lat, center_lon], z
