import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent

hist = pd.read_csv(base_dir / "온실가스_면적병합_전처리완료.csv")
pred = pd.read_csv(base_dir / "XGBoost_예측결과_요약.csv")
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="2050년 XGBoost 예측 시각화", layout="wide")

st.title("2050년 온실가스 면적당 배출량 XGBoost 예측 시각화")
st.markdown(
    """
### 사용 방법

1. 아래에서 **과거 데이터 CSV**와 **2050년 예측 결과 CSV**를 업로드해 주세요.  
2. 업로드가 끝나면 자동으로 지도, 그래프, 테이블이 생성됩니다.

- 과거 데이터 파일 예: `온실가스_면적병합_전처리완료.csv`  
  - 컬럼 예시: `지역, 연도, 배출량, 면적, 면적당_배출량`
- 예측 결과 파일 예: `XGBoost_예측결과_요약.csv`  
  - 컬럼 예시: `지역, 위도, 경도, 2050_면적당배출량`
"""
)

# ---------------------------
# 1. 파일 업로드
# ---------------------------
hist_file = st.file_uploader("① 과거 온실가스 데이터 CSV 업로드 (온실가스_면적병합_전처리완료.csv)", type=["csv"], key="hist")
pred_file = st.file_uploader("② 2050년 XGBoost 예측 결과 CSV 업로드 (XGBoost_예측결과_요약.csv)", type=["csv"], key="pred")

if not hist_file or not pred_file:
    st.info("위의 두 CSV 파일을 모두 업로드하면 분석 결과가 표시됩니다.")
    st.stop()

# ---------------------------
# 2. 데이터 읽기 + 컬럼 정리
# ---------------------------
hist = pd.read_csv(hist_file)
pred = pd.read_csv(pred_file)

# 컬럼 이름 공백/BOM 제거
hist.columns = hist.columns.astype(str).str.replace("\ufeff", "").str.strip()
pred.columns = pred.columns.astype(str).str.replace("\ufeff", "").str.strip()

# 디버깅용 확인 (원하면 잠깐 켜놓고, 나중에 주석처리 해도 됨)
# st.write("hist 컬럼:", hist.columns.tolist())
# st.write("pred 컬럼:", pred.columns.tolist())

필수_컬럼_hist = {"지역", "연도", "면적당_배출량"}
필수_컬럼_pred = {"지역", "위도", "경도", "2050_면적당배출량"}

if not 필수_컬럼_hist.issubset(set(hist.columns)):
    st.error(f"과거 데이터 파일에 필요한 컬럼이 없습니다. 최소 포함되어야 하는 컬럼: {필수_컬럼_hist}")
    st.stop()

if not 필수_컬럼_pred.issubset(set(pred.columns)):
    st.error(f"예측 데이터 파일에 필요한 컬럼이 없습니다. 최소 포함되어야 하는 컬럼: {필수_컬럼_pred}")
    st.stop()

# ---------------------------
# 3. 지역 선택 (사이드바)
# ---------------------------
regions = sorted(pred["지역"].unique())
selected_region = st.sidebar.selectbox("지역 선택", regions)

tab1, tab2, tab3 = st.tabs(["2050 예측 지도", "지역별 추세 그래프", "데이터 테이블"])

# ---------------------------
# 4. 2050 예측 지도
# --------------------------
