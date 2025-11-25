import streamlit as st
import pandas as pd
from pathlib import Path
import folium
from streamlit_folium import st_folium


# -------------------------------------------------------
# 1. 같은 폴더에 있는 CSV 두 개 읽기
# -------------------------------------------------------
def load_data():
    base_dir = Path(__file__).resolve().parent

    hist_path = base_dir / "온실가스_면적병합_전처리완료.csv"
    pred_path = base_dir / "XGBoost_예측결과_요약.csv"

    # CSV 읽기 (인코딩은 상황에 맞게)
    hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")

    # 컬럼 이름 공백/BOM 제거
    hist.columns = hist.columns.astype(str).str.replace("\ufeff", "").str.strip()
    pred.columns = pred.columns.astype(str).str.replace("\ufeff", "").str.strip()

    return hist, pred


# -------------------------------------------------------
# 2. 기본 UI 설정
# -------------------------------------------------------
st.set_page_config(page_title="2050년 XGBoost 예측 시각화", layout="wide")
st.title("2050년 온실가스 면적당 배출량 XGBoost 예측 시각화")

st.write(
    """
과거 온실가스 데이터와 XGBoost 예측 결과를 기반으로  
**2050년 면적당 배출량**을 지도와 그래프로 시각화하는 대시보드입니다.
"""
)

# 데이터 불러오기
hist, pred = load_data()

# -------------------------------------------------------
# 3. 지역 선택 (사이드바)
# -------------------------------------------------------
regions = sorted(pred["지역"].unique())
selected_region = st.sidebar.selectbox("지역 선택", regions)

tab1, tab2, tab3 = st.tabs(["2050 예측 지도", "지역별 추세 그래프", "데이터 테이블"])

# -------------------------------------------------------
# 4. 2050 예측 지도
# -------------------------------------------------------
with tab1:
    st.subheader("2050년 면적당 배출량 예측 지도")

    center_lat = pred["위도"].mean()
    center_lon = pred["경도"].mean()

    # 기본 지도
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    max_val = pred["2050_면적당배출량"].max()

    # 지역별 원 그리기
    for _, row in pred.iterrows():
        value = row["2050_면적당배출량"]
        radius = 5 + 15 * (value / max_val)  # 값 비례 원 크기

        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            popup=f"{row['지역']} : {value:.2f}",
            color="red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    st_folium(m, width=900, height=550)


# -------------------------------------------------------
# 5. 지역별 과거 + 2050 추세 그래프
# -------------------------------------------------------
with tab2:
    st.subheader(f"{selected_region} — 과거 면적당 배출량 + 2050년 예측")

    # 선택 지역 과거 데이터 (연도별 평균)
    region_hist = (
        hist[hist["지역"] == selected_region]
        .groupby("연도", as_index=False)["면적당_배출량"]
        .mean()
    )

    # 2050 예측값 붙이기
    row_pred = pred[pred["지역"] == selected_region]
    if not row_pred.empty:
        val_2050 = row_pred["2050_면적당배출량"].iloc[0]
        region_hist = pd.concat(
            [
                region_hist,
                pd.DataFrame({"연도": [2050], "면적당_배출량": [val_2050]}),
            ],
            ignore_index=True,
        )

    region_hist = region_hist.sort_values("연도")
    st.line_chart(region_hist.set_index("연도"))


# -------------------------------------------------------
# 6. 데이터 테이블
# -------------------------------------------------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2050년 예측 결과 (내림차순)")
        st.dataframe(
            pred.sort_values("2050_면적당배출량", ascending=False),
            use_container_width=True,
        )

    with col2:
        st.subheader("과거 데이터 미리보기")
        st.dataframe(hist.head(50), use_container_width=True)
