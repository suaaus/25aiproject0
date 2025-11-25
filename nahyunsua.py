import streamlit as st
import pandas as pd
from pathlib import Path
import folium
from streamlit_folium import st_folium


# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent

    # 과거 실측 데이터
    hist_path = base_dir / "온실가스_면적병합_전처리완료.csv"
    hist = pd.read_csv(hist_path)

    # 2050 XGBoost 예측 결과
    pred_path = base_dir / "XGBoost_예측결과_요약.csv"
    pred = pd.read_csv(pred_path)

    return hist, pred


hist, pred = load_data()

# ---------------------------
# 2. 기본 UI 설정
# ---------------------------
st.set_page_config(page_title="2050년 온실가스 XGBoost 예측 시각화", layout="wide")

st.title("2050년 온실가스 면적당 배출량 XGBoost 예측 시각화")
st.markdown(
    """
과거 온실가스 데이터와 XGBoost 예측 결과를 바탕으로  
**지역별 2050년 면적당 배출량**을 지도와 그래프로 살펴보는 대시보드입니다.
"""
)

# 지역 목록 (예측결과에 있는 지역 기준)
regions = sorted(pred["지역"].unique())
selected_region = st.sidebar.selectbox("지역 선택", regions)

# ---------------------------
# 3. 탭 구성
# ---------------------------
tab1, tab2, tab3 = st.tabs(["2050 예측 지도", "지역별 추세 그래프", "데이터 테이블"])


# ---------------------------
# 3-1. 2050 예측 지도 (folium)
# ---------------------------
with tab1:
    st.subheader("2050년 면적당 배출량 예측 지도")

    center_lat = pred["위도"].mean()
    center_lon = pred["경도"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    max_val = pred["2050_면적당배출량"].max()

    for _, row in pred.iterrows():
        value = row["2050_면적당배출량"]
        radius = 5 + 15 * (value / max_val)  # 예측값에 따라 원 크기 조절

        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            popup=f"{row['지역']} : {value:.2f}",
            color="red",
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)

    st_folium(m, width=800, height=550)

    st.markdown("**설명:** 원이 클수록 2050년 면적당 배출량 예측값이 큰 지역입니다.")


# ---------------------------
# 3-2. 지역별 과거 + 2050 추세 그래프
# ---------------------------
with tab2:
    st.subheader(f"{selected_region} — 과거 면적당 배출량 + 2050년 예측")

    # 선택한 지역의 과거 데이터 (연도별 평균)
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

    st.markdown(
        """
- **실선 구간:** 과거 실측 데이터를 연도별 평균으로 나타낸 값  
- **마지막 점(2050):** XGBoost로 예측한 2050년 면적당 배출량
"""
    )


# ---------------------------
# 3-3. 데이터 테이블
# ---------------------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2050년 예측 결과 (내림차순 정렬)")
        pred_sorted = pred.sort_values("2050_면적당배출량", ascending=False)
        st.dataframe(pred_sorted, use_container_width=True)

    with col2:
        st.subheader("원본 온실가스 데이터 (일부 미리보기)")
        st.dataframe(hist.head(100), use_container_width=True)
