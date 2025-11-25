import streamlit as st
import pandas as pd
import os
from pathlib import Path
import folium
from streamlit_folium import st_folium

# =====================================================================
# 1) 같은 폴더에서 CSV 자동 탐색 + 읽기 (숨은 문자 포함 문제 자동 해결)
# =====================================================================
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    files = os.listdir(base_dir)

    # CSV 중 온실가스 파일 찾기
    hist_file = None
    pred_file = None

    for f in files:
        fname = f.replace("\ufeff", "").strip()
        if fname.endswith(".csv"):
            # 온실가스 데이터
            if "온실가스" in fname:
                hist_file = f
            # XGBoost 예측 결과
            if "XGBoost" in fname:
                pred_file = f

    if hist_file is None:
        raise FileNotFoundError(f"온실가스 CSV 파일을 못 찾았습니다. 폴더 목록: {files}")

    if pred_file is None:
        raise FileNotFoundError(f"XGBoost CSV 파일을 못 찾았습니다. 폴더 목록: {files}")

    # 파일 경로
    hist_path = base_dir / hist_file
    pred_path = base_dir / pred_file

    # CSV 읽기
    hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")

    # 컬럼 이름 정리 (BOM, 공백 완전 제거)
    hist.columns = hist.columns.astype(str).str.replace("\ufeff", "").str.strip()
    pred.columns = pred.columns.astype(str).str.replace("\ufeff", "").str.strip()

    return hist, pred


# =====================================================================
# 2) UI 기본 설정
# =====================================================================
st.set_page_config(page_title="2050년 XGBoost 예측 시각화", layout="wide")
st.title("2050년 온실가스 면적당 배출량 XGBoost 예측 시각화")

st.write("""
과거 실측 데이터와 XGBoost 예측 결과를 바탕으로  
2050년 면적당 배출량을 지도와 그래프로 시각화하는 대시보드입니다.
""")

# 데이터 불러오기
hist, pred = load_data()

# =====================================================================
# 3) 지역 선택
# =====================================================================
regions = sorted(pred["지역"].unique())
selected_region = st.sidebar.selectbox("지역 선택", regions)

# 탭 UI
tab1, tab2, tab3 = st.tabs(["2050 예측 지도", "지역별 추세 그래프", "데이터 테이블"])


# =====================================================================
# 4) 2050 예측 지도 (folium)
# =====================================================================
with tab1:
    st.subheader("2050년 면적당 배출량 예측 지도")

    center_lat = pred["위도"].mean()
    center_lon = pred["경도"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    max_val = pred["2050_면적당배출량"].max()

    for _, row in pred.iterrows():
        value = row["2050_면적당배출량"]
        radius = 5 + 15 * (value / max_val)

        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            popup=f"{row['지역']} : {value:.2f}",
            color="red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    st_folium(m, width=900, height=550)


# =====================================================================
# 5) 지역별 추세 그래프
# =====================================================================
with tab2:
    st.subheader(f"{selected_region} — 과거 면적당 배출량 + 2050 예측")

    region_hist = (
        hist[hist["지역"] == selected_region]
        .groupby("연도", as_index=False)["면적당_배출량"]
        .mean()
    )

    # 2050 예측값 추가
    row_pred = pred[pred["지역"] == selected_region]
    if len(row_pred) > 0:
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


# =====================================================================
# 6) 데이터 테이블
# =====================================================================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2050 예측 결과 (내림차순)")
        st.dataframe(
            pred.sort_values("2050_면적당배출량", ascending=False),
            use_container_width=True
        )

    with col2:
        st.subheader("과거 데이터 미리보기")
        st.dataframe(hist.head(50), use_container_width=True)
