import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression


# ------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "green_en.csv"

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    # 타입 정리
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["emissions_per_area"] = pd.to_numeric(df["emissions_per_area"], errors="coerce")

    # year이나 타깃이 없는 행 제거
    df = df.dropna(subset=["year", "emissions_per_area"])
    df["year"] = df["year"].astype(int)

    return df


# ------------------------------------------------------------
# 2. 각 지역별 선형회귀로 2050년까지 예측
# ------------------------------------------------------------
@st.cache_data
def fit_linear_trend_and_forecast(df: pd.DataFrame, year_until: int = 2050):
    """
    df: green_en.csv
        columns = ['region', 'year', 'emissions', 'area', 'emissions_per_area']
    """

    df = df.copy()

    # 지역-연도별 평균값으로 정리
    grouped = (
        df.groupby(["region", "year"], as_index=False)
        .agg({"emissions_per_area": "mean"})
    )

    regions = grouped["region"].unique()
    min_year = int(grouped["year"].min())
    max_year = int(grouped["year"].max())

    all_rows = []
    hist_rows = []

    for region in regions:
        g = grouped[grouped["region"] == region].copy()
        g = g.sort_values("year")

        X = g[["year"]].values
        y = g["emissions_per_area"].values

        # 선형 회귀 적합
        model = LinearRegression()
        model.fit(X, y)

        # 과거 부분(실제 데이터)은 그대로 저장
        for _, r in g.iterrows():
            hist_rows.append({
                "region": region,
                "year": int(r["year"]),
                "type": "historical",
                "value": float(r["emissions_per_area"]),
            })

        # 전체 연도(최소 연도 ~ 2050)까지 예측
        years = np.arange(min_year, year_until + 1)
        y_pred = model.predict(years.reshape(-1, 1))

        for yr, val in zip(years, y_pred):
            all_rows.append({
                "region": region,
                "year": int(yr),
                "value": float(val),
            })

    hist_df = pd.DataFrame(hist_rows)
    forecast_df = pd.DataFrame(all_rows)
    forecast_df["type"] = "forecast"

    # 시각화용 full 데이터 (과거 + 선형 예측)
    full = pd.concat(
        [hist_df, forecast_df],
        ignore_index=True
    )

    return full, grouped, forecast_df


# ------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Linear Trend to 2050", layout="wide")
st.title("선형 회귀 기반 2050년 면적당 배출량 추세 대시보드")

st.write("""
**green_en.csv** (과거 데이터)를 사용해서  
각 지역별 **emissions_per_area(면적당 배출량)**에 대해  
**연도에 따른 선형 추세를 추정하고, 2050년까지 직선 경향을 연장**합니다.

- historical: 실제 과거 연도별 평균
- forecast: 선형 회귀로 연장한 2050년까지의 추세
""")

df_hist = load_data()
full, grouped_hist, forecast_df = fit_linear_trend_and_forecast(df_hist, year_until=2050)

regions = sorted(full["region"].unique())
selected_region = st.sidebar.selectbox("지역 선택 (region)", regions)

tab1, tab2, tab3 = st.tabs(["추세 그래프", "지역별 데이터", "전체 예측 다운로드"])


# ------------------------------------------------------------
# 4. 추세 그래프
# ------------------------------------------------------------
with tab1:
    st.subheader(f"{selected_region} — Historical vs Linear Trend (to 2050)")

    region_data_hist = full[(full["region"] == selected_region) & (full["type"] == "historical")].copy()
    region_data_fore = full[(full["region"] == selected_region) & (full["type"] == "forecast")].copy()

    region_data_hist = region_data_hist.sort_values("year")
    region_data_fore = region_data_fore.sort_values("year")

    # 피벗 형태로 만들기
    region_combined = pd.concat([region_data_hist, region_data_fore], ignore_index=True)
    pivot = (
        region_combined.pivot(index="year", columns="type", values="value")
        .sort_index()
    )

    st.line_chart(pivot)

    st.caption("""
- **historical**: 실제 과거 데이터 (연도별 평균)
- **forecast**: 과거 데이터를 기반으로 한 선형 회귀 직선(위/아래로 경향 보임)
""")


# ------------------------------------------------------------
# 5. 지역별 데이터 테이블
# ------------------------------------------------------------
with tab2:
    st.subheader(f"{selected_region} — 데이터 상세")

    region_hist = grouped_hist[grouped_hist["region"] == selected_region].copy()
    region_fore = forecast_df[forecast_df["region"] == selected_region].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Historical (과거)**")
        st.dataframe(
            region_hist[["region", "year", "emissions_per_area"]],
            use_container_width=True,
        )

    with col2:
        st.markdown("**Forecast (선형 추세 예측)**")
        st.dataframe(
            region_fore[["region", "year", "value"]],
            use_container_width=True,
        )


# ------------------------------------------------------------
# 6. 전체 예측 결과 CSV 다운로드
# ------------------------------------------------------------
with tab3:
    st.subheader("전체 예측 결과 (모든 지역, 모든 연도)")

    full_export = full.sort_values(["region", "year", "type"])
    csv_bytes = full_export.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 전체 선형 추세 예측 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="linear_trend_forecast_full.csv",
        mime="text/csv",
    )

    st.write("아래는 전체 예측 데이터 일부 미리보기입니다.")
    st.dataframe(full_export.head(100), use_container_width=True)
