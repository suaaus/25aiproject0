import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


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

    # year / 타깃 없는 행 제거
    df = df.dropna(subset=["year", "emissions_per_area"])
    df["year"] = df["year"].astype(int)

    return df


# ------------------------------------------------------------
# 2. 선형회귀 + XGBoost(잔차) 하이브리드 모델
#    - 선형회귀: 전체 추세(위/아래 방향)
#    - XGBoost: 선형으로 설명 안 되는 굴곡(잔차) 학습
# ------------------------------------------------------------
@st.cache_data
def fit_hybrid_trend(df: pd.DataFrame, year_until: int = 2050):
    """
    df: green_en.csv
        columns = ['region', 'year', 'emissions', 'area', 'emissions_per_area']
    """

    df = df.copy()

    # 지역-연도별 평균값 (연도별 대표값만 사용)
    grouped = (
        df.groupby(["region", "year"], as_index=False)
        .agg({"emissions_per_area": "mean"})
    )

    regions = grouped["region"].unique()
    min_year = int(grouped["year"].min())
    max_year = int(grouped["year"].max())

    hist_rows = []     # 실제 과거값
    forecast_rows = [] # 선형 + 하이브리드 예측

    # 모든 연도 범위 (최소연도 ~ 2050)
    full_years = np.arange(min_year, year_until + 1)

    for region in regions:
        g = grouped[grouped["region"] == region].copy()
        g = g.sort_values("year")

        X_hist = g["year"].values.reshape(-1, 1).astype(np.float32)
        y_hist = g["emissions_per_area"].values.astype(np.float32)

        # 1) 선형 회귀로 "큰 추세" 먼저 잡기
        lin = LinearRegression()
        lin.fit(X_hist, y_hist)

        # 과거 연도에 대한 선형 예측
        y_lin_hist = lin.predict(X_hist)
        # 잔차 = 실제 - 선형추세
        resid_hist = y_hist - y_lin_hist

        # 2) XGBoost로 잔차(residual) 학습
        xgb = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            tree_method="hist",
        )
        xgb.fit(X_hist, resid_hist)

        # 3) 전체 연도(최소연도~2050)에 대해 선형 + 잔차 예측
        X_full = full_years.reshape(-1, 1).astype(np.float32)
        y_lin_full = lin.predict(X_full)                # 선형 추세
        resid_full = xgb.predict(X_full)               # XGBoost 잔차
        y_hybrid_full = y_lin_full + resid_full        # 최종 하이브리드 예측

        # 과거 실제값 저장
        for yr, val in zip(g["year"], y_hist):
            hist_rows.append(
                {
                    "region": region,
                    "year": int(yr),
                    "type": "historical",
                    "value": float(val),
                }
            )

        # 전체 연도에 대해 linear / hybrid 모두 저장
        for yr, val_lin, val_hyb in zip(full_years, y_lin_full, y_hybrid_full):
            forecast_rows.append(
                {
                    "region": region,
                    "year": int(yr),
                    "linear": float(val_lin),
                    "hybrid": float(val_hyb),
                }
            )

    hist_df = pd.DataFrame(hist_rows)
    fore_df = pd.DataFrame(forecast_rows)

    # 시각화용 long 형태 만들기 (historical / linear / hybrid)
    full_list = []

    # 실제 과거
    for _, r in hist_df.iterrows():
        full_list.append(
            {
                "region": r["region"],
                "year": int(r["year"]),
                "type": "historical",
                "value": float(r["value"]),
            }
        )

    # 선형 / 하이브리드
    for _, r in fore_df.iterrows():
        full_list.append(
            {
                "region": r["region"],
                "year": int(r["year"]),
                "type": "linear",
                "value": float(r["linear"]),
            }
        )
        full_list.append(
            {
                "region": r["region"],
                "year": int(r["year"]),
                "type": "hybrid",
                "value": float(r["hybrid"]),
            }
        )

    full_df = pd.DataFrame(full_list)

    return full_df, hist_df, fore_df


# ------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Hybrid (Linear + XGBoost) Forecast to 2050", layout="wide")
st.title("선형회귀 + XGBoost 하이브리드 2050년 면적당 배출량 예측")

st.write("""
**green_en.csv** (과거 데이터)를 사용해  

1. 먼저 **선형 회귀**로 각 지역의 연도별 "큰 추세(위/아래 방향)"를 잡고,  
2. 그 추세에서 벗어나는 **잔차를 XGBoost로 학습**하여,  
3. 두 값을 더한 **하이브리드 예측값**을 2050년까지 계산합니다.

- **historical**: 실제 과거 데이터  
- **linear**: 선형 회귀로만 예측한 직선 추세  
- **hybrid**: 선형 추세 + XGBoost 잔차 예측값 (실제 패턴을 더 잘 따라감)
""")

df_hist = load_data()
full, hist_df, fore_df = fit_hybrid_trend(df_hist, year_until=2050)

regions = sorted(full["region"].unique())
selected_region = st.sidebar.selectbox("지역 선택 (region)", regions)

tab1, tab2, tab3 = st.tabs(["추세 그래프", "지역별 데이터", "전체 예측 다운로드"])


# ------------------------------------------------------------
# 4. 추세 그래프 (historical / linear / hybrid 비교)
# ------------------------------------------------------------
with tab1:
    st.subheader(f"{selected_region} — Historical vs Linear vs Hybrid (to 2050)")

    region_data = full[full["region"] == selected_region].copy()
    region_data = region_data.sort_values("year")

    # index = year, columns = type, values = value
    pivot = (
        region_data.pivot(index="year", columns="type", values="value")
        .sort_index()
    )

    st.line_chart(pivot)

    st.caption("""
- **historical**: 실제 과거 연도별 평균 데이터  
- **linear**: 선형 회귀만 사용한 직선 추세  
- **hybrid**: 선형 추세 + XGBoost 잔차 예측 (과거 구간에서 실제 곡선을 더 잘 따라감)
""")


# ------------------------------------------------------------
# 5. 지역별 데이터 테이블
# ------------------------------------------------------------
with tab2:
    st.subheader(f"{selected_region} — 데이터 상세")

    region_hist = hist_df[hist_df["region"] == selected_region].copy()
    region_fore = fore_df[fore_df["region"] == selected_region].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Historical (과거)**")
        st.dataframe(
            region_hist[["region", "year", "value"]]
            .rename(columns={"value": "emissions_per_area"}),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Forecast (linear & hybrid)**")
        st.dataframe(
            region_fore[["region", "year", "linear", "hybrid"]],
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
        label="📥 전체 하이브리드 예측 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="hybrid_linear_xgboost_forecast_full.csv",
        mime="text/csv",
    )

    st.write("아래는 전체 예측 데이터 일부 미리보기입니다.")
    st.dataframe(full_export.head(100), use_container_width=True)
