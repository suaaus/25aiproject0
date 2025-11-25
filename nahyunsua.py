import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------

@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    hist_path = base_dir / "green_en.csv"   # 과거 데이터 (영어 컬럼)

    df = pd.read_csv(hist_path, encoding="utf-8-sig")

    # 컬럼 정리 (혹시 모를 공백 제거)
    df.columns = df.columns.astype(str).str.strip()

    # year 정수형
    df["year"] = df["year"].astype(int)

    return df


# ------------------------------------------------------------
# 2. XGBoost 학습 + 2050년까지 예측
# ------------------------------------------------------------

@st.cache_data
def train_and_forecast(df: pd.DataFrame, year_until: int = 2050):
    """
    df: green_en.csv 로부터 읽은 원본 데이터
        columns: ['region', 'year', 'emissions', 'area', 'emissions_per_area']
    """

    # (1) 지역 라벨 인코딩
    le_region = LabelEncoder()
    df["region_code"] = le_region.fit_transform(df["region"])

    # (2) 연도별 평균(노이즈 줄이기) – 지역/연도 단위로 평균 면적당 배출량 사용
    grouped = (
        df.groupby(["region", "region_code", "year"], as_index=False)
        .agg({"emissions_per_area": "mean"})
    )

    # 학습 데이터
    X = grouped[["year", "region_code"]]
    y = grouped["emissions_per_area"]

    # (3) XGBoost 회귀 모델
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    # (4) 미래 연도(현재 데이터의 마지막 연도+1 ~ 2050) 생성
    max_hist_year = grouped["year"].max()
    future_years = list(range(max_hist_year + 1, year_until + 1))

    regions = grouped[["region", "region_code"]].drop_duplicates()

    future_rows = []
    for _, row in regions.iterrows():
        r_name = row["region"]
        r_code = row["region_code"]
        for yr in future_years:
            future_rows.append(
                {"region": r_name, "region_code": r_code, "year": yr}
            )

    future_df = pd.DataFrame(future_rows)

    # (5) 예측
    X_future = future_df[["year", "region_code"]]
    future_df["pred_emissions_per_area"] = model.predict(X_future)

    # (6) 과거 + 예측 합치기 (시각화용)
    hist_for_plot = grouped.rename(
        columns={"emissions_per_area": "value"}
    )
    hist_for_plot["type"] = "historical"

    fut_for_plot = future_df.rename(
        columns={"pred_emissions_per_area": "value"}
    )
    fut_for_plot["type"] = "forecast"

    full = pd.concat(
        [hist_for_plot, fut_for_plot],
        ignore_index=True
    )

    return full, grouped, future_df


# ------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------

st.set_page_config(page_title="XGBoost 2050 Forecast", layout="wide")
st.title("XGBoost 기반 2050년 면적당 배출량 예측 대시보드")

st.write("""
이 대시보드는 **green_en.csv**(과거 데이터)를 사용해  
XGBoost 회귀 모델로 2050년까지의 **emissions_per_area**(면적당 배출량)을 예측합니다.

- 왼쪽에서 지역을 선택하면  
  → 과거 + 2050년까지의 예측 추세 그래프가 나오고  
  → 아래에는 해당 지역의 데이터 테이블이 함께 표시됩니다.
""")

# 데이터 & 모델
df_hist = load_data()
full, grouped_hist, future_pred = train_and_forecast(df_hist, year_until=2050)

regions = sorted(full["region"].unique())
selected_region = st.sidebar.selectbox("지역 선택 (region)", regions)

tab1, tab2 = st.tabs(["추세 그래프 (Trend)", "데이터 테이블 (Table)"])

# ------------------------------------------------------------
# 4. 추세 그래프
# ------------------------------------------------------------
with tab1:
    st.subheader(f"{selected_region} — Historical vs Forecast (to 2050)")

    region_data = full[full["region"] == selected_region].copy()
    region_data = region_data.sort_values("year")

    # Streamlit line_chart를 쓰기 위해 피벗 형태로 변환
    # index = year, columns = type, values = value
    pivot = (
        region_data.pivot(index="year", columns="type", values="value")
        .sort_index()
    )

    st.line_chart(pivot)

    st.caption("• historical: 실제 과거 데이터 평균 (연도별)\n"
               "• forecast: XGBoost로 예측한 값")


# ------------------------------------------------------------
# 5. 데이터 테이블
# ------------------------------------------------------------
with tab2:
    st.subheader(f"{selected_region} — 데이터 상세 (historical + forecast)")

    region_hist = grouped_hist[grouped_hist["region"] == selected_region].copy()
    region_fut = future_pred[future_pred["region"] == selected_region].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Historical (과거)**")
        st.dataframe(
            region_hist[["region", "year", "emissions_per_area"]],
            use_container_width=True,
        )

    with col2:
        st.markdown("**Forecast (예측)**")
        st.dataframe(
            region_fut[["region", "year", "pred_emissions_per_area"]],
            use_container_width=True,
        )

    st.markdown("### 전체 예측 결과 다운로드")

    # 전체 예측 CSV (모든 지역, 모든 연도)
    full_export = full.copy()
    full_export = full_export.sort_values(["region", "year", "type"])

    csv_bytes = full_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 전체 예측 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="xgboost_forecast_full.csv",
        mime="text/csv",
    )
