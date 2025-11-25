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
    path = base_dir / "green_en.csv"

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    # 타입 정리
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["emissions_per_area"] = pd.to_numeric(df["emissions_per_area"], errors="coerce")

    # 핵심 타깃이 없는 행은 제거
    df = df.dropna(subset=["year", "emissions_per_area"])
    df["year"] = df["year"].astype(int)

    return df


# ------------------------------------------------------------
# 2. XGBoost 학습 + 2050년까지 예측
# ------------------------------------------------------------
@st.cache_data
def train_and_forecast(df: pd.DataFrame, year_until: int = 2050):
    """
    df: green_en.csv
        columns = ['region', 'year', 'emissions', 'area', 'emissions_per_area']
    """

    df = df.copy()

    # 1) region → code
    le = LabelEncoder()
    df["region_code"] = le.fit_transform(df["region"])

    # 2) 지역/연도별 평균 면적당 배출량 (노이즈 제거)
    grouped = (
        df.groupby(["region", "region_code", "year"], as_index=False)
        .agg({"emissions_per_area": "mean"})
    )

    # 3) 학습용 X, y
    X = grouped[["year", "region_code"]].astype(float)
    y = grouped["emissions_per_area"].astype(float)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
    )

    # numpy array로 확실히 넘겨서 타입 문제 방지
    model.fit(X.values, y.values)

    # 4) 예측용 (과거 + 미래 전체 연도에 대해 예측)
    min_year = int(grouped["year"].min())
    max_year = int(grouped["year"].max())

    all_years = list(range(min_year, year_until + 1))

    regions = grouped[["region", "region_code"]].drop_duplicates()

    rows = []
    for _, row in regions.iterrows():
        r_name = row["region"]
        r_code = int(row["region_code"])
        for yr in all_years:
            rows.append({"region": r_name, "region_code": r_code, "year": yr})

    all_df = pd.DataFrame(rows)

    X_all = all_df[["year", "region_code"]].astype(float)
    all_df["pred_emissions_per_area"] = model.predict(X_all.values)

    # 5) 시각화용 full 데이터 (과거 + 예측)
    hist_for_plot = grouped.rename(columns={"emissions_per_area": "value"})
    hist_for_plot["type"] = "historical"

    fut_for_plot = all_df.rename(
        columns={"pred_emissions_per_area": "value"}
    )
    fut_for_plot["type"] = "forecast"

    full = pd.concat(
        [hist_for_plot[["region", "year", "type", "value"]],
         fut_for_plot[["region", "year", "type", "value"]]],
        ignore_index=True,
    )

    return full, grouped, all_df


# ------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="XGBoost 2050 Forecast", layout="wide")
st.title("XGBoost 기반 2050년 면적당 배출량 예측 대시보드")

st.write("""
**green_en.csv** (과거 데이터)를 사용해서  
XGBoost 회귀 모델로 2050년까지의 **emissions_per_area**(면적당 배출량)을 예측합니다.

- 왼쪽에서 지역을 선택하면  
  → 과거(Historical) + 2050년까지 예측(Forecast) 추세 그래프  
  → 아래 탭에서 지역별 상세 데이터와 전체 예측 결과를 볼 수 있습니다.
""")

df_hist = load_data()
full, grouped_hist, all_pred = train_and_forecast(df_hist, year_until=2050)

regions = sorted(full["region"].unique())
selected_region = st.sidebar.selectbox("지역 선택 (region)", regions)

tab1, tab2, tab3 = st.tabs(["추세 그래프", "지역별 데이터", "전체 예측 다운로드"])


# ------------------------------------------------------------
# 4. 추세 그래프
# ------------------------------------------------------------
with tab1:
    st.subheader(f"{selected_region} — Historical vs Forecast (to 2050)")

    region_data = full[full["region"] == selected_region].copy()
    region_data = region_data.sort_values("year")

    # index = year, columns = type (historical/forecast), values = value
    pivot = (
        region_data.pivot(index="year", columns="type", values="value")
        .sort_index()
    )

    st.line_chart(pivot)

    st.caption("""
- **historical**: 실제 과거 데이터 (연도별 평균)
- **forecast**: XGBoost 모델로 예측한 값
""")


# ------------------------------------------------------------
# 5. 지역별 데이터 테이블
# ------------------------------------------------------------
with tab2:
    st.subheader(f"{selected_region} — 데이터 상세")

    region_hist = grouped_hist[grouped_hist["region"] == selected_region].copy()
    region_pred = all_pred[all_pred["region"] == selected_region].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Historical (과거)**")
        st.dataframe(
            region_hist[["region", "year", "emissions_per_area"]],
            use_container_width=True,
        )

    with col2:
        st.markdown("**Forecast (예측, 전체 연도)**")
        st.dataframe(
            region_pred[["region", "year", "pred_emissions_per_area"]],
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
        label="📥 전체 예측 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="xgboost_forecast_full.csv",
        mime="text/csv",
    )

    st.write("아래는 전체 예측 데이터 일부 미리보기입니다.")
    st.dataframe(full_export.head(100), use_container_width=True)
