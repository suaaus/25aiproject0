import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import folium
from streamlit_folium import st_folium


# ------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "green_en.csv"  # 업로드한 영어 버전

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["emissions_per_area"] = pd.to_numeric(df["emissions_per_area"], errors="coerce")

    df = df.dropna(subset=["year", "emissions_per_area"])
    df["year"] = df["year"].astype(int)

    return df


# ------------------------------------------------------------
# 2. 선형회귀 + XGBoost 잔차 하이브리드 예측
# ------------------------------------------------------------
@st.cache_data
def fit_hybrid(df: pd.DataFrame, year_until: int = 2050):
    df = df.copy()

    # 지역-연도별 평균 정리
    grouped = (
        df.groupby(["region", "year"], as_index=False)
        .agg({"emissions_per_area": "mean"})
    )

    regions = grouped["region"].unique()
    min_year = int(grouped["year"].min())
    max_year = int(grouped["year"].max())

    hist_rows = []
    forecast_rows = []

    all_years = np.arange(min_year, year_until + 1)

    for region in regions:
        g = grouped[grouped["region"] == region].sort_values("year")

        X_hist = g["year"].values.reshape(-1, 1).astype(np.float32)
        y_hist = g["emissions_per_area"].values.astype(np.float32)

        # 1) 선형 회귀로 전체 추세 잡기
        lin = LinearRegression()
        lin.fit(X_hist, y_hist)
        y_lin_hist = lin.predict(X_hist)
        resid_hist = y_hist - y_lin_hist

        # 2) 잔차를 XGBoost로 학습
        xgb = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
        xgb.fit(X_hist, resid_hist)

        # 3) 전체 연도(미래 포함) 선형+하이브리드 예측
        X_full = all_years.reshape(-1, 1).astype(np.float32)
        y_lin_full = lin.predict(X_full)
        resid_full = xgb.predict(X_full)
        y_hybrid_full = y_lin_full + resid_full

        # 실제 데이터 저장
        for yr, val in zip(g["year"], y_hist):
            hist_rows.append({
                "region": region, "year": int(yr),
                "type": "historical", "value": float(val)
            })

        # 미래 예측 저장
        for yr, v_lin, v_hyb in zip(all_years, y_lin_full, y_hybrid_full):
            forecast_rows.append({
                "region": region, "year": int(yr),
                "linear": float(v_lin), "hybrid": float(v_hyb)
            })

    hist_df = pd.DataFrame(hist_rows)
    fore_df = pd.DataFrame(forecast_rows)

    # 시각화용 full 데이터 (historical, linear, hybrid 모두 포함)
    full_rows = []

    for _, r in hist_df.iterrows():
        full_rows.append({
            "region": r["region"], "year": int(r["year"]),
            "type": "historical", "value": float(r["value"])
        })

    for _, r in fore_df.iterrows():
        full_rows.append({
            "region": r["region"], "year": int(r["year"]),
            "type": "linear", "value": float(r["linear"])
        })
        full_rows.append({
            "region": r["region"], "year": int(r["year"]),
            "type": "hybrid", "value": float(r["hybrid"])
        })

    full_df = pd.DataFrame(full_rows)

    return full_df, hist_df, fore_df


# ------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Hybrid Forecast + Map", layout="wide")
st.title("🌎 선형회귀 + XGBoost 하이브리드 기반 2050년 예측 대시보드")

df_hist = load_data()
full, hist_df, fore_df = fit_hybrid(df_hist)


# ------------------------------------------------------------
# 지도 만들 때 필요한 위도/경도 병합
# ------------------------------------------------------------
# green_en.csv에는 위도/경도 없음 → region별 대표 좌표 제공 필요
# → 여기에 기본 좌표 테이블 생성 (대한민국 시·도 중심)
region_coords = {
    "서울특별시": (37.5665, 126.9780),
    "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8714, 128.6014),
    "인천광역시": (37.4563, 126.7052),
    "광주광역시": (35.1595, 126.8526),
    "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114),
    "세종특별자치시": (36.4800, 127.2890),
    "경기도": (37.4363, 127.5500),
    "강원도": (37.8228, 128.1555),
    "충청북도": (36.8000, 127.7000),
    "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7175, 127.1530),
    "전라남도": (34.8194, 126.8930),
    "경상북도": (36.4919, 128.8889),
    "경상남도": (35.4606, 128.2132),
    "제주특별자치도": (33.4996, 126.5312),
}

# 좌표 merge
coord_df = pd.DataFrame(
    [{"region": k, "lat": v[0], "lon": v[1]} for k, v in region_coords.items()]
)

# 2050년 hybrid 예측값만 추출
pred_2050 = fore_df[fore_df["year"] == 2050]
pred_2050 = pred_2050.merge(coord_df, on="region", how="left")

regions = sorted(full["region"].unique())
selected_region = st.sidebar.selectbox("지역 선택", regions)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 예측 그래프", "📍 지도", "📋 지역 데이터", "📥 다운로드"
])


# ------------------------------------------------------------
# 4. 예측 그래프
# ------------------------------------------------------------
with tab1:
    st.subheader(f"{selected_region} — Historical / Linear / Hybrid")

    region_data = full[full["region"] == selected_region].copy()
    region_data = region_data.sort_values("year")

    pivot = region_data.pivot(index="year", columns="type", values="value")

    st.line_chart(pivot)


# ------------------------------------------------------------
# 5. 지도 시각화 (2050 기준)
# ------------------------------------------------------------
with tab2:
    st.subheader("🌎 2050년 하이브리드 예측 지도")

    m = folium.Map(location=[36.5, 127.9], zoom_start=7)

    max_val = pred_2050["hybrid"].max()

    for _, row in pred_2050.iterrows():
        if pd.isna(row["lat"]): continue

        val = row["hybrid"]
        radius = 5 + 20 * (val / max_val)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            popup=f"{row['region']} : {val:.2f}",
            color="red",
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    st_folium(m, width=900, height=600)


# ------------------------------------------------------------
# 6. 지역별 테이블
# ------------------------------------------------------------
with tab3:
    st.subheader(f"{selected_region} — Data Table")

    region_hist = hist_df[hist_df["region"] == selected_region]
    region_fore = fore_df[fore_df["region"] == selected_region]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Historical**")
        st.dataframe(region_hist, use_container_width=True)

    with col2:
        st.markdown("**Forecast (Linear + Hybrid)**")
        st.dataframe(region_fore, use_container_width=True)


# ------------------------------------------------------------
# 7. 전체 데이터 다운로드
# ------------------------------------------------------------
with tab4:
    st.subheader("전체 예측 결과 다운로드 (CSV)")

    csv_bytes = full.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 hybrid_linear_xgboost_2050_full.csv 다운로드",
        data=csv_bytes,
        file_name="hybrid_linear_xgboost_2050_full.csv",
        mime="text/csv",
    )

    st.dataframe(full.head(50))
