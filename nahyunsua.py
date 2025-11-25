import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import folium
from folium import CircleMarker
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ===========================
# 0. 기본 설정
# ===========================
st.set_page_config(
    page_title="대한민국 도시별 온실가스 배출량 예측 모델",
    layout="wide"
)

st.title("대한민국 도시별 온실가스 배출량 예측 모델")


# ===========================
# 1. 데이터 불러오기
# ===========================
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    hist_path = base_dir / "green_en.csv"
    coord_path = base_dir / "XGBoostData_en.csv"

    df_hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    df_coord = pd.read_csv(coord_path, encoding="utf-8-sig")

    df_hist.columns = df_hist.columns.str.strip()
    df_coord.columns = df_coord.columns.str.strip()

    df_hist["region"] = df_hist["region"].astype(str)
    df_hist["year"] = pd.to_numeric(df_hist["year"], errors="coerce").astype("Int64")
    df_hist["emissions_per_area"] = pd.to_numeric(df_hist.get("emissions_per_area"), errors="coerce")

    df_hist = df_hist.dropna(subset=["region", "year", "emissions_per_area"])
    df_hist["year"] = df_hist["year"].astype(int)

    # 지역 × 연도 평균
    agg_dict = {"emissions_per_area": "mean"}
    if "emissions" in df_hist.columns:
        agg_dict["emissions"] = "mean"
    if "area" in df_hist.columns:
        agg_dict["area"] = "mean"

    df_hist_clean = df_hist.groupby(["region", "year"], as_index=False).agg(agg_dict)

    df_coord["region"] = df_coord["region"].astype(str)
    df_coord["lat"] = pd.to_numeric(df_coord["lat"], errors="coerce")
    df_coord["lon"] = pd.to_numeric(df_coord["lon"], errors="coerce")

    return df_hist_clean, df_coord


# ===========================
# 2. 모델 학습 + 예측
# ===========================
@st.cache_data
def train_and_forecast(df_hist: pd.DataFrame, year_until: int = 2050):

    regions = sorted(df_hist["region"].unique())
    min_year = int(df_hist["year"].min())
    all_years = np.arange(min_year, year_until + 1)

    full_rows = []
    mae_rows = []

    for region in regions:
        g = df_hist[df_hist["region"] == region].sort_values("year")
        years = g["year"].values.astype(np.float32)
        y = g["emissions_per_area"].values.astype(np.float32)

        X_hist = years.reshape(-1, 1)

        lin = LinearRegression()
        lin.fit(X_hist, y)
        y_lin_hist = lin.predict(X_hist)
        resid_hist = y - y_lin_hist

        use_xgb = len(g) >= 4
        if use_xgb:
            xgb = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
            )
            xgb.fit(X_hist, resid_hist)
            resid_pred_hist = xgb.predict(X_hist)
        else:
            resid_pred_hist = np.zeros_like(resid_hist)

        y_hybrid_hist = y_lin_hist + resid_pred_hist

        X_full = all_years.reshape(-1, 1).astype(np.float32)
        y_lin_full = lin.predict(X_full)

        if use_xgb:
            resid_full = xgb.predict(X_full)
        else:
            resid_full = np.zeros_like(y_lin_full)

        y_hybrid_full = y_lin_full + resid_full

        mae = float(mean_absolute_error(y, y_hybrid_hist))
        mae_rows.append({"region": region, "MAE": mae})

        for yr, actual, pred in zip(years, y, y_hybrid_hist):
            full_rows.append({
                "region": region,
                "year": int(yr),
                "kind": "history",
                "actual": float(actual),
                "pred": float(pred),
            })

        for yr, pred in zip(all_years, y_hybrid_full):
            if yr in years:
                continue
            full_rows.append({
                "region": region,
                "year": int(yr),
                "kind": "forecast",
                "actual": np.nan,
                "pred": float(pred),
            })

    df_full = pd.DataFrame(full_rows)
    df_mae = pd.DataFrame(mae_rows).sort_values("MAE")

    return df_full, df_mae


# ===========================
# 3. 추세 설명
# ===========================
def describe_trend_and_solution(df_full: pd.DataFrame, region: str) -> str:

    df_r = df_full[df_full["region"] == region].sort_values("year")
    x = df_r["year"].values
    y = df_r["pred"].values

    slope = np.polyfit(x, y, 1)[0]
    start_val = y[0]
    end_val = y[-1]

    if slope > 0:
        trend_text = f"→ {region}은(는) 배출량이 **증가하는 추세**입니다."
    elif slope < 0:
        trend_text = f"→ {region}은(는) 배출량이 **감소하는 추세**입니다."
    else:
        trend_text = f"→ {region}은(는) **변화가 거의 없는 정체 상태**입니다."

    change_ratio = (end_val - start_val) / max(start_val, 1e-6) * 100
    change_text = f"   · 2050년에는 약 **{change_ratio:.1f}%** 변화가 예상됩니다."

    text = "\n".join([trend_text, change_text])
    return text


# ===========================
# 4. 지도 생성 (마커 클릭 제거, 툴팁만)
# ===========================
def create_map(df_full, df_coord, selected_year):

    df_year = df_full[df_full["year"] == selected_year].copy()
    if df_year.empty:
        return None

    df_year["value"] = df_year["pred"]
    df_map = pd.merge(df_year, df_coord, on="region", how="inner")
    df_map = df_map.dropna(subset=["lat", "lon", "value"])

    vmin = df_map["value"].min()
    vmax = df_map["value"].max()

    cmap = LinearColormap(
        colors=["#4575b4", "#ffffbf", "#d73027"],
        vmin=vmin, vmax=vmax
    )

    m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles="cartodbpositron")

    base_radius = 7
    extra_radius = 10

    for _, row in df_map.iterrows():

        if vmax > vmin:
            norm = (row["value"] - vmin) / (vmax - vmin)
        else:
            norm = 0.5

        radius = base_radius + extra_radius * norm

        CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=cmap(row["value"]),
            fill=True,
            fill_color=cmap(row["value"]),
            fill_opacity=0.9,
            weight=1.5,
            tooltip=f"{row['value']:.2f} tCO₂eq/km²"
        ).add_to(m)

    cmap.caption = f"{selected_year}년 면적당 온실가스 배출량"
    cmap.add_to(m)

    return m


# ===========================
# 5. 메인 UI
# ===========================
df_hist, df_coord = load_data()
df_full, df_mae = train_and_forecast(df_hist, year_until=2050)

min_year = int(df_full["year"].min())

tab1, tab2, tab3 = st.tabs([
    "1) 지도 & 지역별 추세",
    "2) 데이터 & 다운로드",
    "3) 예측 정확도",
])

# TAB 1
with tab1:

    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        selected_year = st.slider("연도 선택", min_value=min_year, max_value=2050, value=2050)
        all_regions = sorted(df_full["region"].unique())
        selected_region = st.selectbox("지역 선택", all_regions)

    with col_map:
        m = create_map(df_full, df_coord, selected_year)
        if m:
            st_folium(m, height=600, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 선택된 지역: **{selected_region}**")

    df_r = df_full[df_full["region"] == selected_region].sort_values("year")
    df_chart = df_r.pivot(index="year", columns="kind", values="pred")

    st.line_chart(df_chart)
    st.markdown("#### 추세 요약 & 정책 제안")
    st.markdown(describe_trend_and_solution(df_full, selected_region))


# TAB 2
with tab2:
    st.dataframe(df_full)
    st.download_button(
        label="전체 데이터 다운로드",
        data=df_full.to_csv(index=False).encode("utf-8-sig"),
        file_name="korea_emissions_full.csv",
        mime="text/csv"
    )


# TAB 3
with tab3:
    st.dataframe(df_mae)
    st.bar_chart(df_mae.set_index("region")["MAE"])
