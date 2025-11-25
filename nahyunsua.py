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
# 1. 데이터 불러오기 (+ 전처리: 지역×연도 평균)
# ===========================
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    hist_path = base_dir / "green_en.csv"          # 과거 데이터
    coord_path = base_dir / "XGBoostData_en.csv"   # 지역별 위도/경도

    df_hist = pd.read_csv(hist_path, encoding="utf-8-sig")
    df_coord = pd.read_csv(coord_path, encoding="utf-8-sig")

    # 컬럼 이름 공백 제거
    df_hist.columns = df_hist.columns.str.strip()
    df_coord.columns = df_coord.columns.str.strip()

    # 타입 정리
    df_hist["region"] = df_hist["region"].astype(str)
    df_hist["year"] = pd.to_numeric(df_hist["year"], errors="coerce").astype("Int64")
    df_hist["emissions_per_area"] = pd.to_numeric(
        df_hist.get("emissions_per_area"), errors="coerce"
    )

    # 쓰레기 행 제거
    df_hist = df_hist.dropna(subset=["region", "year", "emissions_per_area"])
    df_hist["year"] = df_hist["year"].astype(int)

    # 🔥 핵심 전처리: 지역×연도별 평균으로 1행씩만 남기기
    agg_dict = {"emissions_per_area": "mean"}
    if "emissions" in df_hist.columns:
        agg_dict["emissions"] = "mean"
    if "area" in df_hist.columns:
        agg_dict["area"] = "mean"

    df_hist_clean = (
        df_hist
        .groupby(["region", "year"], as_index=False)
        .agg(agg_dict)
    )

    # 좌표 타입 정리
    df_coord["region"] = df_coord["region"].astype(str)
    df_coord["lat"] = pd.to_numeric(df_coord["lat"], errors="coerce")
    df_coord["lon"] = pd.to_numeric(df_coord["lon"], errors="coerce")

    return df_hist_clean, df_coord


# ===========================
# 2. 선형회귀 + XGBoost(잔차) 하이브리드 예측 + MAE
# ===========================
@st.cache_data
def train_and_forecast(df_hist: pd.DataFrame, year_until: int = 2050):
    """
    1) 각 지역별로 (year → emissions_per_area) 선형회귀로 큰 추세 잡기
    2) 그 추세에서 벗어나는 잔차를 XGBoost로 학습
    3) 최종 예측 = 선형추세 + 잔차예측 (hybrid)
    4) 과거 구간에서 hybrid와 실제 값의 MAE 계산
    """
    regions = sorted(df_hist["region"].unique())
    min_year = int(df_hist["year"].min())
    max_year = int(df_hist["year"].max())
    all_years = np.arange(min_year, year_until + 1)

    full_rows = []
    mae_rows = []

    for region in regions:
        g = df_hist[df_hist["region"] == region].sort_values("year").copy()
        years = g["year"].values.astype(np.float32)
        y = g["emissions_per_area"].values.astype(np.float32)

        X_hist = years.reshape(-1, 1)

        # 1) 선형 회귀로 큰 추세
        lin = LinearRegression()
        lin.fit(X_hist, y)
        y_lin_hist = lin.predict(X_hist)
        resid_hist = y - y_lin_hist

        # 2) XGBoost로 잔차 학습 (데이터가 너무 적으면 생략)
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

        # 하이브리드 예측(과거 구간)
        y_hybrid_hist = y_lin_hist + resid_pred_hist

        # 3) 미래 구간 예측
        X_full = all_years.reshape(-1, 1).astype(np.float32)
        y_lin_full = lin.predict(X_full)

        if use_xgb:
            resid_full = xgb.predict(X_full)
        else:
            resid_full = np.zeros_like(y_lin_full)

        y_hybrid_full = y_lin_full + resid_full

        # MAE: 과거 구간에서 실제 vs hybrid
        mae = float(mean_absolute_error(y, y_hybrid_hist))
        mae_rows.append({"region": region, "MAE": mae})

        # full_rows 구성 (과거 + 미래)
        for yr, actual, pred in zip(years, y, y_hybrid_hist):
            full_rows.append(
                {
                    "region": region,
                    "year": int(yr),
                    "kind": "history",
                    "actual": float(actual),
                    "pred": float(pred),
                }
            )

        for yr, pred in zip(all_years, y_hybrid_full):
            # 미래 구간: actual 없음
            if yr in years:
                # 이미 위에서 history로 넣었으니 스킵
                continue
            full_rows.append(
                {
                    "region": region,
                    "year": int(yr),
                    "kind": "forecast",
                    "actual": np.nan,
                    "pred": float(pred),
                }
            )

    df_full = pd.DataFrame(full_rows)
    df_mae = pd.DataFrame(mae_rows).sort_values("MAE")

    return df_full, df_mae


# ===========================
# 3. 추세 설명 & 해결방안 텍스트
# ===========================
def describe_trend_and_solution(df_full: pd.DataFrame, region: str) -> str:
    df_r = df_full[df_full["region"] == region].copy()
    df_r = df_r.sort_values("year")

    x = df_r["year"].values
    y = df_r["pred"].values
    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    start_val = y[0]
    end_val = y[-1]

    if slope > 0:
        trend_text = f"→ {region}은(는) 2050년까지 **면적당 온실가스 배출량이 증가하는 추세**입니다."
    elif slope < 0:
        trend_text = f"→ {region}은(는) 2050년까지 **면적당 온실가스 배출량이 감소하는 추세**입니다."
    else:
        trend_text = f"→ {region}은(는) 2050년까지 **큰 변화가 없는 정체 추세**를 보입니다."

    change_ratio = (end_val - start_val) / max(start_val, 1e-6) * 100
    change_text = f"   · 초기 연도와 비교했을 때, 2050년에는 약 **{change_ratio:.1f}%** 변화가 예상됩니다."

    # 전체 forecast 값 기준으로 상·하위 구간 정의
    all_forecast = df_full[df_full["kind"] == "forecast"]["pred"].dropna()
    high_threshold = np.percentile(all_forecast, 75)
    low_threshold = np.percentile(all_forecast, 25)
    level = end_val

    if level >= high_threshold and slope > 0:
        level_text = (
            "   · 예측상 2050년에도 **전국 상위 25% 수준의 높은 배출 밀도**를 유지하고 있어, "
            "강력한 감축 정책이 필요한 지역입니다."
        )
        solution_text = (
            "- 대형 산업·발전 시설의 **에너지 효율 개선 및 연료 전환**(석탄→가스·재생에너지) 추진\n"
            "- 건물·수송 부문의 **에너지 효율 리모델링**과 전기차·수소차 보급 확대\n"
            "- 지역 분산에너지(태양광, 풍력, 바이오가스 등)와 **마이크로그리드** 구축으로 "
            "전력 자립률을 높이는 전략이 필요합니다.\n"
            "- 지자체 차원에서 **탄소중립지원센터**와 연계한 감축사업 발굴, 주민 참여형 태양광 등 "
            "지역 맞춤형 프로젝트가 중요합니다."
        )
    elif level <= low_threshold and slope < 0:
        level_text = (
            "   · 2050년에는 **전국 하위 25% 수준의 낮은 배출 밀도**를 보이며, "
            "감축이 비교적 잘 이뤄지고 있는 지역입니다."
        )
        solution_text = (
            "- 이미 진행 중인 감축정책(재생에너지 확대, 건물 효율화 등)을 유지하면서, "
            "지역 특화 산업과 연계한 **녹색 일자리 창출**에 초점을 둘 수 있습니다.\n"
            "- 농촌·어촌 지역이라면 바이오가스, 농업 폐기물 에너지화 등 "
            "**지역 자원 기반 분산에너지 모델**을 강화하는 것이 좋습니다.\n"
            "- 주민 참여 프로그램과 기후 교육·홍보를 통해 **지역 탄소중립 문화를 정착**시키는 것이 중요합니다."
        )
    else:
        level_text = (
            "   · 배출 밀도는 전국 평균~중간 수준이며, "
            "정책 방향에 따라 향후 추세가 크게 달라질 수 있는 지역입니다."
        )
        solution_text = (
            "- 건물·교통·산업 부문의 **기본적인 에너지 효율 기준 강화**와 친환경 설비 도입을 병행해야 합니다.\n"
            "- 공공건물 지붕 및 유휴부지를 활용한 **태양광·연료전지 설치** 등, "
            "공유부지 재생에너지 사업을 적극 검토할 필요가 있습니다.\n"
            "- 광역 지자체 및 중앙정부의 감축목표와 연계하여, "
            "**기초지자체 단위의 실천형 감축사업(그린리모델링, 친환경 교통 인프라 등)**을 꾸준히 확장해야 합니다."
        )

    text = "\n".join([
        trend_text,
        change_text,
        level_text,
        "",
        "📌 **정책·해결 방안 제안**",
        solution_text
    ])
    return text


# ===========================
# 4. 지도 생성 함수 (파랑~빨강, Top5 강조)
# ===========================
def create_map(df_full, df_coord, selected_year, top5_year=2050):
    """
    선택 연도 기준으로 지역별 pred 값을 지도에 표시.
    - 파랑(낮음) ~ 빨강(높음)
    - Top5 지역은 굵은 원 + ⚠️ 표시
    """
    df_year = df_full[df_full["year"] == selected_year].copy()
    if df_year.empty:
        return None

    df_year["value"] = df_year["pred"]
    df_map = pd.merge(df_year, df_coord, on="region", how="inner")

    df_map = df_map.dropna(subset=["lat", "lon", "value"])
    if df_map.empty:
        return None

    vmin = df_map["value"].min()
    vmax = df_map["value"].max()

    cmap = LinearColormap(
        colors=["#4575b4", "#ffffbf", "#d73027"],  # 파랑 → 노랑 → 빨강
        vmin=vmin,
        vmax=vmax
    )

    center_lat, center_lon = 36.5, 127.8
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=7,
                   tiles="cartodbpositron")

    # Top5 (top5_year 기준)
    df_2050 = df_full[df_full["year"] == top5_year].copy()
    df_2050["value"] = df_2050["pred"]
    top5_regions = (
        df_2050.sort_values("value", ascending=False)["region"]
        .head(5)
        .tolist()
    )

    for _, row in df_map.iterrows():
        color = cmap(row["value"])
        radius = 9
        weight = 1.5
        popup_text = (
            f"{row['region']}<br>"
            f"{selected_year}년 면적당 배출량: {row['value']:.2f} tCO₂eq/km²"
        )

        # Top5 경고 스타일
        if row["region"] in top5_regions:
            radius = 13
            weight = 3
            popup_text = "⚠️ [Top 5 배출 밀도] ⚠️<br>" + popup_text
            border_color = "black"
        else:
            border_color = color

        CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=border_color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=weight,
            popup=popup_text,
            tooltip=row["region"],
        ).add_to(m)

    cmap.caption = f"{selected_year}년 면적당 온실가스 배출량 (tCO₂eq/km²)"
    cmap.add_to(m)

    return m


# ===========================
# 5. 메인 UI
# ===========================
df_hist, df_coord = load_data()
df_full, df_mae = train_and_forecast(df_hist, year_until=2050)

min_year = int(df_full["year"].min())
max_year = int(df_full["year"].max())

tab1, tab2, tab3 = st.tabs([
    "1) 지도 & 지역별 추세",
    "2) 데이터 & 다운로드",
    "3) 예측 정확도(MAE) 평가",
])

# ---------- TAB 1: 지도 & 지역별 추세 ----------
with tab1:
    st.subheader("전국 지도에서 한눈에 보는 면적당 온실가스 배출량")

    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("### 연도 선택")
        selected_year = st.slider(
            "연도",
            min_value=min_year,
            max_value=2050,
            value=2050,
            step=1,
        )
        st.caption(
            "슬라이더를 움직이면 연도별로 색깔이 변하면서\n"
            "면적당 배출량 변화가 **애니메이션처럼** 보입니다."
        )

        # 수동 지역 선택도 가능하게
        all_regions = sorted(df_full["region"].unique())
        default_region = all_regions[0] if all_regions else None
        selected_region_manual = st.selectbox(
            "지역 직접 선택",
            all_regions,
            index=0 if default_region else None,
        )

    with col_map:
        m = create_map(df_full, df_coord, selected_year)
        if m is None:
            st.error("선택한 연도에 대한 지도 데이터를 찾을 수 없습니다.")
            map_state = {}
        else:
            st.caption("단위: tCO₂eq/km² (면적당 온실가스 배출량)")
            map_state = st_folium(m, use_container_width=True, height=600)

    # 지도 클릭/지역 선택 처리
    if "selected_region" not in st.session_state:
        st.session_state["selected_region"] = selected_region_manual

    clicked_region = None
    if "last_object_clicked_popup" in (map_state or {}):
        popup_html = map_state["last_object_clicked_popup"]
        if popup_html:
            clicked_region = popup_html.split("<br>")[0].replace("⚠️ [Top 5 배출 밀도] ⚠️", "").strip()

    if clicked_region:
        st.session_state["selected_region"] = clicked_region
    else:
        st.session_state["selected_region"] = selected_region_manual

    selected_region = st.session_state["selected_region"]

    st.markdown("---")
    st.markdown(f"### 선택된 지역: **{selected_region}**")

    df_r_full = df_full[df_full["region"] == selected_region].copy()
    df_r_full = df_r_full.sort_values("year")

    # History / Forecast 분리해서 그래프용 데이터 만들기
    df_r_plot = pd.DataFrame({
        "year": df_r_full["year"],
        "History / Forecast": np.where(
            df_r_full["kind"] == "history",
            "History",
            "Forecast"
        ),
        "value": df_r_full["pred"],
    })

    # 혹시 모를 중복 방지 위해 평균으로 한 번 더 묶기
    df_r_plot = (
        df_r_plot
        .groupby(["year", "History / Forecast"], as_index=False)["value"]
        .mean()
    )

    df_pivot = df_r_plot.pivot(
        index="year",
        columns="History / Forecast",
        values="value"
    )

    st.line_chart(df_pivot)
    st.caption("※ 실선은 선형추세 + XGBoost 잔차를 더한 **하이브리드 예측값**입니다. 과거 구간에서는 실제 값과 거의 일치합니다.")

    st.markdown("#### 추세 요약 & 정책 제안")
    text = describe_trend_and_solution(df_full, selected_region)
    st.markdown(text)


# ---------- TAB 2: 데이터 & 다운로드 ----------
with tab2:
    st.subheader("지역별 데이터 & CSV 다운로드")

    st.markdown("**① 전체 데이터 (과거 + 예측)**")
    df_export = df_full.copy().rename(columns={
        "region": "Region",
        "year": "Year",
        "actual": "Actual_Emissions_per_Area",
        "pred": "Predicted_Emissions_per_Area",
        "kind": "Type",  # history / forecast
    })

    st.dataframe(df_export.head(200), use_container_width=True)

    csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="전체 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="korea_emissions_full_hybrid_forecast.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("**② 2050년 예측값만 정리한 테이블**")

    df_2050 = df_full[df_full["year"] == 2050].copy()
    df_2050 = df_2050[["region", "pred"]].rename(columns={
        "region": "Region",
        "pred": "Emissions_per_Area_2050",
    }).sort_values("Emissions_per_Area_2050", ascending=False)

    st.dataframe(df_2050, use_container_width=True)

    csv_2050 = df_2050.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="2050년 예측 데이터만 CSV로 다운로드",
        data=csv_2050,
        file_name="korea_emissions_2050_only.csv",
        mime="text/csv",
    )


# ---------- TAB 3: 예측 정확도 (MAE) ----------
with tab3:
    st.subheader("XGBoost 하이브리드 예측 정확도 평가 (MAE 기준)")

    st.markdown(
        """
        **MAE(Mean Absolute Error, 평균 절대 오차)** 는  
        모델 예측값과 실제 값의 차이를 절대값으로 만들어 평균낸 지표입니다.

        - 값이 **0에 가까울수록** 예측이 실제 값과 거의 같다는 뜻입니다.  
        - 단위는 목표 변수와 동일합니다. (여기서는 `tCO₂eq/km²`)

        **해석 예시 (대략적인 기준)**  
        - MAE **< 5** : 매우 높은 예측 정확도  
        - **5 ≤ MAE < 15** : 보통 수준의 예측 정확도  
        - MAE **≥ 15** : 실제 값과 차이가 꽤 큰 편 → 데이터 보완 또는 모델 개선 필요
        """
    )

    st.bar_chart(df_mae.set_index("region")["MAE"])
    st.dataframe(
        df_mae.rename(columns={"region": "Region", "MAE": "MAE (tCO₂eq/km²)"}),
        use_container_width=True,
    )

    csv_mae = df_mae.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="지역별 MAE 결과 CSV 다운로드",
        data=csv_mae,
        file_name="korea_emissions_mae_by_region.csv",
        mime="text/csv",
    )
