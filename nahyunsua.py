import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import folium
from folium import CircleMarker
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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
    hist_path = base_dir / "green_en.csv"          # 과거 데이터
    coord_path = base_dir / "XGBoostData_en.csv"   # 지역별 위도/경도

    df_hist = pd.read_csv(hist_path)
    df_coord = pd.read_csv(coord_path)

    # 컬럼 이름 맞추기 (필요하면 여기서 rename 해도 됨)
    # 예시: df_hist.rename(columns={"지역":"region", ...}, inplace=True)

    # 타입 정리
    df_hist["year"] = df_hist["year"].astype(int)
    df_hist["region"] = df_hist["region"].astype(str)

    # 좌표도 region 기준으로 정리
    df_coord["region"] = df_coord["region"].astype(str)

    return df_hist, df_coord


# ===========================
# 2. XGBoost + polynomial feature 로 예측 + MAE 계산
# ===========================
@st.cache_data
def train_and_forecast(df_hist: pd.DataFrame, year_until: int = 2050):
    """
    각 지역별로 면적당 배출량(emissions_per_area)을
    연도(year, year^2) + 인구(population)로 XGBoost 회귀 예측.
    - 마지막 3년을 테스트로 두고 MAE 계산
    - max_year+1 ~ year_until 까지 예측 생성
    """

    all_regions = sorted(df_hist["region"].unique())

    full_list = []      # 과거 + 예측 전체
    future_list = []    # 미래 예측 (max_year+1 ~ year_until)
    mae_list = []       # 지역별 MAE

    for region in all_regions:
        df_r = df_hist[df_hist["region"] == region].copy()
        df_r = df_r.sort_values("year")

        if df_r.shape[0] < 5:
            # 데이터가 너무 적으면 단순 선형 추세 사용 (polyfit)
            x = df_r["year"].values
            y = df_r["emissions_per_area"].values
            coef = np.polyfit(x, y, 1)  # 1차
            # 과거 예측
            df_r["pred"] = np.polyval(coef, x)
            df_r["kind"] = "history"

            max_year = df_r["year"].max()
            future_years = np.arange(max_year + 1, year_until + 1)
            future_pred = np.polyval(coef, future_years)

            df_future = pd.DataFrame({
                "region": region,
                "year": future_years,
                "emissions_per_area": np.nan,
                "pred": future_pred,
                "kind": "forecast"
            })

            mae = float(np.mean(np.abs(df_r["emissions_per_area"] - df_r["pred"])))
        else:
            # -------------------------
            # XGBoost 모델 학습
            # -------------------------
            df_r["year2"] = df_r["year"] ** 2  # polynomial feature

            feature_cols = ["year", "year2", "population"]
            X = df_r[feature_cols]
            y = df_r["emissions_per_area"]

            # train / test split: 마지막 3년을 test 로 사용
            last_years = sorted(df_r["year"].unique())[-3:]
            train_mask = ~df_r["year"].isin(last_years)
            test_mask = df_r["year"].isin(last_years)

            if train_mask.sum() < 3:
                # 너무 적으면 그냥 랜덤 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, shuffle=True, random_state=42
                )
            else:
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]

            model = XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
            )

            model.fit(X_train, y_train)

            # 과거 데이터에 대한 예측
            df_r["pred"] = model.predict(X[feature_cols])
            df_r["kind"] = "history"

            # MAE 계산
            y_pred_test = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_pred_test))

            # -------------------------
            # 미래 예측 (max_year+1 ~ year_until)
            # -------------------------
            max_year = df_r["year"].max()
            future_years = np.arange(max_year + 1, year_until + 1)
            future_df = pd.DataFrame({
                "region": region,
                "year": future_years
            })
            future_df["year2"] = future_df["year"] ** 2

            # 인구는 가장 최근 인구를 그대로 사용(단순 가정)
            recent_pop = df_r.sort_values("year")["population"].iloc[-1]
            future_df["population"] = recent_pop

            future_pred = model.predict(future_df[feature_cols])

            df_future = future_df.copy()
            df_future["emissions_per_area"] = np.nan
            df_future["pred"] = future_pred
            df_future["kind"] = "forecast"

        full_list.append(df_r)
        future_list.append(df_future)
        mae_list.append({
            "region": region,
            "MAE": mae
        })

    df_full = pd.concat(full_list + future_list, ignore_index=True)
    df_future_all = pd.concat(future_list, ignore_index=True)
    df_mae = pd.DataFrame(mae_list).sort_values("MAE")

    return df_full, df_future_all, df_mae


# ===========================
# 3. 추세 설명 & 해결방안 텍스트 생성
# ===========================
def describe_trend_and_solution(df_full, region):
    """선택한 지역의 전체(과거+예측) 추세를 보고 텍스트 설명 + 해결방안 생성"""
    df_r = df_full[df_full["region"] == region].copy()
    df_r = df_r.sort_values("year")

    # 실제 + 예측 모두 포함해서 단순 1차 회귀로 기울기 파악
    x = df_r["year"].values
    y = df_r["pred"].fillna(df_r["emissions_per_area"]).values
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
    change_text = f"   · 1990년대 초반과 비교했을 때, 2050년에는 약 **{change_ratio:.1f}%** 변화가 예상됩니다."

    # 배출 수준에 따른 해결방안 템플릿
    high_threshold = np.percentile(
        df_full[df_full["kind"] == "forecast"]["pred"].dropna(), 75
    )
    low_threshold = np.percentile(
        df_full[df_full["kind"] == "forecast"]["pred"].dropna(), 25
    )
    level = end_val

    if level >= high_threshold and slope > 0:
        level_text = (
            "   · 예측상 2050년에도 **전국 상위 25% 수준의 높은 배출 밀도**를 유지하고 있어, "
            "강력한 감축 정책이 필요한 지역입니다."
        )
        solution_text = (
            "- 대형 산업·발전 시설의 **에너지 효율 개선 및 연료 전환**(석탄→가스·재생에너지) 추진\n"
            "- 건물·수송 부문의 **에너지 효율 리모델링**과 전기차·수소차 보급 확대\n"
            "- 지역 분산에너지(태양광, 풍력, 바이오가스 등)와 **마이크로그리드** 구축을 통해 "
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
            "지역 특화 산업과 연계한 **녹색 일자리 창출**에 초점을 맞출 수 있습니다.\n"
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

    text = "\n".join([trend_text, change_text, level_text, "", "📌 **정책·해결 방안 제안**", solution_text])
    return text


# ===========================
# 4. 지도 생성 함수
# ===========================
def create_map(df_full, df_coord, selected_year, top5_year=2050):
    """
    선택한 연도 기준으로 지역별 pred 값을 사용해 지도에 표시.
    - 파랑(낮음) ~ 빨강(높음) 스케일
    - Top5 지역은 큰 원 + ⚠️ 표시
    """

    # 선택 연도의 데이터 (과거는 emissions_per_area, 미래는 pred 우선)
    df_year = df_full[df_full["year"] == selected_year].copy()
    df_year["value"] = df_year["pred"].fillna(df_year["emissions_per_area"])

    # 좌표와 merge
    df_map = pd.merge(df_year, df_coord, on="region", how="inner")

    if df_map.empty:
        return None

    vmin = df_map["value"].min()
    vmax = df_map["value"].max()

    cmap = LinearColormap(
        colors=["#4575b4", "#ffffbf", "#d73027"],  # 파랑 → 노랑 → 빨강
        vmin=vmin,
        vmax=vmax
    )

    # 중심 좌표(대한민국 대략 중앙)
    center_lat, center_lon = 36.5, 127.8
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="cartodbpositron")

    # Top5 (기본은 2050년 기준)
    df_2050 = df_full[df_full["year"] == top5_year].copy()
    df_2050["value"] = df_2050["pred"].fillna(df_2050["emissions_per_area"])
    top5_regions = df_2050.sort_values("value", ascending=False)["region"].head(5).tolist()

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df_map.iterrows():
        color = cmap(row["value"])
        radius = 9
        weight = 1.5
        popup_text = (
            f"{row['region']}<br>"
            f"{selected_year}년 면적당 배출량: {row['value']:.2f} tCO₂eq/km²"
        )

        # Top5 이면 경고 스타일
        if row["region"] in top5_regions:
            radius = 13
            weight = 3
            popup_text = "⚠️ [Top 5 배출 밀도] ⚠️<br>" + popup_text

        CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=weight,
            popup=popup_text,
            tooltip=row["region"]
        ).add_to(marker_cluster)

    # 색상 범례 + 단위
    cmap.caption = f"{selected_year}년 면적당 온실가스 배출량 (tCO₂eq/km²)"
    cmap.add_to(m)

    return m


# ===========================
# 5. 메인 UI - 탭 구성
# ===========================
df_hist, df_coord = load_data()
df_full, df_future, df_mae = train_and_forecast(df_hist, year_until=2050)

min_year = int(df_full["year"].min())
max_year = int(df_full["year"].max())

tab1, tab2, tab3 = st.tabs([
    "1) 지도 & 지역별 추세",
    "2) 데이터 & 다운로드",
    "3) 예측 정확도(MAE) 평가"
])

# -----------------------------------
# TAB 1: 지도 & 지역별 추세
# -----------------------------------
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
            step=1
        )
        st.markdown(
            "- 슬라이더를 움직이면 **연도별로 지도 색상(파랑→빨강)**이 바뀌면서\n"
            "  면적당 배출량 변화가 애니메이션처럼 보입니다."
        )

    with col_map:
        m = create_map(df_full, df_coord, selected_year)
        if m is None:
            st.error("선택한 연도에 대한 지도 데이터를 찾을 수 없습니다.")
        else:
            map_state = st_folium(m, use_container_width=True, height=600)

    # -------------------------------
    # 지도에서 지역 클릭 → 해당 지역 추세 + 텍스트
    # -------------------------------
    if "selected_region" not in st.session_state:
        st.session_state["selected_region"] = sorted(df_hist["region"].unique())[0]

    clicked_region = None
    if "last_object_clicked_popup" in (map_state or {}):
        popup_html = map_state["last_object_clicked_popup"]
        if popup_html:
            # popup_html 안에서 지역 이름만 추출(마크업 제거)
            clicked_region = popup_html.split("<br>")[0].replace("⚠️ [Top 5 배출 밀도] ⚠️", "").strip()

    if clicked_region:
        st.session_state["selected_region"] = clicked_region

    selected_region = st.session_state["selected_region"]

    st.markdown("---")
    st.markdown(f"### 선택된 지역: **{selected_region}**")

    df_r_full = df_full[df_full["region"] == selected_region].copy()
    df_r_full = df_r_full.sort_values("year")

    # 실제/예측 구분해서 라인 차트용 데이터 만들기
    df_r_plot = pd.DataFrame({
        "year": df_r_full["year"],
        "History / Forecast": np.where(df_r_full["kind"] == "history", "History", "Forecast"),
        "value": df_r_full["pred"].fillna(df_r_full["emissions_per_area"])
    })

    # Streamlit line_chart는 wide format을 좋아해서 pivot
    df_pivot = df_r_plot.pivot(index="year", columns="History / Forecast", values="value")

    st.line_chart(df_pivot)

    st.caption("※ 실선은 XGBoost 기반 예측값(History+Forecast)이며, 과거 구간은 실제 값과 거의 일치합니다.")

    # 추세 설명 + 해결방안 텍스트
    st.markdown("#### 추세 요약 & 정책 제안")
    text = describe_trend_and_solution(df_full, selected_region)
    st.markdown(text)


# -----------------------------------
# TAB 2: 데이터 & 다운로드
# -----------------------------------
with tab2:
    st.subheader("지역별 데이터 & CSV 다운로드")

    st.markdown("**① 전체 데이터 (과거 + 예측)**")
    df_export = df_full.copy()
    df_export = df_export.rename(columns={
        "region": "Region",
        "year": "Year",
        "emissions_per_area": "Emissions_per_Area",
        "pred": "Predicted_Emissions_per_Area",
        "kind": "Type"   # history / forecast
    })

    st.dataframe(df_export.head(200))

    csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="전체 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name="korea_emissions_full_forecast.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("**② 2050년 예측값만 정리한 테이블**")

    df_2050 = df_full[df_full["year"] == 2050].copy()
    df_2050["value"] = df_2050["pred"].fillna(df_2050["emissions_per_area"])
    df_2050 = df_2050[["region", "value"]].rename(columns={
        "region": "Region",
        "value": "Emissions_per_Area_2050"
    }).sort_values("Emissions_per_Area_2050", ascending=False)

    st.dataframe(df_2050)

    csv_2050 = df_2050.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="2050년 예측 데이터만 CSV로 다운로드",
        data=csv_2050,
        file_name="korea_emissions_2050_only.csv",
        mime="text/csv"
    )


# -----------------------------------
# TAB 3: 예측 정확도(MAE) 평가
# -----------------------------------
with tab3:
    st.subheader("XGBoost 예측 정확도 평가 (MAE 기준)")

    st.markdown(
        """
        **MAE(Mean Absolute Error, 평균 절대 오차)** 는  
        모델 예측값과 실제 값의 차이를 절대값으로 만들어 평균낸 지표입니다.

        - 값이 **0에 가까울수록** 예측이 실제 값과 거의 같다는 뜻입니다.  
        - 단위는 목표 변수와 동일합니다. (여기서는 `tCO₂eq/km²`)

        아래는 각 지역별로 최근 3년 데이터를 테스트로 사용하여 계산한 MAE입니다.
        """
    )

    # 해석용 구간 (대략적인 기준)
    st.markdown(
        """
        **해석 가이드(예시)**  
        - MAE **< 5** : 매우 높은 예측 정확도  
        - **5 ≤ MAE < 15** : 보통 수준의 예측 정확도  
        - MAE **≥ 15** : 실제 값과 차이가 꽤 큰 편 → 데이터 보완 또는 모델 개선 필요
        """
    )

    st.bar_chart(df_mae.set_index("region")["MAE"])

    st.dataframe(df_mae.rename(columns={"region": "Region", "MAE": "MAE (tCO₂eq/km²)"}))

    csv_mae = df_mae.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="지역별 MAE 결과 CSV 다운로드",
        data=csv_mae,
        file_name="korea_emissions_mae_by_region.csv",
        mime="text/csv"
    )
