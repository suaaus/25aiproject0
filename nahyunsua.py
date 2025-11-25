import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(layout="wide")
st.title("2050년까지 지역별 면적당 온실가스 배출량 예측 및 대응 대시보드")

# 데이터 불러오기
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "XGBoost_예측결과_요약.csv"
    st.write("CSV 경로:", csv_path)
    return pd.read_csv(csv_path)
# 지도 시각화
st.subheader("2050년 면적당 온실가스 배출량 (단위: 톤/㎢)")
map_center = [36.5, 127.8]
map_zoom = 6.5
m = folium.Map(location=map_center, zoom_start=map_zoom)

# Top5 지역 선정
top5 = data.sort_values("2050_면적당배출량", ascending=False).head(5)

for _, row in data.iterrows():
    tooltip = f"{row['지역']}\n2050년: {row['2050_면적당배출량']:.2f} 톤/㎢"
    folium.CircleMarker(
        location=[row["위도"], row["경도"]],
        radius=8,
        color="red" if row["지역"] in top5["지역"].values else "blue",
        fill=True,
        fill_opacity=0.7,
        tooltip=tooltip
    ).add_to(m)

st_folium(m, width=1000)

st.divider()

# 배출량 증가 이유와 대응
st.subheader("지역별 배출량 증가 요인 및 대응 전략")

explanations = {
    "서울특별시": {"이유": "고밀도 도시 구조 및 교통 중심의 에너지 소비 집중", "대응": "제로에너지건축 확대, 도시철도 중심 교통 전환"},
    "부산광역시": {"이유": "산업 및 항만물류 집중", "대응": "산업단지 에너지효율화, 저탄소 항만 조성"},
    "세종특별자치시": {"이유": "도시 개발 및 인구 유입 증가", "대응": "저탄소 도시계획, 공공건물 제로에너지화"},
    "울산광역시": {"이유": "에너지다소비 산업 밀집", "대응": "공업단지 탈탄소화, 수소 기반 산업 전환"},
    "제주특별자치도": {"이유": "관광수요 증가로 인한 교통 부문 배출 증가", "대응": "전기차, 수소버스 확대, 신재생에너지 확대"},
    "경기도": {"이유": "수도권 인구 집중으로 인한 교통 및 주거 수요 증가", "대응": "광역교통망 전환, 에너지 고효율 도시계획"},
    "인천광역시": {"이유": "항만 및 배후공업단지의 에너지 소비 지속", "대응": "배후단지 탄소관리, 친환경 항만 도입"},
    "강원특별자치도": {"이유": "면적 대비 저밀도이지만 인구 감소로 총량 감소 어려움", "대응": "공공부문 고효율화, 산림흡수원 강화"},
    "충청북도": {"이유": "내륙산업도시의 에너지 소비 지속", "대응": "신재생에너지 보급, 산업단지 에너지효율화"},
    "충청남도": {"이유": "석탄화력발전소 및 중공업 밀집", "대응": "석탄발전 감축, 수소 발전소 전환"},
    "전라북도": {"이유": "농축산 중심 지역으로 분산된 배출", "대응": "스마트 축산, 농업기계 전동화"},
    "전라남도": {"이유": "산단 및 정유시설 분포", "대응": "정유·화학산업 고효율화, 해상풍력 확산"},
    "광주광역시": {"이유": "도시 기반시설 노후화", "대응": "건물 리모델링 통한 에너지 효율 향상"},
    "대전광역시": {"이유": "연구단지·행정도시로 인한 고정 에너지 수요", "대응": "공공기관 에너지 절감 프로그램 강화"},
    "대구광역시": {"이유": "내륙 도시로 교통 중심 배출 구조", "대응": "전기 대중교통 확대, 도심 재개발 저탄소화"},
    "경상북도": {"이유": "농촌지역과 산업도시 혼재로 구조 복잡", "대응": "도시별 맞춤형 에너지 전환 전략"},
    "경상남도": {"이유": "제조업 중심 도시 다수 분포", "대응": "스마트팩토리 확대, 에너지 절감 설비 지원"}
}

for region, details in explanations.items():
    with st.expander(f"{region}"):
        st.markdown(f"**배출 증가 요인:** {details['이유']}")
        st.markdown(f"**권장 대응 방안:** {details['대응']}")

st.divider()

# 지역별 예측 그래프 이미지 보기
st.subheader("지역별 배출량 예측 그래프")

graph_dir = "xgb_predictions"
if os.path.exists(graph_dir):
    image_files = [f for f in os.listdir(graph_dir) if f.endswith(".png")]
    image_files.sort()
    for img_file in image_files:
        st.image(os.path.join(graph_dir, img_file), caption=img_file.replace("_예측.png", ""), use_column_width=True)
else:
    st.warning("예측 그래프 이미지 폴더(xgb_predictions)를 찾을 수 없습니다. 파일을 해당 경로에 넣어주세요.")

st.info("배출량 분석과 대응 전략은 F2025-50 국가장기전략 보고서 (2050 탄소중립 시나리오) 기반으로 구성되었습니다.")
