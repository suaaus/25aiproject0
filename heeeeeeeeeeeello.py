import streamlit as st
import random
from streamlit.components.v1 import html

# MBTI별 추천 공부 공간 (이건 가상의 추천이야!)
study_space_match = {
    "INTJ": ["조용한 도서관 📚", "창밖이 보이는 커피숍 ☕", "스터디 카페 🖥️"],
    "INFP": ["책이 가득한 조용한 방 🏡", "아늑한 카페 ☕", "공원 벤치 🌳"],
    "INFJ": ["조용한 도서관 📖", "고요한 방 🛋️", "스터디 카페 🖥️"],
    "ISFJ": ["조용한 방 🏠", "책이 있는 카페 📚", "작은 스터디 카페 📖"],
    "ESFJ": ["따뜻한 카페 ☕", "조용한 방 🛋️", "친구들과 스터디 카페 🧑‍🤝‍🧑"],
    "ESTJ": ["조용한 도서관 📚", "스터디 카페 📖", "집에서 집중 가능한 방 🏡"],
    "ENFP": ["카페 ☕", "창밖이 보이는 공부 공간 🪴", "스터디 카페 🖥️"],
    "ENTP": ["스마트한 스터디 카페 📚", "창밖이 보이는 카페 ☕", "과학적인 연구소 🧪"],
    "INTP": ["조용한 도서관 📖", "자신의 방 🏡", "독립적인 카페 ☕"],
    "ENFJ": ["친구들과 스터디 카페 🧑‍🤝‍🧑", "따뜻한 카페 🍰", "조용한 도서관 📚"],
    "ENTJ": ["스터디 카페 🖥️", "집에서 집중 가능한 방 🏡", "고요한 도서관 📖"],
    "ISTJ": ["집에서 공부하는 방 🏠", "조용한 도서관 📚", "스터디 카페 📖"],
    "ISFP": ["자연 속 벤치 🌳", "조용한 카페 ☕", "자기만의 방 🏡"],
    "ESFP": ["파티룸 🎉", "친구들과 카페 ☕", "활기찬 스터디카페 🧑‍🤝‍🧑"],
    "ESTP": ["파티룸 🎉", "카페 ☕", "조용한 공원 벤치 🌳"],
    "ISTP": ["조용한 방 🏠", "자기만의 카페 ☕", "스마트한 스터디 카페 📖"]
}

# MBTI 특징 (왜 이런 공간을 추천했는지)
mbti_traits = {
    "INTJ": "INTJ는 독립적이고 논리적인 성향을 가집니다. 집중할 수 있는 조용하고 혼자서 작업할 수 있는 공간이 필요합니다.",
    "INFP": "INFP는 창의적이고 감성적인 성향을 가지고 있습니다. 차분하고 자연적인 분위기에서 창의성을 발휘합니다.",
    "INFJ": "INFJ는 내향적이고, 깊이 있는 사고를 좋아합니다. 조용한 환경에서 깊은 집중이 가능합니다.",
    "ISFJ": "ISFJ는 안정적이고, 따뜻한 환경을 선호합니다. 아늑하고 편안한 공간에서 공부하는 것을 좋아합니다.",
    "ESFJ": "ESFJ는 사교적이고, 협력적인 성향이 강합니다. 친구들과 함께 공부할 수 있는 분위기를 선호합니다.",
    "ESTJ": "ESTJ는 체계적이고 조직적인 성향을 가집니다. 깔끔하고 집중할 수 있는 환경을 선호합니다.",
    "ENFP": "ENFP는 활발하고 창의적입니다. 외부 자극이 있는 활기찬 환경에서 더 잘 공부합니다.",
    "ENTP": "ENTP는 지적이고 호기심이 많습니다. 다양한 자극이 있는 공간에서 창의력을 발휘할 수 있습니다.",
    "INTP": "INTP는 독립적이고 논리적입니다. 혼자서 깊은 사고를 할 수 있는 조용한 환경이 필요합니다.",
    "ENFJ": "ENFJ는 사람들과의 교류를 즐깁니다. 다른 사람들과 함께 공부할 수 있는 따뜻한 환경을 선호합니다.",
    "ENTJ": "ENTJ는 목표 지향적이고 효율적인 성향입니다. 집중할 수 있는 공간에서 체계적으로 공부하는 것을 선호합니다.",
    "ISTJ": "ISTJ는 신뢰성과 안정성을 중시합니다. 조용하고 규칙적인 환경에서 집중력을 발휘할 수 있습니다.",
    "ISFP": "ISFP는 감성적이고 자연적인 분위기를 좋아합니다. 자연과 가까운 곳에서 공부하는 것이 이상적입니다.",
    "ESFP": "ESFP는 활발하고 외향적인 성향을 가집니다. 사람들과 함께하는 활동적인 공간에서 공부가 잘 됩니다.",
    "ESTP": "ESTP는 도전적이고 외향적입니다. 자극적인 환경에서 에너지를 얻으며 공부할 수 있습니다.",
    "ISTP": "ISTP는 실용적이고 분석적인 성향을 가집니다. 혼자서 자유롭게 작업할 수 있는 환경에서 집중할 수 있습니다."
}

# 백색 소음 추천 (MBTI별 맞춤형)
white_noise = {
    "INTJ": "🔇 조용한 백색 소음 (배경 소음이 없는 조용한 환경)",
    "INFP": "🌳 자연 소리 (바람 소리, 새 소리 등 자연의 소리)",
    "INFJ": "🔕 고요한 백색 소음 (배경 소음이 없는 고요한 소리)",
    "ISFJ": "☕ 카페 소리 (부드러운 커피 머신 소리와 대화 소리)",
    "ESFJ": "🎶 부드러운 배경 음악 (편안한 음악과 함께)",
    "ESTJ": "🔇 집중을 위한 백색 소음 (책상 위에서 흐르는 바람 소리)",
    "ENFP": "🎵 활기찬 음악 (리듬감 있는 음악, 약간의 백색 소음)",
    "ENTP": "🧠 창의적인 소리 (논리적 사고를 돕는 음악과 백색 소음)",
    "INTP": "🔇 조용한 백색 소음 (배경 소음이 없는 고요한 환경)",
    "ENFJ": "🎧 편안한 음악 (조용하고 따뜻한 음악 소리)",
    "ENTJ": "🔕 집중을 돕는 백색 소음 (규칙적인 소리)",
    "ISTJ": "🔇 고요한 환경 (배경 소음이 없는 조용한 공간)",
    "ISFP": "🌱 자연 소리 (바람, 새, 물소리 등 자연의 소리)",
    "ESFP": "🎉 파티 소리 (즐거운 파티 소리와 활기찬 음악)",
    "ESTP": "🎶 역동적인 음악 (빠른 템포의 음악과 함께)",
    "ISTP": "🔇 조용한 백색 소음 (자유로운 공간에서의 소리)"
}

# 웹앱 제목과 인사말
st.title("MBTI별 공부 공간 & 백색 소음 추천 📚🎧✨")
st.markdown("""
    MBTI를 입력하면, 당신에게 어울리는 공부 공간과 백색 소음을 추천해 드려요! 🌟
    어떤 공부 환경이 잘 맞는지 알아보세요! 🎉
""")

# MBTI 유형 선택
mbti = st.selectbox(
    "당신의 MBTI를 선택해주세요! 😎",
    ["INTJ", "INFP", "INFJ", "ISFJ", "ESFJ", "ESTJ", "ENFP", "ENTP", 
     "INTP", "ENFJ", "ENTJ", "ISTJ", "ISFP", "ESFP", "ESTP", "ISTP"]
)

# 추천된 공부 공간과 백색 소음, 특징 출력
st.markdown("### 📍 당신에게 어울리는 공부 공간은?")
matches = study_space_match.get(mbti, [])
noise = white_noise.get(mbti, "")
traits = mbti_traits.get(mbti, "")

if matches:
    st.write(f"✨ {mbti}에게 어울리는 공부 공간은 바로! ✨")
    for match in matches:
        st.write(f"➡️ **{match}** 😍")

    st.write(f"📢 추천하는 백색 소음: **{noise}**")
    st.write(f"🧠 MBTI 특징: {traits}")

# 풍선 애니메이션 효과 추가
html_code = """
    <div style="position:relative; z-index:999;">
        <script src="https://cdn.jsdelivr.net/npm/balloon-css@1.0.0/dist/balloon.min.js"></script>
        <button data-balloon="너에게 맞는 공부 환경 찾았어요! 💡" data-balloon-pos="up" style="font-size:20px; padding:10px 20px; background-color:#FF6F61; color:white; border:none; border-radius:10px; cursor:pointer;">
            나만의 공부 공간 찾기! 🎈
        </button>
    </div>
"""
html(html_code)

# 배경 색상 효과 (랜덤으로 색상 변경)
bg_colors = ['#FFD700', '#FF6F61', '#FFB6C1', '#90EE90', '#ADD8E6']
random_color = random.choice(bg_colors)
st.markdown(f"""
    <style>
        body {{
            background-color: {random_color};
        }}
    </style>
""", unsafe_allow_html=True)

# 추가적인 재미 요소로 유저와의 상호작용을 강화
if st.button("🎉 결과 다시 보기! 🥳"):
    st.experimental_rerun()
