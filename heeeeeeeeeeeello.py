import streamlit as st

# 🎈 타이틀 & 풍선 효과
st.set_page_config(page_title="MBTI 공부 공간 추천 🎧✨", page_icon="🎈")
st.title("🌟 나의 MBTI에 딱 맞는 공부 공간은? ✨")
st.balloons()

st.markdown("## 🧠 MBTI를 선택해 주세요 👇")

# 🎯 MBTI 리스트
mbti_list = [
    "INTJ 🧩", "INTP 🧪", "ENTJ 🧠", "ENTP 🔍",
    "INFJ 🌌", "INFP 🎨", "ENFJ 🎤", "ENFP 🌈",
    "ISTJ 📘", "ISFJ 🧸", "ESTJ 📊", "ESFJ 🎀",
    "ISTP 🛠️", "ISFP 🍃", "ESTP 🎯", "ESFP 🎉"
]

mbti_selection = st.selectbox("🔎 나의 MBTI는?", mbti_list)

# 🎵 백색소음 & 공간 추천 딕셔너리
recommendations = {
    "INTJ 🧩": ("📚 조용한 도서관 코너", "💻 타이핑 소리 + 빗소리 🌧️"),
    "INTP 🧪": ("🧪 실험실 같은 집중 공간", "📖 책 넘기는 소리 + 물방울 💧"),
    "ENTJ 🧠": ("🏢 모던한 코워킹 스페이스", "🚇 백색 지하철 소리 + 저음 브금"),
    "ENTP 🔍": ("☕ 창가 테이블 카페", "🔊 사람 웅성임 + 재즈 브금 🎷"),
    "INFJ 🌌": ("🌌 분위기 있는 개인 방", "🛌 잔잔한 피아노 + 바람 소리 🌬️"),
    "INFP 🎨": ("🎨 감성적인 카페 구석", "🎧 로파이 음악 + 조용한 대화"),
    "ENFJ 🎤": ("🎤 밝은 팀스터디룸", "💬 스터디 백색소음 + 커피 머신 소리 ☕"),
    "ENFP 🌈": ("🌈 감성책방 or 오픈카페", "🎵 재즈 + 새소리 + 키보드 소리 🐦"),
    "ISTJ 📘": ("📘 조용한 스터디룸", "📎 에어컨 소리 + 시계 초침"),
    "ISFJ 🧸": ("🧸 아늑한 방", "🕯️ 벽난로 소리 + 고양이 그르렁 🐱"),
    "ESTJ 📊": ("📊 조직적인 사무공간", "⌨️ 키보드 소리 + 팩스/프린터"),
    "ESFJ 🎀": ("🎀 꾸며진 홈카페", "🎶 케이팝 Lo-Fi + 커피 내리는 소리"),
    "ISTP 🛠️": ("🛠️ 창작공방 스타일", "🔧 공구소리 + 우드워크"),
    "ISFP 🍃": ("🍃 자연과 가까운 베란다", "🌲 새소리 + 잎사귀 흔들림"),
    "ESTP 🎯": ("🎯 넓은 스터디카페", "🗣️ 백색소음 + 배경 EDM 🎶"),
    "ESFP 🎉": ("🎉 활기찬 카페", "🎼 팝음악 + 사람 목소리 🍰"),
}

if mbti_selection:
    space, sound = recommendations.get(mbti_selection, ("❓ 공간 없음", "❓ 소리 없음"))
    st.markdown(f"### 🪄 어울리는 공부 공간: **{space}**")
    st.markdown(f"### 🔊 추천 백색소음: **{sound}**")

    st.success("✨ 나만의 공부 공간이 완성됐어요! 집중력 UP! 📈")
    st.balloons()

    # 이모지 대폭발!
    st.markdown("💡 공부 TIP: 좋아하는 분위기에서 집중력을 유지해보세요! 💖")
    st.markdown("🚀 파이팅! 오늘도 멋진 하루를 보내세요! 🌟")
