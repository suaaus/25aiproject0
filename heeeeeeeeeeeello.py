import streamlit as st

st.set_page_config(page_title="MBTI 공부 공간 추천 🎧✨", page_icon="📚")
st.title("🌟 나의 MBTI에 딱 맞는 공부 공간은? ✨")

st.markdown("## 🧠 MBTI를 선택해 주세요 👇")

mbti_list = [
    "INTJ 🧩", "INTP 🧪", "ENTJ 🧠", "ENTP 🔍",
    "INFJ 🌌", "INFP 🎨", "ENFJ 🎤", "ENFP 🌈",
    "ISTJ 📘", "ISFJ 🧸", "ESTJ 📊", "ESFJ 🎀",
    "ISTP 🛠️", "ISFP 🍃", "ESTP 🎯", "ESFP 🎉"
]

mbti_selection = st.selectbox("🔎 나의 MBTI는?", mbti_list)

if st.button("✨ 결과 보러가기! ✨"):
    recommendations = {
        "INTJ 🧩": ("📚 조용한 도서관 코너", "💻 타이핑 소리 + 빗소리 🌧️",
            "🔍 분석적이고 목표 지향적인 INTJ는 방해받지 않는 깊은 사고 공간이 필수! 조용한 도서관은 최고의 집중력을 선사해요 📖💡 빗소리와 타이핑은 생각을 정리하기에 딱이에요."),
        "INTP 🧪": ("🧪 실험실 같은 집중 공간", "📖 책 넘기는 소리 + 물방울 💧",
            "🌀 끝없는 호기심과 아이디어 뱅크 INTP! 조용하고 실험실 같은 공간은 상상력 폭발💥에 완벽해요. 책 넘기는 소리와 물방울은 INTP의 몰입을 도와줘요."),
        "ENTJ 🧠": ("🏢 모던한 코워킹 스페이스", "🚇 백색 지하철 소리 + 저음 브금",
            "🔥 리더십 넘치는 ENTJ는 에너지가 흐르는 공간에서 빛나요! 코워킹 스페이스는 목표를 향한 추진력 🚀을 자극하고, 배경 저음은 집중력을 높여줘요."),
        "ENTP 🔍": ("☕ 창가 테이블 카페", "🔊 사람 웅성임 + 재즈 브금 🎷",
            "💡 아이디어가 뿜뿜한 ENTP! 활기찬 창가 카페는 자극적인 환경에서 창의력이 솟아나게 해줘요. 재즈와 사람 소리는 무한 상상의 연료예요 🎨🧠"),
        "INFJ 🌌": ("🌌 분위기 있는 개인 방", "🛌 잔잔한 피아노 + 바람 소리 🌬️",
            "🌙 깊은 내면을 가진 INFJ는 고요하고 의미 있는 공간을 사랑해요. 잔잔한 음악과 바람 소리는 명상 같은 집중을 도와준답니다 🧘‍♀️✨"),
        "INFP 🎨": ("🎨 감성적인 카페 구석", "🎧 로파이 음악 + 조용한 대화",
            "🎠 감성을 따라 떠나는 INFP는 아늑하고 따뜻한 공간에서 창의력이 폭발해요. 로파이와 잔잔한 대화 소리는 마음을 편안하게 만들어줘요 💖"),
        "ENFJ 🎤": ("🎤 밝은 팀스터디룸", "💬 스터디 백색소음 + 커피 머신 소리 ☕",
            "🤝 사람들과 함께할 때 힘을 받는 ENFJ! 팀스터디룸은 에너지를 충전하는 공간⚡ 커피소리와 대화 소리는 협업 분위기를 UP!"),
        "ENFP 🌈": ("🌈 감성책방 or 오픈카페", "🎵 재즈 + 새소리 + 키보드 소리 🐦",
            "🌟 자유로운 영혼 ENFP는 오픈된 분위기에서 상상력이 샘솟아요! 자연소리와 음악이 흐르는 공간은 에너지 넘치는 집중을 도와줘요 🎶💡"),
        "ISTJ 📘": ("📘 조용한 스터디룸", "📎 에어컨 소리 + 시계 초침",
            "📏 체계적이고 꼼꼼한 ISTJ에게는 정돈된 스터디룸이 딱! 에어컨 소리와 시계 소리는 일관성과 안정감을 줘서 집중력이 높아져요 🧩"),
        "ISFJ 🧸": ("🧸 아늑한 방", "🕯️ 벽난로 소리 + 고양이 그르렁 🐱",
            "🧡 따뜻한 마음을 가진 ISFJ는 포근한 공간에서 진가를 발휘해요. 벽난로 소리와 고양이 소리는 마음을 안정시켜주는 최고의 조합이에요 🎵🛏️"),
        "ESTJ 📊": ("📊 조직적인 사무공간", "⌨️ 키보드 소리 + 프린터 소리 🖨️",
            "📈 계획적이고 생산성 중시하는 ESTJ는 깔끔한 사무공간이 딱! 키보드와 프린터 소리는 업무 모드 ON! 상태를 만들어줘요 💼💪"),
        "ESFJ 🎀": ("🎀 꾸며진 홈카페", "🎶 케이팝 Lo-Fi + 커피 내리는 소리",
            "💕 정감 넘치는 ESFJ는 감성 넘치는 홈카페 같은 분위기에서 힘을 얻어요! 케이팝 로파이와 커피소리는 마음까지 따뜻해져요 ☕🎧"),
        "ISTP 🛠️": ("🛠️ 창작공방 스타일", "🔧 공구소리 + 우드워크",
            "🛠️ 손재주 좋은 ISTP는 만들고 탐구하는 공간에서 집중력이 최고! 공구소리와 나무소리는 창작 모드에 딱 어울려요 🎨🔨"),
        "ISFP 🍃": ("🍃 자연과 가까운 베란다", "🌲 새소리 + 잎사귀 흔들림",
            "🌿 자연과 조화로운 ISFP는 햇살과 식물이 있는 공간에서 평온함을 느껴요. 자연의 소리는 감성 충전을 도와주는 힐링 그 자체 🌞🍀"),
        "ESTP 🎯": ("🎯 넓은 스터디카페", "🗣️ 백색소음 + 배경 EDM 🎶",
            "⚡ 빠른 행동력의 ESTP는 활동적인 공간에서 집중력이 올라가요! EDM과 카페 소음은 멀티태스킹에 최적화된 환경을 만들어줘요 🎧🔥"),
        "ESFP 🎉": ("🎉 활기찬 카페", "🎼 팝음악 + 사람 목소리 🍰",
            "🎊 파티 플래너 타입의 ESFP는 생동감 있는 분위기에서 에너지가 샘솟아요! 팝음악과 사람들의 목소리가 활력 넘치는 집중 타임을 선사해요 🥳🎶"),
    }

    space, sound, reason = recommendations.get(mbti_selection, ("❓ 공간 없음", "❓ 소리 없음", "❓ 설명 없음"))
    st.markdown(
    f"""
    <style>
    .result-box {{
        border: 3px dashed #f39c12;
        border-radius: 15px;
        background-color: #fff8e1;
        padding: 25px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
        font-size: 18px;
        line-height: 1.7;
    }}
    .animated {{
        animation: pulse 1.8s infinite;
        color: #d35400;
        font-weight: bold;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    </style>

    <div class="result-box">
        <p>🪄 <span class="animated">어울리는 공부 공간</span>: <strong>{space}</strong></p>
        <p>🔊 <span class="animated">추천 백색소음</span>: <strong>{sound}</strong></p>
        <p>🤔 <span class="animated">이유</span>:<br> {reason}</p>
    </div>
    """,
    unsafe_allow_html=True
)

    st.success("✨ 나만의 공부 공간이 완성됐어요! 집중력 UP! 📈")
    st.balloons()

    st.markdown("💡 공부 TIP: 나에게 딱 맞는 분위기에서 공부하면 효율이 UP UP! 🚀")
    st.markdown("🌟 오늘도 멋진 하루 보내세요! 파이팅! 💪")
