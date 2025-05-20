import streamlit as st
import random
from streamlit.components.v1 import html

# MBTI 유형별 추천할 어울리는 MBTI들 (가상의 추천)
mbti_match = {
    "INTJ": ["ENFP", "ENTP", "INFJ", "INTP"],
    "INFP": ["ENFJ", "ENTP", "ISFJ", "INTP"],
    "INFJ": ["ENFP", "INFP", "INTP", "ENTJ"],
    "ISFJ": ["ESFP", "ENFJ", "ISTJ", "INFP"],
    "ESFJ": ["INFP", "ISTJ", "ISFJ", "ENFP"],
    "ESTJ": ["ISFP", "ISTJ", "ENFJ", "INTP"],
    "ENFP": ["INTJ", "INFJ", "ENFJ", "INFP"],
    "ENTP": ["INFJ", "ENFP", "INTP", "ENTJ"],
    "INTP": ["ENTP", "INFP", "INTJ", "ISFP"],
    "ENFJ": ["ISFJ", "ENFP", "INFJ", "ENTP"],
    "ENTJ": ["INTJ", "ENFP", "INFJ", "ENFJ"],
    "ISTJ": ["ISFJ", "ESTJ", "ISFP", "INTP"],
    "ISFP": ["INFP", "ESFP", "ISTP", "INTP"],
    "ESFP": ["ISFP", "ENFP", "ESTJ", "ISFJ"],
    "ESTP": ["ISFP", "INTP", "ESFP", "INFP"],
    "ISTP": ["ISFP", "INTP", "ESFP", "ESTJ"]
}

# 웹앱 제목과 인사말
st.title("MBTI와 잘 어울리는 사람 찾기 🧑‍🤝‍🧑✨")
st.markdown("""
    MBTI를 입력하면, 어떤 MBTI와 잘 어울릴지 알려드릴게요! 🥳
    좋아하는 MBTI를 선택해 주세요! 🎉
""")

# MBTI 유형 선택
mbti = st.selectbox(
    "당신의 MBTI를 선택해주세요! 😎",
    ["INTJ", "INFP", "INFJ", "ISFJ", "ESFJ", "ESTJ", "ENFP", "ENTP", 
     "INTP", "ENFJ", "ENTJ", "ISTJ", "ISFP", "ESFP", "ESTP", "ISTP"]
)

# 선택된 MBTI를 바탕으로 잘 어울리는 MBTI 추천
st.markdown("### 👯‍♂️ 추천되는 MBTI: ")

matches = mbti_match.get(mbti, [])

# 추천된 MBTI가 없을 경우
if matches:
    st.write(f"✨ {mbti}와 잘 어울리는 MBTI는 바로! ✨")
    for match in matches:
        st.write(f"➡️ **{match}** 😍")
else:
    st.write("🤷‍♀️ 아쉽게도 추천할 수 없어요! 😅")

# 풍선 애니메이션 효과 추가
html_code = """
    <div style="position:relative; z-index:999;">
        <script src="https://cdn.jsdelivr.net/npm/balloon-css@1.0.0/dist/balloon.min.js"></script>
        <button data-balloon="와우! 엄청 잘 어울려요! 💫" data-balloon-pos="up" style="font-size:20px; padding:10px 20px; background-color:#FF6F61; color:white; border:none; border-radius:10px; cursor:pointer;">
            추천된 MBTI 확인하기! 🎈
        </button>
    </div>
"""
html(html_code)

# 추가적인 재미 요소로 유저와의 상호작용을 강화
if st.button("🎉 결과 다시 보기! 🥳"):
    st.experimental_rerun()
