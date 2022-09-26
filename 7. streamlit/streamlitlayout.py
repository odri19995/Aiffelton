import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# 맨위 사이드바 모양 수정 및 이름 설정
st.set_page_config(
    page_title="리뷰 챗봇",
    page_icon="♥",
)

#제목
st.header('This is a header')

#부제목
st.subheader('This is a subheader')

#작고 투명한 글씨
st.caption('This is a string that explains something above.')

# 표 생성
df = pd.DataFrame(
   np.random.randn(10, 5),
   columns=('col %d' % i for i in range(5)))

st.table(df)


# metric

col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")


#matplotlib 그래프

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)


#긍 부정 선택 리뷰
#[theme]
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

option = st.selectbox(
    '어떤 리뷰를 보시겠습니까?',
    ('긍정', '부정'))

st.write('You selected:', option)


col1, col2 = st.columns(2)

with col1:

   st.header("A cat")
   st.write("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("A dog")
   st.write("https://static.streamlit.io/examples/dog.jpg")

# streamlit empty

# 시간이 다 되기 까지 밑의 것은 손댈수 없다. 밑의 내용이 변경되면 다시 타이머가 돌아가는 것 같다.일종의 텀을 주는게 목적
import time

with st.empty():
    for seconds in range(60):
        st.write(f"⏳ {seconds} seconds have passed")
        #time.sleep(0.5)
    st.write("✔️ 1 minute over!")




#  multiselect box      https://docs.streamlit.io/library/api-reference/widgets/st.multiselect

options = st.multiselect(
    'What are your favorite colors',
    ['맛', '양', '배달'])

st.write('You selected:', options)


# side bar 
st.sidebar.header("모델 설명")
st.sidebar.write(""" 본 모델은 땡 \n
땡
땡""")
st.sidebar.button("초기화 버튼 ! 눌러주세요")



# https://unicode-table.com/kr/ 유니코드 사이트 원하는 아이콘 불러오기 가능
# https://htmlcolorcodes.com/  색상 변경 

color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

