from konlpy.tag import Okt
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud
from collections import Counter


# OKt 객체 선언
okt= Okt()

word_cloud = ['피자 맛있었어요', '양은 많았어요', '배달은 빨랐어요','피자가 느끼해요']

tokenized_list = []
for text in word_cloud:
    tokenized_doc = okt.pos(text)
    tags = ['Noun','Adjective']

#words = [word[0] for word in tokenized_doc if word[1] in tag]
# #tokenized_nouns = ' '.join(words)
#word[0]=단어 word[1]= tag
    words = []
    for word in tokenized_doc:
        noun, tag = word

        if tag in tags:
            words.append(noun)
                
    tokenized_list.append(words) 
       

word_cloud2 =[]
for token in tokenized_list:
    for splited_t in token:
            word_cloud2.append(splited_t)


# 단어 수정 함수
def word_replace(before_word, after_word,word_list):
    temp_list = []
    for i in word_list:
            temp_list.append(i.replace(before_word,after_word))
    return temp_list

word_cloud2= word_replace("양은","양",word_cloud2)

counter = Counter(word_cloud2)
print(dict(counter))

# 이미지 파일 불러오기
flower_mask=np.array(Image.open('flower.png'))

wordcloud = WordCloud(
    font_path = 'malgun.ttf', # 한글 글씨체 설정
    background_color='white', # 배경색은 흰색으로 
    colormap='Reds', # 글씨색은 빨간색으로
    mask=flower_mask, # 워드클라우드 모양 설정
).generate_from_frequencies(dict(counter))

#사이즈 설정 및 출력
plt.figure(figsize=(7,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off') # 차트로 나오지 않게
plt.savefig('wordcloud.png')
