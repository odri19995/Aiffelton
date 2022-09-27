#####################################################################################################################
#                                                     모델 코드
#####################################################################################################################
from logging import PlaceHolder
import streamlit as st
from streamlit.proto.RootContainer_pb2 import SIDEBAR
from streamlit_chat import message
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import time
import matplotlib.pyplot as plt
from soynlp.normalizer import *
import re 
from konlpy.tag import Kkma
from kiwipiepy import Kiwi
import altair as alt
import pandas as pd
import matplotlib.font_manager as fm
from konlpy.tag import Okt
from PIL import Image
from wordcloud import WordCloud
from collections import Counter
st.set_page_config(layout="wide")


import os
os.environ["KMP_DUPLICATE_LIB_OK"]= 'True'
#####################################################################################################################
TAPT_path = "./TAPT_Model_Save"
element_path = "./element_klue_model_dict_(26.46)0.913.pt"
emotional_path = "./emotional_klue_model_dict_0920.pt"
style1_path = "./style.css"
style2_path = "./style2.css"
# Pretrain 모델 불러오기
#####################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(TAPT_path)
bertmodel = AutoModel.from_pretrained(TAPT_path)
max_len = 104
#####################################################################################################################

# Element Model Cumstom Dataset
#####################################################################################################################
class BERTDataset(Dataset): 
  def __init__(self, dataset, bert_tokenizer, max_len):
    
    self.sentences = [bert_tokenizer(i, truncation=True, return_token_type_ids=True, padding="max_length",
                                     max_length = max_len, return_tensors = 'pt') for i in dataset['review']]
    self.max_len = max_len
    self.taste = [np.int64(i) for i in dataset['taste']]
    self.quantity = [np.int64(i) for i in dataset['quantity']]
    self.delivery = [np.int64(i) for i in dataset['delivery']]
  
  def __getitem__(self,i):
    self.input_ids = self.sentences[i]['input_ids']
    self.attention_mask_token = self.sentences[i]['attention_mask']
    self.token_type_ids = self.sentences[i]['token_type_ids']
    
    return self.input_ids, self.attention_mask_token, self.token_type_ids, self.taste[i], self.quantity[i], self.delivery[i]

  def __len__(self):
    return (len(self.taste))
#####################################################################################################################

# Element Cumstom Model
#####################################################################################################################
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=1,    # 클래스 수 조정
                 dr_rate=0.3,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.taste = nn.Linear(hidden_size , num_classes)
        self.quantity = nn.Linear(hidden_size , num_classes)
        self.delivery = nn.Linear(hidden_size , num_classes)
        
        self.dropout = nn.Dropout(p=dr_rate)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        out = self.bert(input_ids = input_ids.long(), token_type_ids = token_type_ids.long(), attention_mask = attention_mask.to(device))
        out = self.dropout(out.pooler_output)
        
        taste_out = self.taste(out)
        quantity_out = self.quantity(out)
        delivery_out = self.delivery(out)   
        
        return self.sigmoid(taste_out), self.sigmoid(quantity_out), self.sigmoid(delivery_out)
#####################################################################################################################
    
model = BERTClassifier(bertmodel).to(device)
# 모델 가중치 불러오기 
model.load_state_dict(torch.load(element_path, map_location=device))

# 결과값 예측값 함수
#####################################################################################################################
def predict(predict_sentence):
    
    dataset_another = pd.DataFrame({'review': predict_sentence, 'taste' : 0 , 'quantity' : 0, 'delivery' : 0}, index = [0])

    another_test = BERTDataset(dataset_another,tokenizer, max_len)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=1, num_workers=0)

    model.eval()
    
    for batch_id, (input_ids, attention_mask_token, token_type_ids, taste_label, quantity_label, delivery_label) in enumerate(test_dataloader):
            
        input_ids = input_ids.long().to(device)
        input_ids = input_ids.squeeze(1)

        token_type_ids = token_type_ids.long().to(device)
        token_type_ids = token_type_ids.squeeze(1)
        
        attention_mask_token= attention_mask_token.squeeze(1)
        
        taste_label = taste_label.long().to(device)
        quantity_label = quantity_label.long().to(device)
        delivery_label = delivery_label.long().to(device)
        
        taste_out, quantity_out, delivery_out = model(input_ids, attention_mask_token, token_type_ids)
        
        for taste, quantity, delivery in zip(taste_out, quantity_out, delivery_out):
            
            logits_taste = taste.detach().cpu().tolist()
            logits_quantity = quantity.detach().cpu().tolist()
            logits_delivery = delivery.detach().cpu().tolist()
            
    return [logits_taste[0], logits_quantity[0], logits_delivery[0]]
#####################################################################################################################
#                                                   리뷰 긍정 분석
#####################################################################################################################

emotional_max_len = 250
emotional_tokenizer = AutoTokenizer.from_pretrained(TAPT_path)
emotional_bertmodel = AutoModel.from_pretrained(TAPT_path)

# Emotional Model Custim Dataset
#####################################################################################################################
class emotional_Dataset(Dataset): 
  def __init__(self, dataset, bert_tokenizer, max_len):
    
    self.sentences = [bert_tokenizer(i, truncation=True, return_token_type_ids=True, padding="max_length",
                                     max_length = max_len, return_tensors = 'pt') for i in dataset['content']]
    self.max_len = max_len
    self.label = [np.int64(i) for i in dataset['label']]
  
  def __getitem__(self,i):
    self.input_ids = self.sentences[i]['input_ids']
    self.attention_mask_token = self.sentences[i]['attention_mask']
    self.token_type_ids = self.sentences[i]['token_type_ids']

    return self.input_ids, self.attention_mask_token, self.token_type_ids, self.label[i]

  def __len__(self):
    return len(self.label)

# Emotional Custim Model
#####################################################################################################################
class emotional_custom_model(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 ################# 클래스 수 조정 ##################
                 num_classes=2,   
                 dr_rate=None,
                 params=None):
        super(emotional_custom_model, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.dense = nn.Linear(hidden_size , num_classes)

        self.dropout = nn.Dropout(p=dr_rate)
    
    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids = input_ids.long(), token_type_ids = token_type_ids.long(), attention_mask = attention_mask.to(device))
        out = self.dropout(out.pooler_output)

        return self.dense(out)
# 모델 불러오기
#####################################################################################################################
emotional_model = emotional_custom_model(emotional_bertmodel,dr_rate=0.3).to(device)
emotional_model.load_state_dict(torch.load(emotional_path, map_location=device))

# 결과값 예측값 함수
#####################################################################################################################
def softmax(x):
  exp_x = np.exp(x)
  result = exp_x / np.sum(exp_x)

  return result

def emotional_predict(predict_sentence):

    dataset_another = pd.DataFrame({'content': predict_sentence, 'label' : 0 }, index = [0])

    another_test = emotional_Dataset(dataset_another,emotional_tokenizer, emotional_max_len)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=1, num_workers=0)
    
    emotional_model.eval()

    for input_ids, attention_mask_token, token_type_ids, label in test_dataloader:

          input_ids = input_ids.long().to(device)
          input_ids = input_ids.squeeze(1)

          token_type_ids = token_type_ids.long().to(device)
          token_type_ids = token_type_ids.squeeze(1)
          
          attention_mask_token= attention_mask_token.squeeze(1)

          out = emotional_model(input_ids, attention_mask_token, token_type_ids)
          negative, positive = softmax(out[0].detach().cpu().numpy())
    
    return negative, positive

#####################################################################################################################                    
####################################################  리뷰 코드 #######################################################
#####################################################################################################################
st.title('Review Chatbot')
####################################################### 함수 #########################################################
# 문장 분리 코드 
class TextSpliter:
    def __init__(self):
        self.kkm = Kkma()
        
    def split_review(self, text):
        word_morp = self.kkm.pos(text)
        split_idx = []
        idx = 0
        find_range = 0
        for word, morp in word_morp:
            try:
                while text[idx] == ' ':
                    idx += 1
            except:
                idx -= 1


            find_idx = text.find(word, idx, idx+len(word)+find_range)
            if find_idx > -1:
                idx = find_idx + len(word)
                find_range = 0
            else:
                re_word = re.sub('[ㄱ-ㅎ]+', '', word)
                find_idx = text.find(re_word, idx, idx+len(re_word)+2)
                if find_idx > -1:
                    idx = find_idx + len(word)
                    find_range = 0
                else:
                    find_range += len(word)

            if morp in ('ECE', 'EFN', 'ETN', 'EMO'):
                split_idx.append(idx)
        return split_idx
    
    def get_splited_review(self, text):

        splited_reviews = []
        split_idx = self.split_review(text)

        last_idx = 0
        for idx in split_idx:
            splited_reivew = text[last_idx:idx].strip()
            if splited_reivew:
                splited_reviews.append(text[last_idx:idx].strip())
                last_idx = idx
        splited_reivew = text[last_idx:]
        if splited_reivew:
            splited_reviews.append(text[last_idx:].strip())

        return splited_reviews
    
# 소비자가 작성한 리뷰 전처리
def extract_word(text):
  text = text.lower() # 소문자 변환
  first = emoticon_normalize(text, num_repeats = 2) # 이모티콘
  first = repeat_normalize(first, num_repeats = 2) # ㅋㅋㅎㅎ 같은 반복 반복문장 2개만 남기고 

  convert = re.compile('^\d*\d$|[^가-힣a-zA-Z0-9.ㅋㅜㅎㅠ?!:]') # 해당 문자만 남기고 나머지 제거
  result = convert.sub(' ',first) # 변환

  result = re.sub(r'[" "]+', " ",result) # 공백 여러개를 한개로 변환
  result = result.strip() # 양쪽 공백 제거

  return result

# 요소별 응답 리스트
def vodcap(ele_list):
    if ele_list == [0]:
        tt = '음식의 맛은 어떠셨나요?'
    elif ele_list == [1]:
        tt = '음식의 양 혹은 서비스는 어떠셨나요?'
    elif ele_list == [2]:
        tt = '배달 서비스는 어떠셨나요?'
    elif ele_list == [0,1]:
        tt = '음식의 맛과 양(서비스)는 어떠셨나요?' 
    elif ele_list == [0,2]:
        tt = '음식의 맛과 배달은 어떠셨나요?'
    elif ele_list == [1,2]:
        tt = '음식의 양 또는 서비스, 배달은 어떠셨나요?'
    elif ele_list == [ ]:
        tt = '리뷰를 작성해 주셔서 감사합니다!'    
    else :
        tt = '리뷰를 다시 입력해 주세요'        
    return tt

# 감정분석 후 session_state에 넣기 
def emotional_check(text):
    negative, positive = emotional_predict(text)
    if positive >= 0.6:
        st.session_state.positive_space.append(text)
    elif negative >= 0.6:
        st.session_state.negative_space.append(text)
    return negative, positive

# 긍부정 도넛 차트
def pie_chart(emotional, head, chart):
    while True :
        for i in range(len(emotional)):
            pie_value = [emotional.loc[i]['negative_score'],emotional.loc[i]['positive_score']]
            colors = ['#fbc78f','#9cdbad']
            labels = ['negative','positive']
            fig1, ax1 = plt.subplots(facecolor = '#f9f9f9')

            ax1.pie(pie_value, autopct='%.2f%%',
                    startangle=90, colors = colors, textprops = {'fontsize':15, 'fontweight': 'bold'} ,
                    wedgeprops=dict(width=0.69))
            ax1.axis('equal')
            ax1.legend(labels, loc = 'center', fontsize = 11)
            with head.container():
                st.markdown(""" <style> @import url('https://fonts.googleapis.com/css2?family=Do+Hyeon&display=swap');
                                .font {
                                font-size:28px ; font-family: 'Do Hyeon', sans-serif ; color: #31333F ;
                                text-align : center ; font-weight : 30 ; margin: auto;} 
                                </style> """, unsafe_allow_html=True)
                st.markdown(f'<p class="font">{emotional.loc[i]["review"]}</p>', unsafe_allow_html=True)
            with chart.container():
                st.pyplot(fig1)
            time.sleep(4)

# 3요소 판별 
def is_in_ele(ele_list, text,sidebar2):
    chatbot_output1 = predict(extract_word(text)) # 문장 전처리 후 3요소 예측 진행
    # 바차트 그래프 그리기 
    element_bar_chart(chatbot_output1, text, sidebar2)
    ok_elelist = [ele for ele in ele_list if chatbot_output1[ele]>=0.5] # 기준값 보다 높으면 ok_list에 생성
    for ele in ok_elelist:
        ele_list.remove(ele) # ele_list에 있는 3요소를 ok_list와 비교해서 제거 
    return ele_list





# 3요소 바차트
def element_bar_chart(chatbot_output1, text, sidebar2):
    ############################ 한글 설정 ############################
    BMJUA = fm.FontProperties(fname='C:/work/python/BMfont.ttf')
    #################################################################
    
    bar_value = [chatbot_output1[0],chatbot_output1[1],chatbot_output1[2]]
    labels = ['맛','양 or 서비스','배달']
    fig, ax = plt.subplots()
    ax.bar(x = labels, height = bar_value, color = ['r','g','b'],
            align = 'center', edgecolor = 'lightgray', linewidth = 3)
    
    ############## 그래프에 숫자 넣기 ##############
    for i, v in enumerate(labels):
        ax.text(v, bar_value[i], round(bar_value[i],3), 
                fontsize = 14, 
                color='black',
                horizontalalignment='center',  
                verticalalignment='bottom')    
    ############################################
    parameters = {'axes.titlesize': 20}
    plt.rcParams.update(parameters)
    ax.set_title(text,fontproperties=BMJUA)
    ax.set_xticklabels(labels = labels, fontsize = 20, fontproperties=BMJUA)
    ax.set_yticks([0,0.5,1.2])
    with sidebar2.container():
        st.pyplot(fig)
    
# css 디자인             
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
#####################################################################################################################

class chat():
    
    def __init__(self, emotional, text_list):
        message("안녕하세요 고객님! 맛있는 식사 하셨나요??")
        # time.sleep(1)
        message("맛있게 식사하셨다면 리뷰를 작성해 주세요!!")
        # time.sleep(1)
        
        ############### 문장분리 패키지 선택 ###############
        self.spliter = TextSpliter() 
        self.kiwi = Kiwi() 
        self.emotional = emotional
        self.text_list = text_list
        
    def chatting(self):
        local_css(style2_path)
        placeholder = st.empty()
        placeholder2 = st.empty()
        timer_sidebar = st.sidebar.empty()
        sidebar2 = st.sidebar.empty()
        col1,col2 = st.columns(2)
        
        with timer_sidebar.container():
            sidebar = st.sidebar.empty()
            with sidebar.container():
                st.write('설명부분')
                
        with placeholder2.container():
            with st.form('전송', clear_on_submit = True):
                self.user_input = st.text_input('리뷰를 작성해 주세요: ', "")
                self.submitted = st.form_submit_button('전송')
                    
        if self.submitted and self.user_input:
            self.user_input1 = self.user_input.strip() # 채팅 입력값 및 여백제거
            st.session_state.past.append(self.user_input1) # 입력값을 past 에 append -> 채팅 로그값 저장
            
        ele_list = [0,1,2] # 맛 : 0 / 양(서비스) : 1 / 배달 : 2
        
        with placeholder.container(): # 리뷰 버튼을 밑으로 내리기 위해 필요 

            for i in range(len(st.session_state['past'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user') # 작성된 리뷰 session_state로 누적
                bot_text = st.session_state['past'][i] # 작성된 리뷰 화면 표시
                must_say = is_in_ele(text=bot_text, ele_list=ele_list, sidebar2 = sidebar2) # 부족한 요소 확인
                if len(ele_list)!=0 :
                    bot_sentence = vodcap(must_say)
                    message(bot_sentence, key=str(i) + '_bot')
                
                else :
                    message('리뷰를 작성해 주셔서 감사합니다', key=str(i) + '_bot')
                    local_css(style1_path)
                    placeholder2.empty()
                    timer_sidebar.empty()
                    for i in range (len(st.session_state['past'])):
                        ###################### kkma package 문장분리 ######################
                        split_text = self.spliter.get_splited_review(extract_word(st.session_state['past'][i]))# 문장분리 
                        
                        ###################### kiwi package 문장분리 ######################
                        # split_text = self.kiwi.split_into_sents(extract_word(st.session_state['past'][i]))
                        
                        for i in split_text:
                            st.session_state.word_cloud.append(i)
                            nega_score, posi_score = emotional_check(i)
                            second = pd.DataFrame({'review':i , 'negative_score' : f'{nega_score:.4f}', 'positive_score' : posi_score}, index = [0])
                            self.emotional = pd.concat([self.emotional, second], ignore_index= True)


                    with col1.container():
                        tab1,tab2,tab3 = st.tabs(['긍정','부정','워드'])
                        with tab1.container():
                            for i in st.session_state['positive_space']:
                                st.write(i)
                        with tab2.container():
                            for i in st.session_state['negative_space']:
                                st.write(i)
                        
                        with tab3.container():
                            
                            ############################################## 워드 클라우드 ###############################################
                            okt= Okt() # OKt 객체 선언

                            tokenized_list = []
                            for text in st.session_state['word_cloud']:
                                tokenized_doc = okt.pos(text)
                                tags = ['Noun','Adjective']

                                words = []
                                for word in tokenized_doc:
                                    noun, tag = word

                                    if tag in tags:
                                        words.append(noun)
                                            
                                tokenized_list.append(words) 

                            word_cloud2 =['배달 빠름','건강한 맛','위생','친절','재주문']
                            for token in tokenized_list:
                                for splited_t in token:
                                        word_cloud2.append(splited_t)

                            # 단어 수정 함수
                            def word_replace(before_word, after_word, word_list):
                                temp_list = []
                                for i in word_list:
                                        temp_list.append(i.replace(before_word,after_word))
                                return temp_list

                            word_cloud2= word_replace("양은","양",word_cloud2)

                            counter = Counter(word_cloud2)
                            print(dict(counter))

                            # 이미지 파일 불러오기
                            flower_mask=np.array(Image.open('./flower.png'))

                            wordcloud = WordCloud(
                                font_path = 'malgun.ttf', # 한글 글씨체 설정
                                background_color='#f9f9f9', # 배경색은 흰색으로 
                                colormap='Reds', # 글씨색은 빨간색으로
                                mask=flower_mask, # 워드클라우드 모양 설정
                            ).generate_from_frequencies(dict(counter))

                            #사이즈 설정 및 출력
                            save_path = './wordcloud.png'
                            plt.figure(figsize=(3,3))
                            plt.imshow(wordcloud,interpolation='bilinear')
                            plt.axis('off') # 차트로 나오지 않게
                            plt.savefig(save_path)
                            st.image(save_path)
                            #################################################################################################################################
                    with col2.container():
                        st.write('긍부정 비율')
                        head = st.empty()
                        chart = st.empty()
                        pie_chart(self.emotional, head, chart)
                    st.stop()
                     
            ###################################### 카운트 다운 ######################################
            with st.empty():
                if self.submitted == True and (len(ele_list) != 0):
                    ts = int(30)
                    while (ts != 0):
                        with timer_sidebar.container():
                            st.header(f"{ts}")
                        ts -= 1
                        if self.user_input == True:
                            break
                        else: 
                            if (ts == 0) or (len(ele_list) == 0):
                                message('리뷰를 분석 중입니다.', key = str(st.session_state['generated']) +'_bot')
                                local_css(style1_path)
                                placeholder2.empty()
                                timer_sidebar.empty()
                                for i in range (len(st.session_state['past'])):
                                    # st.session_state.trash.append(i)
                                    split_text = self.spliter.get_splited_review(st.session_state['past'][i]) # 문장분리 
                                    for i in split_text:
                                        nega_score, posi_score = emotional_check(i)
                                        second = pd.DataFrame({'review':i , 'negative_score' : nega_score, 'positive_score' : posi_score}, index = [0])
                                        self.emotional = pd.concat([self.emotional, second], ignore_index= True)

                                with col1.container():
                                    tab1,tab2,tab3 = st.tabs(['긍정','부정','워드'])
                                    with tab1.container():
                                        for i in st.session_state['positive_space']:
                                            st.write(i)
                                    with tab2.container():
                                        for i in st.session_state['negative_space']:
                                            st.write(i)

                                with col2.container():
                                    st.write('긍부정 비율')
                                    head = st.empty()
                                    chart = st.empty()
                                    pie_chart(self.emotional, head, chart)
                                st.stop()
                                break        
                        time.sleep(1)
                # with a.container():
                #     st.write('문장분리 ', st.session_state.split_sentence) # 추후 제거 필요 (확인용)
                #     st.write('긍정문 공간', st.session_state.positive_space) # 추후 제거 필요 (확인용)
                #     st.write('부정문 공간', st.session_state.negative_space)
                #     st.write('유저인풋공간',  st.session_state.past) # 추후 제거 필요 (확인용)
            #######################################################################################
                
def main():
    
    
    if 'past' not in st.session_state: # 내 입력채팅값 저장할 리스트
        st.session_state['past'] = [] 

    if 'generated' not in st.session_state: # 챗봇채팅값 저장할 리스트
        st.session_state['generated'] = []
        
    if 'positive_space' not in st.session_state: # 긍정 문장 저장할 리스트
        st.session_state['positive_space'] = []
        
    if 'negative_space' not in st.session_state: # 부정 문장 저장할 리스트
        st.session_state['negative_space'] = []
    
    if 'split_sentence' not in st.session_state: # 분리된 문장 저장할 리스트 
        st.session_state['split_sentence'] = [] 
        
    if 'word_cloud' not in st.session_state: # word cloud 문장 저장할 리스트
        st.session_state['word_cloud'] = []
             
    emotional = pd.DataFrame()
    global text_list
    text_list = pd.DataFrame()
    
    start = chat(emotional,text_list)
    start.chatting()
    
if __name__ == "__main__":
    main()


        
# def is_in_ele(ele_list, text,sidebar2):
#     chatbot_output1 = predict(extract_word(text)) # 문장 전처리 후 3요소 예측 진행
    
#     select_box(chatbot_output1, text, sidebar2)
#     ok_elelist = [ele for ele in ele_list if chatbot_output1[ele]>=0.5] # 기준값 보다 높으면 ok_list에 생성
#     for ele in ok_elelist:
#         ele_list.remove(ele) # ele_list에 있는 3요소를 ok_list와 비교해서 제거 
#     return ele_list
   
# def select_box(chatbot_output1, text, sidebar2):
    
#     append_list = pd.DataFrame({'review':text, 'taste':chatbot_output1[0], 'quantity' : chatbot_output1[1], 'delivery' : chatbot_output1[2]}, index = [0])
#     global text_list
#     text_list = pd.concat([text_list, append_list], ignore_index= True)
    
#     with sidebar2.container():
#         option = st.selectbox('pick!', text_list['review'])
#         # st.write('gogo', option)
        
#         if option in text_list['review'].tolist():
#             index = text_list['review'].tolist().index(option)
#             ########################### 한글 설정 ############################
#             font_path = "/Library/Fonts/BMJUA_otf.otf"
#             font = font_manager.FontProperties(fname = font_path).get_name()
#             rc('font',family = font)
#             #################################################################
            
#             bar_value = [text_list.loc[index][1],text_list.loc[index][2],text_list.loc[index][3]]
#             labels = ['맛','양 or 서비스','배달']
#             fig, ax = plt.subplots()
#             ax.bar(x = labels, height = bar_value, color = ['r','g','b'],
#                     align = 'center', edgecolor = 'lightgray', linewidth = 3)
            
#             ############## 그래프에 숫자 넣기 ##############
#             for i, v in enumerate(labels):
#                 ax.text(v, bar_value[i], round(bar_value[i],3), 
#                         fontsize = 14, 
#                         color='black',
#                         horizontalalignment='center',  
#                         verticalalignment='bottom')    
#             ############################################
            
#             parameters = {'axes.titlesize': 20}
#             plt.rcParams.update(parameters)
#             ax.set_title(text_list.loc[index][0])
#             ax.set_xticklabels(labels = labels, fontsize = 15)
#             ax.set_yticks([0,0.5,1.2])
#             # with sidebar2.container():
#             st.pyplot(fig)
