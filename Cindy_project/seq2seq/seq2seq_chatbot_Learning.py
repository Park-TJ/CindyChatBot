# -*- coding: utf-8 -*-
"""Chatbot learning
학습시 생성된 vocab 딕셔너리 파일을 Cindy ui 실행시 경로를 동일시 해주어야 연결성 있는 문장을 생성해줍니다.
"""



from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras import preprocessing

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
import re

from konlpy.tag import Okt
import pickle
import tensorflow as tf
tf.__version__

# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 30

# 임베딩 벡터 차원
embedding_dim = 100

# LSTM 히든레이어 차원
lstm_hidden_dim = 128

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()]")

# 챗봇 데이터 로드
chatbot_data = pd.read_csv('./seq2seq/ChatbotData_Cindy.csv', encoding='utf-8')
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])

chatbot_data.head()

len(chatbot_data['Q'].unique())

# 데이터 개수
len(question)


# 형태소분석 함수
def pos_tag(sentences):
    
    # KoNLPy 형태소분석기 설정
    tagger = Okt()
    
    # 문장 품사 변수 초기화
    sentences_pos = []
    
    # 모든 문장 반복
    for sentence in sentences:
        # 특수기호 제거
        sentence = re.sub(RE_FILTER, "", sentence)
        #print(sentence)
        # 배열인 형태소분석의 출력을 띄어쓰기로 구분하여 붙임
        sentence = " ".join(tagger.morphs(sentence))
        sentences_pos.append(sentence)
        
    return sentences_pos

# 형태소분석 수행
question = pos_tag(question)
answer = pos_tag(answer)


# 질문과 대답 문장들을 하나로 합침
sentences = []
sentences.extend(question)
sentences.extend(answer)

words = []

# 단어들의 배열 생성
for sentence in sentences:
    for word in sentence.split():
        words.append(word)

# 길이가 0인 단어는 삭제
words = [word for word in words if len(word) > 0]
# 중복된 단어 삭제
words = list(set(words))
# 제일 앞에 태그 단어 삽입
words[:0] = [PAD, STA, END, OOV]
# 단어 개수
len(words)

# 단어와 인덱스의 딕셔너리 생성


word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}


#word_index vocab 저장 - > 
with open('./seq2seq/vocab_dict/word_to_index_final.pickle', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)

with open('./seq2seq/vocab_dict/index_to_word_final.pickle', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)


# 단어 -> 인덱스
# 문장을 인덱스로 변환하여 모델 입력으로 사용
print(dict(list(word_to_index.items())[:20]))

# 인덱스 -> 단어
# 모델의 예측 결과인 인덱스를 문장으로 변환시 사용
print(dict(list(index_to_word.items())[:20]))

# 문장을 인덱스로 변환
def convert_text_to_index(sentences, vocabulary, type): 
    
    sentences_index = []
    
    # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = []
        
        # 디코더 입력일 경우 맨 앞에 START 태그 추가
        if type == DECODER_INPUT:
            sentence_index.extend([vocabulary[STA]])
        
        # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence.split():
            if vocabulary.get(word) is not None:
                # 사전에 있는 단어면 해당 인덱스를 추가
                sentence_index.extend([vocabulary[word]])
            else:
                # 사전에 없는 단어면 OOV 인덱스를 추가
                sentence_index.extend([vocabulary[OOV]])

        # 최대 길이 검사
        if type == DECODER_TARGET:
            # 디코더 목표일 경우 맨 뒤에 END 태그 추가
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else:
                sentence_index += [vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]
            
        # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]
        
        # 문장의 인덱스 배열을 추가
        sentences_index.append(sentence_index)

    return np.asarray(sentences_index)

# 인코더 입력 인덱스 변환
x_encoder = convert_text_to_index(question, word_to_index, ENCODER_INPUT)

# 첫 번째 인코더 입력 출력 (12시 땡)
x_encoder[0]

# 디코더 입력 인덱스 변환
x_decoder = convert_text_to_index(answer, word_to_index, DECODER_INPUT)

# 첫 번째 디코더 입력 출력 (START 하루 가 또 가네요)
x_decoder[0]

len(x_decoder[0])

# 디코더 목표 인덱스 변환
y_decoder = convert_text_to_index(answer, word_to_index, DECODER_TARGET)

# 첫 번째 디코더 목표 출력 (하루 가 또 가네요 END)
print(y_decoder[0])

# 원핫인코딩 초기화
one_hot_data = np.zeros((len(y_decoder), max_sequences, len(words)))

# 디코더 목표를 원핫인코딩으로 변환
# 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
for i, sequence in enumerate(y_decoder):
    for j, index in enumerate(sequence):
        one_hot_data[i, j, index] = 1

# 디코더 목표 설정
y_decoder = one_hot_data

# 첫 번째 디코더 목표 출력
print(y_decoder[0])

#--------------------------------------------
# 훈련 모델 인코더 정의
#--------------------------------------------

# 입력 문장의 인덱스 시퀀스를 입력으로 받음
encoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어
encoder_outputs = layers.Embedding(len(words), embedding_dim)(encoder_inputs)

# return_state가 True면 상태값 리턴
# LSTM은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
encoder_outputs, state_h, state_c = layers.LSTM(lstm_hidden_dim,
                                                dropout=0.1,
                                                recurrent_dropout=0.5,
                                                return_state=True)(encoder_outputs)

# 히든 상태와 셀 상태를 하나로 묶음
encoder_states = [state_h, state_c]



#--------------------------------------------
# 훈련 모델 디코더 정의
#--------------------------------------------

# 목표 문장의 인덱스 시퀀스를 입력으로 받음
decoder_inputs = layers.Input(shape=(None,))

# 임베딩 레이어
decoder_embedding = layers.Embedding(len(words), embedding_dim)
decoder_outputs = decoder_embedding(decoder_inputs)

# 인코더와 달리 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴
# 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
decoder_lstm = layers.LSTM(lstm_hidden_dim,
                           dropout=0.1,
                           recurrent_dropout=0.5,
                           return_state=True,
                           return_sequences=True)

# initial_state를 인코더의 상태로 초기화
decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                     initial_state=encoder_states)

# 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
decoder_dense = layers.Dense(len(words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



#--------------------------------------------
# 훈련 모델 정의
#--------------------------------------------

# 입력과 출력으로 함수형 API 모델 생성
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 학습 방법 설정
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#--------------------------------------------
#  예측 모델 인코더 정의
#--------------------------------------------

# 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
encoder_model = models.Model(encoder_inputs, encoder_states)



#--------------------------------------------
# 예측 모델 디코더 정의
#--------------------------------------------

# 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
# 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim,))
decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]    

# 임베딩 레이어
decoder_outputs = decoder_embedding(decoder_inputs)

# LSTM 레이어
decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs,
                                                 initial_state=decoder_states_inputs)

# 히든 상태와 셀 상태를 하나로 묶음
decoder_states = [state_h, state_c]

# Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
decoder_outputs = decoder_dense(decoder_outputs)

# 예측 모델 디코더 설정
decoder_model = models.Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

# 인덱스를 문장으로 변환
def convert_index_to_text(indexs, vocabulary): 
    
    sentence = ''
    
    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == END_INDEX:
            # 종료 인덱스면 중지
            break;
        if vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            # 사전에 없는 인덱스면 OOV 단어를 추가
            sentence.extend([vocabulary[OOV_INDEX]])
            
        # 빈칸 추가
        sentence += ' '

    return sentence

# len(x_decoder)
#
# len(y_decoder)

#model.summary()

#encoder_model.summary()

#decoder_model.summary()

from tqdm import tqdm
#에폭 반복
for epoch in range(10):
    print('Total Epoch :', epoch + 1)

    history = model.fit([x_encoder, x_decoder], y_decoder, epochs=100, batch_size=64, verbose=1)
    model.summary()

    # 정확도와 손실 출력
    print('accuracy :', history.history['accuracy'][-1])
    print('loss :', history.history['loss'][-1])

    # 문장 예측 테스트
        # (3 박 4일 놀러 가고 싶다) -> (여행 은 언제나 좋죠)
    input_encoder = x_encoder[2].reshape(1, x_encoder[2].shape[0])
    input_decoder = x_decoder[2].reshape(1, x_decoder[2].shape[0])
    results = model.predict([input_encoder, input_decoder])

    # 결과의 원핫인코딩 형식을 인덱스로 변환
    # 1축을 기준으로 가장 높은 값의 위치를 구함
    indexs = np.argmax(results[0], 1)

        # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)



#모델 가중치 저장
model.save_weights('./seq2seq/seq2seq_model/seq2seq2_model_weights')
encoder_model.save_weights('./seq2seq/seq2seq_model/seq2seq2_encoder_model_weights')
decoder_model.save_weights('./seq2seq/seq2seq_model/seq2seq2_decoder_model_weights')


# 예측을 위한 입력 생성
def make_predict_input(sentence):

    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT)
    
    return input_seq

# 텍스트 생성
def generate_text(input_seq):
    
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX
    
    # 인덱스 초기화
    indexs = []
    
    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
                                                [target_seq] + states)


        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)
        
        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
        
    return sentence

# 문장을 인덱스로 변환
input_seq = make_predict_input('3박4일 놀러가고 싶다')
input_seq

# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
print(sentence)

# 문장을 인덱스로 변환
input_seq = make_predict_input('3박4일 같이 놀러가고 싶다')
input_seq

# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
print(sentence)

# 문장을 인덱스로 변환
input_seq = make_predict_input('3박4일 놀러가려고')
print(sentence)


# 예측 모델로 텍스트 생성
sentence = generate_text(input_seq)
print(sentence)


input_seq = make_predict_input('SNS 시간낭비인데')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('PPL 너무나 심하네')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('가상화폐 망함')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('가스불')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('가스비')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('가족 보고 싶어')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('간식 먹고 싶어')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('간접흡연 싫어')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('감기 기운 잇어')
sentence = generate_text(input_seq)
print(sentence)

input_seq = make_predict_input('내일 날씨 어떄?')
sentence = generate_text(input_seq)
print(sentence)


