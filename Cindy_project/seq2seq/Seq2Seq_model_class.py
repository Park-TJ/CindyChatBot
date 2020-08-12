#!/usr/bin/env python
# coding: utf-8

# In[13]:


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras import preprocessing

import numpy as np
import pandas as pd

import os
import re
import pickle

from konlpy.tag import Okt

class seq2seq:
  def __init__(self):
    # 태그 단어
    self.PAD = "<PADDING>"   # 패딩
    self.STA = "<START>"     # 시작
    self.END = "<END>"       # 끝
    self.OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

    # 태그 인덱스
    self.PAD_INDEX = 0
    self.STA_INDEX = 1
    self.END_INDEX = 2
    self.OOV_INDEX = 3

    # 데이터 타입
    self.ENCODER_INPUT  = 0
    self.DECODER_INPUT  = 1
    self.DECODER_TARGET = 2

    # 한 문장에서 단어 시퀀스의 최대 개수
    self.max_sequences = 30

    # 임베딩 벡터 차원
    self.embedding_dim = 100

    # LSTM 히든레이어 차원
    self.lstm_hidden_dim = 128

    # 정규 표현식 필터
    self.RE_FILTER = re.compile("[.,!?\"':;~()]")

    #학습시 생성한 word_index vocab호출
    with open('./seq2seq/vocab_dict/index_to_word_final.pickle', 'rb') as f:
      self.index_to_word = pickle.load(f)

    with open('./seq2seq/vocab_dict/word_to_index_final.pickle', 'rb') as f:
      self.word_to_index = pickle.load(f)
    
        #--------------------------------------------
    # 훈련 모델 인코더 정의
    #--------------------------------------------
    # 입력 문장의 인덱스 시퀀스를 입력으로 받음
    encoder_inputs = layers.Input(shape=(None,))
    # 임베딩 레이어
    encoder_outputs = layers.Embedding(len(self.index_to_word), self.embedding_dim)(encoder_inputs)
    # return_state가 True면 상태값 리턴
    # LSTM은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
    encoder_outputs, state_h, state_c = layers.LSTM(self.lstm_hidden_dim,
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
    decoder_embedding = layers.Embedding(len(self.index_to_word), self.embedding_dim)
    decoder_outputs = decoder_embedding(decoder_inputs)
    # 인코더와 달리 return_sequences를 True로 설정하여 모든 타임 스텝 출력값 리턴
    # 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
    decoder_lstm = layers.LSTM(self.lstm_hidden_dim,
                              dropout=0.1,
                              recurrent_dropout=0.5,
                              return_state=True,
                              return_sequences=True)

    # initial_state를 인코더의 상태로 초기화
    decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                        initial_state=encoder_states)

    # 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
    decoder_dense = layers.Dense(len(self.index_to_word), activation='softmax')
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
    decoder_state_input_h = layers.Input(shape=(self.lstm_hidden_dim,))
    decoder_state_input_c = layers.Input(shape=(self.lstm_hidden_dim,))
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
    
    self.model = model
    self.encoder_model = encoder_model
    self.decoder_model = decoder_model

    #가중치 불러오기
    self.model.load_weights('./seq2seq/seq2seq_model/seq2seq2_model_weights')
    self.encoder_model.load_weights('./seq2seq/seq2seq_model/seq2seq2_encoder_model_weights')
    self.decoder_model.load_weights('./seq2seq/seq2seq_model/seq2seq2_decoder_model_weights')
    print(self.model.summary())
  
  # 형태소분석 함수
  def pos_tag(self, sentences):
      
      # KoNLPy 형태소분석기 설정
      tagger = Okt()
      
      # 문장 품사 변수 초기화
      sentences_pos = []
      
      # 모든 문장 반복
      for sentence in sentences:
          # 특수기호 제거
          sentence = re.sub(self.RE_FILTER, "", sentence)
          #print(sentence)
          # 배열인 형태소분석의 출력을 띄어쓰기로 구분하여 붙임
          sentence = " ".join(tagger.morphs(sentence))
          sentences_pos.append(sentence)
          
      return sentences_pos

  def make_predict_input(self, sentence):

    sentences = []
    sentences.append(sentence)
    sentences = self.pos_tag(sentences)
    input_seq = self.convert_text_to_index(sentences, self.word_to_index, self.ENCODER_INPUT)
    
    return input_seq

    # 인덱스를 문장으로 변환
  def convert_index_to_text(self, indexs, vocabulary): 
      
      sentence = ''
      
      # 모든 문장에 대해서 반복
      for index in indexs:
          if index == self.END_INDEX:
              # 종료 인덱스면 중지
              break;
          if vocabulary.get(index) is not None:
              # 사전에 있는 인덱스면 해당 단어를 추가
              sentence += vocabulary[index]
          else:
              # 사전에 없는 인덱스면 OOV 단어를 추가
              sentence.extend([vocabulary[self.OOV_INDEX]])
              
          # 빈칸 추가
          sentence += ' '

      return sentence

      
  # 문장을 인덱스로 변환
  def convert_text_to_index(self, sentences, vocabulary, type): 
      
      sentences_index = []
      
      # 모든 문장에 대해서 반복
      for sentence in sentences:
          sentence_index = []
          
          # 디코더 입력일 경우 맨 앞에 START 태그 추가
          if type == self.DECODER_INPUT:
              sentence_index.extend([vocabulary[self.STA]])
          
          # 문장의 단어들을 띄어쓰기로 분리
          for word in sentence.split():
              if vocabulary.get(word) is not None:
                  # 사전에 있는 단어면 해당 인덱스를 추가
                  sentence_index.extend([vocabulary[word]])
              else:
                  # 사전에 없는 단어면 OOV 인덱스를 추가
                  sentence_index.extend([vocabulary[self.OOV]])

          # 최대 길이 검사
          if type == self.DECODER_TARGET:
              # 디코더 목표일 경우 맨 뒤에 END 태그 추가
              if len(sentence_index) >= self.max_sequences:
                  sentence_index = sentence_index[:self.max_sequences-1] + [vocabulary[self.END]]
              else:
                  sentence_index += [vocabulary[self.END]]
          else:
              if len(sentence_index) > self.max_sequences:
                  sentence_index = sentence_index[:self.max_sequences]
              
          # 최대 길이에 없는 공간은 패딩 인덱스로 채움
          sentence_index += (self.max_sequences - len(sentence_index)) * [vocabulary[self.PAD]]
          
          # 문장의 인덱스 배열을 추가
          sentences_index.append(sentence_index)

      return np.asarray(sentences_index)


      # 텍스트 생성
  def generate_text(self, input_seq):
      
      # 입력을 인코더에 넣어 마지막 상태 구함
      states = self.encoder_model.predict(input_seq)

      # 목표 시퀀스 초기화
      target_seq = np.zeros((1, 1))
      
      # 목표 시퀀스의 첫 번째에 <START> 태그 추가
      target_seq[0, 0] = self.STA_INDEX
      
      # 인덱스 초기화
      indexs = []
      
      # 디코더 타임 스텝 반복
      while 1:
          # 디코더로 현재 타임 스텝 출력 구함
          # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
          decoder_outputs, state_h, state_c = self.decoder_model.predict(
                                                  [target_seq] + states)

          # 결과의 원핫인코딩 형식을 인덱스로 변환
          index = np.argmax(decoder_outputs[0, 0, :])
          indexs.append(index)
          
          # 종료 검사
          if index == self.END_INDEX or len(indexs) >= self.max_sequences:
              break

          # 목표 시퀀스를 바로 이전의 출력으로 설정
          target_seq = np.zeros((1, 1))
          target_seq[0, 0] = index
          
          # 디코더의 이전 상태를 다음 디코더 예측에 사용
          states = [state_h, state_c]

      # 인덱스를 문장으로 변환
      sentence = self.convert_index_to_text(indexs, self.index_to_word)
      to_matching = sentence.split(' ')
      to_matching = to_matching[:-1]
      chatbot_data = pd.read_csv('./seq2seq/ChatbotData_Cindy.csv', encoding='utf-8')
      try:
        for_matching = list(chatbot_data[chatbot_data.A.apply(lambda sentence1: all(word in sentence1 for word in to_matching))]['A'])
        return_sentence = for_matching[0]
      except IndexError:
        return_sentence = sentence
      return return_sentence
      

  def get_answer(self, text):
    input_seq = self.make_predict_input(text)
    return self.generate_text(input_seq)

      




