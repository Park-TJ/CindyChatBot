# Ai_chatbot Cindy project - 나만의 비서 Cindy
#### * 나만의 비서 Cindy는 ETRI Korbert(Google bert모델기반) 와 Seq2Seq model을 기반으로 한 자연어처리 대화형 챗봇입니다.
#### * 뉴스, 날씨, 맛집, 버스에 대한 질문이 가능하며, 일상대화도 가능합니다

## Requirements
Python==3.6.10

tensorflow==2.1.0

torch==1.5.1+cu101

PyQt5==5.15.0

bert-for-tf2==0.14.4

konlpy==0.5.2

Keras==2.4.3

Pretrained model for intent_classification :

ETRI Korean_BERT_WordPiece

ETRI 의 [한국어 BERT 모델](http://aiopen.etri.re.kr/)을 활용하시려면 다음의 [링크](http://aiopen.etri.re.kr/service_dataset.php) 에서 서약서를 작성하시고 키를 받으셔서 다운받으시면 됩니다. 
(사용 허가 협약서를 준수하기 때문에 pretrained 모델을 공개하지 않습니다.**)

Pretrained model for NER_classification:

## 실행방법
./seq2seq/seq2seq_chatbot_Learning.py 실행 - 대화형 데이터 학습 및 가중치 저장

./intent_data/intent_cl_Learning.py 실행 - Category data 학습 및 가중치 저장

./ner/ner_class_Learning.py 실행 - NER-파악 data 학습 및 가중치 저장 

Cindy_project/pyqt_UI_run.py 실행 -> 저장된 모델들과 가중치 load 및 Cindy 챗봇 실행 

## 나만의 비서 Cindy 개발기입니다


## Cindy process
<p align="center">
<img width="5000" src="https://user-images.githubusercontent.com/63779100/89380978-6a015e80-d733-11ea-9c40-3d8b8529ba12.jpg">
</p>

## Intent_classiication
<p align="center">
<img width="5000" src="https://user-images.githubusercontent.com/63779100/89381132-a634bf00-d733-11ea-93aa-d8cb17316bab.png">
</p>

## Ner_Classification
<p align="center">
<img width="5000" src="https://user-images.githubusercontent.com/63779100/89381135-a8971900-d733-11ea-89c3-42a816c6f644.png">
</p>


###  오라클 자바 교육센터 - 파이썬을 활용한 빅데이터 분석 인공지능(AI) 머신러닝 개발자 양성과정 1기 
#### Cindy_project Maintainers - 구대웅, [김준연](https://github.com/pentagram5), 박태준, 성지연, 정재훈, 황명수, 이민석
