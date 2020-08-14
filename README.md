# Ai_chatbot Cindy project - 나만의 비서 Cindy
#### * 나만의 비서 Cindy는 ETRI Korbert(Google bert모델기반) 와 Seq2Seq model을 기반으로 한 자연어처리 대화형 챗봇입니다.
#### * 뉴스, 날씨, 맛집, 버스에 대한 질문이 가능하며, 일상대화도 가능합니다

# 실행화면 
#### pyqt_UI_run.py 실행
<div>
  <img width='350', src="https://user-images.githubusercontent.com/63779100/89993048-1ce83400-dcc1-11ea-85dc-49d396530f88.gif">
  <img width='350', src="https://user-images.githubusercontent.com/63779100/89993364-8cf6ba00-dcc1-11ea-96e5-23bba4c2669a.gif">
 </div>
 <div>
 <img width='350', src="https://user-images.githubusercontent.com/63779100/89993454-b3b4f080-dcc1-11ea-8458-b744af9e1993.gif">
 <img width='350', src="https://user-images.githubusercontent.com/63779100/89993458-b4e61d80-dcc1-11ea-8f0d-6284f5797902.gif">
 </div>

# Requirements
Python==3.6.10  
tensorflow==2.1.0  
PyQt5==5.15.0  
bert-for-tf2==0.14.4  
konlpy==0.5.2  
Keras==2.4.3  
torch,torchvision -> https://pytorch.org/

### Pretrained model for intent_classification :
Requirements - keras_bert, keras_radam

ETRI Korean_BERT_WordPiece - 
ETRI 의 [한국어 BERT 모델](http://aiopen.etri.re.kr/)을 활용하시려면 다음의 [링크](http://aiopen.etri.re.kr/service_dataset.php) 에서 서약서를 작성하시고 키를 받으셔서 다운받으시면 됩니다. 
(사용 허가 협약서를 준수하기 때문에 pretrained 모델을 공개하지 않습니다.)

### Pretrained model for NER_classification -  
Tokenizer - [SentencePiece](https://github.com/google/sentencepiece)  
koBert_Model - monologg님의 kobert를 활용하여 진행하였습니다.  

### Requirements Install 
```
pip install -r requirements.txt
```



# Hyperparameters 

| Model | batch size | epoch | 학습환경 | 학습시간 |
| ------ | ------ | ------ |------ | ------ |
| Intent_CL | 10 | 10 | Google Colab | GPU 런타임 기준 약 50분 |
| NER | 32 | 3 | Google Colab | TPU 런타임 기준 약 20분 |
| Seq2seq | 100 | 10 | Local PC GPU | RTX 2080 TI 기준 약 20시간 |


## Cindy process
<p align="left">
<img width="800" src="https://user-images.githubusercontent.com/63779100/89996152-69357300-dcc5-11ea-8777-d1c1c3eca2a3.png">
</p>  
Normal Answer : seq2seq Model Return<br>
Weather Answer : Get_Weather() - Google Search Weather Crawling<br>
News Answer : Get_News() - Naver News Crawling<br>
Food Answer : Get_Food() - Daum Map Search Crawling<br>  
Bus Answer : Get_Bus() - Seoul Bus API Crawling<br>

## Intent_Classification
<p align="left">
<img width="800" src="https://user-images.githubusercontent.com/63779100/89996754-2758fc80-dcc6-11ea-9002-83dc27515398.PNG">
</p>


## NER_Classification
<p align="left">
<img width="800" src="https://user-images.githubusercontent.com/63779100/89996764-2d4edd80-dcc6-11ea-9972-280e0fc09289.png">
</p>



# 실행방법
./Intent_cl/intent_cl_Learning.py 실행 - Category data 학습 및 가중치 저장

./ner/ner_Learning.py 실행 - NER-파악 data 학습 및 가중치 저장 

Cindy_project/pyqt_UI_run.py 실행 -> 저장된 모델들과 가중치 load 및 Cindy 챗봇 실행 

- seq2seq model은 학습된 모델 가중치를 업로드 했습니다. 대화형 데이터 셋을  
수정하고 다시 학습해보고자 하신다면 ChatbotData_Cindy.csv를 수정하시고
./seq2seq/seq2seq_chatbot_Learning.py 를 실행해주세요. 학습에 사용된 vocab_dict는 동일한 pickle파일로 호출해주셔야 합니다. 


###  오라클 자바 교육센터 - 파이썬을 활용한 빅데이터 분석 인공지능(AI) 머신러닝 개발자 양성과정 1기 
#### Cindy_project Maintainers - [구대웅](https://github.com/GuDaeWoong), [김준연](https://github.com/pentagram5), [박태준](https://github.com/Park-TJ), 성지연, 정재훈, 황명수, 이민석
