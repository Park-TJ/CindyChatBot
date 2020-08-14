# 라이브러리
import pickle
import gzip
import time
import pandas as pd
import tensorflow as tf
from transformers import *
import numpy as np
import re
from bs4 import BeautifulSoup
import urllib
import bs4
import requests
import datetime
from selenium import webdriver as wd
import logging
import os
import unicodedata
from shutil import copyfile
from transformers import PreTrainedTokenizer

# PreTrainedTokenizer를 상속받은 KoBertTokenizer 클래스 선언
# sentencepiece 사용
# monologg님의 kobert 활용
class KoBertTokenizer(PreTrainedTokenizer):
    logger = logging.getLogger(__name__)

    VOCAB_FILES_NAMES = {"vocab_file": "tokenizer_78b3253a26.model",
                         "vocab_txt": "vocab.txt"}

    PRETRAINED_VOCAB_FILES_MAP = {
        "vocab_file": {
            "monologg/kobert": "./ner/vocab/tokenizer_78b3253a26.model",
        },
        "vocab_txt": {
            "monologg/kobert": "./ner/vocab/vocab.txt",
        }
    }

    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
        "monologg/kobert": 512,
    }

    PRETRAINED_INIT_CONFIGURATION = {
        "monologg/kobert": {"do_lower_case": False},
    }

    SPIECE_UNDERLINE = u'▁'

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            vocab_file,
            vocab_txt,
            do_lower_case=False,
            remove_space=True,
            keep_accents=False,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",

            **kwargs):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

        # Build vocab
        self.token2idx = dict()
        self.idx2token = []
        with open(vocab_txt, 'r', encoding='utf-8') as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token.append(token)

        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece")

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.vocab_txt = vocab_txt

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.idx2token)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, return_unicode=True, sample=False):
        """ Tokenize a string. """
        text = self.preprocess_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def _convert_token_to_id(self, token):
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def _convert_id_to_token(self, index, return_unicode=True):
        return self.idx2token[index]

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory):

        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return

        # 1. Save sentencepiece model
        out_vocab_model = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_model):
            copyfile(self.vocab_file, out_vocab_model)

        # 2. Save vocab.txt
        index = 0
        out_vocab_txt = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_txt"])
        with open(out_vocab_txt, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(out_vocab_txt)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

        return out_vocab_model, out_vocab_txt

# NER 클래스 선언
class NER:
    # 토크나이저와 각변수 및 단어사전 pickle 로드
    def __init__(self):
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        self.SEQ_LEN = 72
        self.w_type = 1
        with gzip.open('./ner/pickle/label_dict.pickle', 'rb') as f:
            self.label_dict = pickle.load(f)
        with gzip.open('./ner/pickle/index_to_ner.pickle', 'rb') as f:
            self.index_to_ner = pickle.load(f)

        self.news_date_tags = []
        self.news_categori_tags = []
        self.news_answer_date = []
        self.news_news_tags = []

        self.tagstate = 'N'  # N : 기본 L : LOC 없음 D: DAT 없음 F: 답변완료
        self.date_tags = []
        self.loc_tags = []
        self.answer_date = []

        self.rtn_str = ''

    # monologg/kobert의 사전학습된 모델 생성 후 파인튜닝한 가중치 로드
    def create_model(self):
        model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True, num_labels=len(self.label_dict),
                                            output_attentions=False, output_hidden_states=False)

        token_inputs = tf.keras.layers.Input((self.SEQ_LEN,), dtype=tf.int32, name='input_word_ids')  # 토큰 인풋
        mask_inputs = tf.keras.layers.Input((self.SEQ_LEN,), dtype=tf.int32, name='input_masks')  # 마스크 인풋

        bert_outputs = model([token_inputs, mask_inputs])
        bert_outputs = bert_outputs[0]  # shape : (Batch_size, max_len, 30(개체의 총 개수))
        nr = tf.keras.layers.Dense(36, activation='softmax')(bert_outputs)  # shape : (Batch_size, max_len, 30)

        self.nr_model = tf.keras.Model([token_inputs, mask_inputs], nr)

        self.nr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00002),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                              metrics=['sparse_categorical_accuracy'])
        self.nr_model.summary()
        # 자신의환경에 맞게 경로를 수정하셔야 합니다.
        self.nr_model.load_weights('./ner/model/ner_class_model_weight.h5')
        return self.nr_model

    # 입력된 질문에 대해서 라벨과 토큰으로 분류해주는 함수 선언
    def ner_inference(self, test_sentence):
        self.label_list = []
        self.token_list = []
        self.tokenized_sentence = np.array(
            [self.tokenizer.encode(test_sentence, max_length=self.SEQ_LEN, pad_to_max_length=True, truncation=True)])
        self.tokenized_mask = np.array([[int(x != 1) for x in self.tokenized_sentence[0].tolist()]])
        self.ans = self.nr_model.predict([self.tokenized_sentence, self.tokenized_mask])
        self.ans = np.argmax(self.ans, axis=2)

        self.tokens = self.tokenizer.convert_ids_to_tokens(self.tokenized_sentence[0])
        self.new_tokens, self.new_labels = [], []
        for self.token, self.label_idx in zip(self.tokens, self.ans[0]):

            if (self.token.startswith("▁")):
                self.new_labels.append(self.index_to_ner[self.label_idx])
                self.new_tokens.append(self.token[1:])
            elif (self.token == '[CLS]'):
                pass
            elif (self.token == '[SEP]'):
                pass
            elif (self.token == '[PAD]'):
                pass
            elif (self.token != '[CLS]' or token != '[SEP]'):
                self.new_tokens[-1] = self.new_tokens[-1] + self.token

        for token, label in zip(self.new_tokens, self.new_labels):
            self.label_list.append(label)
            self.token_list.append(token)
        return self.label_list, self.token_list

    # 네이버 많이본 뉴스에서 크롤링해오는 함수 선언
    # 어제,오늘과 같은 단어에 대해서 함수내부에서 별도처리
    def get_news(self, question):
        self.news_date_tags = []
        self.news_categori_tags = []
        self.news_answer_date = []
        self.news_news_tags = []

        labels, tokens = self.ner_inference(question)
        for i, label in enumerate(labels):
            if label.find('DAT') >= 0:
                self.news_answer_date.append(tokens[i])
                self.news_date_tags.append(tokens[i])
            if label.find('NWS_I') >= 0:
                self.news_categori_tags.append(tokens[i])
            if label.find('NWS_B') >= 0:
                self.news_news_tags.append(tokens[i])

        if len(self.news_date_tags) > 0:
            if ' '.join(self.news_date_tags).find('어제') >= 0:
                now_date = datetime.datetime.now()
                year = str((now_date - datetime.timedelta(days=1)).year)
                month = str((now_date - datetime.timedelta(days=1)).month).zfill(2)
                day = str((now_date - datetime.timedelta(days=1)).day).zfill(2)
                target_date = year + month + day

            elif ' '.join(self.news_date_tags).find('오늘') >= 0:
                now_date = datetime.datetime.now()
                target_date = str(now_date.year) + str(now_date.month).zfill(2) + str(now_date.day).zfill(2)

            elif (' '.join(self.news_date_tags).find('월') >= 0) & (' '.join(self.news_date_tags).find('일') >= 0):
                for d in self.news_date_tags:
                    if d.find('월') >= 0 & d.find('일') >= 0:
                        month = ''.join(re.findall('\d', d.split('월')[0])).zfill(2)
                        day = ''.join(re.findall('\d', d.split('월')[1])).zfill(2)
                    else:
                        if d.find('월') >= 0:
                            month = ''.join(re.findall('\d', d)).zfill(2)
                        if d.find('일') >= 0:
                            day = ''.join(re.findall('\d', d)).zfill(2)
                now_date = datetime.datetime.now()
                target_date = str(now_date.year) + month + day
            else:
                now_date = datetime.datetime.now()
                target_date = str(now_date.year) + str(now_date.month).zfill(2) + str(now_date.day).zfill(2)

            url = 'https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&date=' + target_date

            r = requests.get(url)
            html = r.content
            soup = BeautifulSoup(html, 'html.parser')
            titles_html = soup.select('.ranking_section > ol > li > dl > dt > a')

            news = []
            link = []
            cate = []
            j = 0
            lis = ["정치", "경제", "사회", "생활/문화", "세계", "It/과학"]
            lis_color = {'정치': '#FB4538', '경제': '#3F18FE', '사회': '#AF810E', '생활/문화': '#207DFD', '세계': '#1FDA3E',
                         'It/과학': '#7B84ED'}
            for i in range(len(titles_html)):
                if i in [5, 10, 15, 20, 25]:
                    news.append(titles_html[i].text)
                    link.append("https://news.naver.com" + titles_html[i].get('href'))
                    j += 1
                    cate.append(lis[j])
                else:
                    news.append(titles_html[i].text)
                    link.append("https://news.naver.com" + titles_html[i].get('href'))
                    cate.append(lis[j])

            result_cate = []
            result_title = []
            result_link = []
            if len(self.news_categori_tags) > 0:
                print('(', ' '.join(self.news_answer_date), ' '.join(self.news_categori_tags), ')로 요청하신 정보입니다.')
                self.rtn_str = "<table width='80%' height='80%' style='margin-bottom:2px;border-color:#9ABAD9;background-color:snow;padding:5px 5px 5px 5px;'>"
                for i, val in enumerate(cate):
                    for c in self.news_categori_tags:
                        if (c.find(val) >= 0) | (c[0:2] == val[0:2]):
                            self.rtn_str += "<tr><td><font color='" + lis_color[
                                val] + "'>[<b>{}</b>]</font></td><td>{}</td><td><a href='{}'><img width='15' height='15' src='./ner/img/link_icon2.png'></a></td></tr>".format(
                                val, news[i], link[i])
                return self.rtn_str + '</table>'
            else:
                self.rtn_str = "<table width='80%' height='80%' style='margin-bottom:2px;border-color:#9ABAD9;background-color:snow;padding:5px 5px 5px 5px;'>"
                for i, val in enumerate(cate):
                    self.rtn_str += "<tr><td><font color='" + lis_color[
                        val] + "'>[<b>{}</b>]</font></td><td>{}</td><td><a href='{}'><img width='15' height='15' src='./ner/img/link_icon2.png'></a></td></tr>".format(
                        val, news[i], link[i])
                return self.rtn_str + '</table>'
        else:
            self.rtn_str = '다시 요청해주세요.'
            return self.rtn_str

    # 입력된 라벨과 토큰 기준으로 구글에서 날씨 정보를 크롤링하는 함수 선언
    # 내부 상태값으로 위치와 날짜에 대한 정보를 제어
    def get_weather(self, question):
        answer = ''
        # global tagstate, date_tags, loc_tags, answer_date

        if question == '초기화':
            self.tagstate = 'N'
            self.date_tags = []
            self.loc_tags = []
            self.answer_date = []

        labels, tokens = self.ner_inference(question)

        for i, label in enumerate(labels):
            if self.tagstate == 'N':
                if label.find('DAT') >= 0:
                    self.answer_date.append(tokens[i])
                    self.date_tags.append(urllib.parse.quote(tokens[i]))
                if label.find('LOC') >= 0:
                    self.loc_tags.append(urllib.parse.quote(tokens[i]))
            elif self.tagstate == 'L':
                self.tagstate = 'N'
                if label.find('LOC') >= 0:
                    self.loc_tags.append(urllib.parse.quote(tokens[i]))
            elif self.tagstate == 'D':
                self.tagstate = 'N'
                if label.find('DAT') >= 0:
                    self.answer_date.append(tokens[i])
                    self.date_tags.append(urllib.parse.quote(tokens[i]))
            elif self.tagstate == 'F':
                if label.find('DAT') >= 0:
                    self.answer_date = []
                    self.date_tags = []
                    self.answer_date.append(tokens[i])
                    self.date_tags.append(urllib.parse.quote(tokens[i]))
                if label.find('LOC') >= 0:
                    self.loc_tags = []
                    self.loc_tags.append(urllib.parse.quote(tokens[i]))

        if (len(self.loc_tags) == 0) & (len(self.date_tags) == 0):
            self.tagstate = 'N'
            answer = '언제 어디 날씨를 알고 싶으세요?'
            return answer
        elif len(self.loc_tags) == 0:
            self.tagstate = 'L'
            answer = '어디 날씨를 알고 싶으신가요?'
            return answer
        elif len(self.date_tags) == 0:
            self.tagstate = 'D'
            answer = '언제 날씨를 알고 싶으세요?'
            return answer
        else:
            date_loc = '+'.join(self.date_tags) + '+' + '+'.join(self.loc_tags) + '+' + urllib.parse.quote('날씨')

            url = 'https://www.google.com/search?hl=ko&sxsrf=ALeKk03fvQH8AdR9Iudhkj2zC31YuobOFQ%3A1595310234185&source=hp&ei=moAWX8TdCM-Yr7wPl9OgcA&q=' + date_loc
            options = wd.ChromeOptions()
            options.add_argument('headless')
            options.add_argument('window-size=1920x1080')
            options.add_argument("disable-gpu")
            # 혹은 options.add_argument("--disable-gpu")
            driver = wd.Chrome('./ner/driver/chromedriver.exe', options=options)

            driver.get(url)
            soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
            try:
                loc = soup.find('div', id='wob_loc').text
                dts = soup.find('div', id='wob_dts').text
                dc = soup.find('span', id='wob_dc').text
                ttm = soup.find('span', id='wob_tm').text
                pp = soup.find('span', id='wob_pp').text
                hm = soup.find('span', id='wob_hm').text
                ws = soup.find('span', id='wob_ws').text

                if int(pp.replace('%', '')) > 30:
                    answer += self.get_weather_form(loc, dts, ttm, pp, hm, ws, dc, 2)
                else:
                    answer += self.get_weather_form(loc, dts, ttm, pp, hm, ws, dc, 1)
                self.tagstate = 'F'
                driver.close()
                return answer
            except:
                answer = '정확하게 요청해주세요.'
                self.date_tags = []
                self.loc_tags = []
                self.answer_date = []
                driver.close()
                return answer

    # 다음 맵에서 맛집 검색을 기준으로 가져오는 크롤링 함수 선언
    def get_food(self, question):

        self.food_loc_tags = []
        self.food_nm_tags = []

        labels, tokens = self.ner_inference(question)

        for i, label in enumerate(labels):
            if label.find('LOC') >= 0:
                self.food_loc_tags.append(tokens[i])
            if label.find('FOD') >= 0:
                self.food_nm_tags.append(tokens[i])

        j_food_loc = ' '.join(self.food_loc_tags)
        j_food_nm = ' '.join(self.food_nm_tags)

        if (len(j_food_loc) > 0) | (len(j_food_nm) > 0):
            options = wd.ChromeOptions()
            options.add_argument('headless')
            options.add_argument('window-size=1920x1080')
            options.add_argument("disable-gpu")
            options.add_argument("disable-gpu")  # 가속 사용 x
            options.add_argument("lang=ko_KR")  # 가짜 플러그인 탑재
            options.add_argument(
                'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')

            # 혹은 options.add_argument("--disable-gpu")
            # colab에서 사용시 kora.selenium를 사용하셔도 됩니다.
            # 자신의 크롬드라이브 경로에 따라 경로를 변경해주시면됩니다.
            driver = wd.Chrome('./ner/driver/chromedriver.exe', options=options)
            driver.implicitly_wait(20)
            driver.get('https://map.kakao.com/')

            loc = j_food_loc
            food = j_food_nm
            r_key = loc + " " + food
            driver.find_element_by_xpath("""/html""").click()
            driver.find_element_by_xpath("""//*[@id="search.keyword.query"]""").clear()
            driver.find_element_by_xpath("""//*[@id="search.keyword.query"]""").send_keys(r_key)
            driver.find_element_by_xpath("""//*[@id="search.keyword.submit"]""").click()

            results = []

            time.sleep(0.1)
            for store in driver.find_elements_by_css_selector("li.PlaceItem"):
                # 세부 데이터 수집
                name = store.find_element_by_css_selector("a.link_name").text
                addr = store.find_element_by_css_selector("p.lot_number").text
                point = store.find_element_by_css_selector("em.num").text
                if store.find_element_by_css_selector("span.phone") is None:
                    phone = "전화번호없음"
                else:
                    phone = store.find_element_by_css_selector("span.phone").text
                results.append({"이름": name, "주소": addr, "전화번호": phone, "평점": point})
            df_results = pd.DataFrame(results)
            df_results.sort_values('평점', ascending=False, inplace=True)
            df_results = df_results.reset_index()

            df_results = df_results.head(3)
            self.rtn_food = "<table width='80%' height='80%' style='margin-bottom:2px;border-color:#9ABAD9;background-color:snow;padding:5px 5px 5px 5px;'>" \
                            '   <tr>' \
                            '       <th>이름</th>' \
                            '       <th>주소</th>' \
                            '       <th>전화번호</th>' \
                            '       <th>평점</th>' \
                            '   </tr>'
            for i, nm, ad, pn, sc in df_results.values:
                self.rtn_food += '<tr>' \
                                 '  <td>' + nm + '</td>' \
                                                 '  <td>' + ad + '</td>' \
                                                                 '  <td>' + pn + '</td>' \
                                                                                 '  <td>' + sc + '</td>' \
                                                                                                 '</tr>'
            self.rtn_food += '</table>'
            return self.rtn_food

        else:
            return '검색 정보가 없습니다. 다시 요청해주세요.'

    # 날씨정보에 대해서 별도 포맷을 만들어주는 함수 선언
    def get_weather_form(self, loc, dts, ttm, pp, hm, ws, dc, w_type):
        print('w_type=', w_type)
        if w_type == 1:
            self.rtn_html = "<table width='80%' height='80%' style='margin-bottom:2px;border-color:#9ABAD9;background-color:snow;padding:5px 5px 5px 5px;'><tr><td colspan='2'><b><font size='4'>" + loc + "</font></b></td></tr><tr><td colspan='2'><b><font size='3' color='gray'>" + dts + "</font></b></td></tr><tr><td rowspan='2'><img style='float:left;height:64px;width:64px' src='./ner/img/" + dc + ".png' data-atf='1'></td><td><font size='3'>" + dc + "</font></td></tr><tr><td><b><font size='5'>" + ttm + "</font> <font color='red'>°C</font></b></td></tr><tr><td colspan='2'><ul style='list-style:none;'><li><font size='3' color='gray'>강수확률 : " + pp + "</font></li><li><font size='3' color='gray'>습도 : " + hm + "</font></li><li><font size='3' color='gray'>풍속 : " + ws + "</font></li></ul></td></tr></table>"
        elif w_type == 2:
            self.rtn_html = "<table width='80%' height='80%' style='margin-bottom:2px;border-color:#9ABAD9;background-color:snow;padding:5px 5px 5px 5px;'><tr><td colspan='2'><b><font size='4'>" + loc + "</font></b></td></tr><tr><td colspan='2'><b><font size='3' color='gray'>" + dts + "</font></b></td></tr><tr><td rowspan='2'><img style='float:left;height:64px;width:64px' src='./ner/img/" + dc + ".png' data-atf='1'></td><td><font size='3'>" + dc + "</font></td></tr><tr><td><b><font size='5'>" + ttm + "</font> <font color='red'>°C</font></b></td></tr><tr><td colspan='2'><ul style='list-style:none;'><li><font size='3' color='gray'>강수확률 : " + pp + "</font></li><li><font size='3' color='gray'>습도 : " + hm + "</font></li><li><font size='3' color='gray'>풍속 : " + ws + "</font></li></ul></td></tr><tr><td colspan='2'> Tip. 우산 챙기시는게 좋을거 같습니다. </td></tr></table>"
        else:
            self.rtn_html = "<table width='80%' height='80%' style='margin-bottom:2px;border-color:#9ABAD9;background-color:snow;padding:5px 5px 5px 5px;'><tr><td colspan='2'><b><font size='4'>" + loc + "</font></b></td></tr><tr><td colspan='2'><b><font size='3' color='gray'>" + dts + "</font></b></td></tr><tr><td rowspan='2'><img style='float:left;height:64px;width:64px' src='./ner/img/" + dc + ".png' data-atf='1'></td><td><font size='3'>" + dc + "</font></td></tr><tr><td><b><font size='5'>" + ttm + "</font> <font color='red'>°C</font></b></td></tr><tr><td colspan='2'><ul style='list-style:none;'><li><font size='3' color='gray'>강수확률 : " + pp + "</font></li><li><font size='3' color='gray'>습도 : " + hm + "</font></li><li><font size='3' color='gray'>풍속 : " + ws + "</font></li></ul></td></tr></table>"

        return self.rtn_html
