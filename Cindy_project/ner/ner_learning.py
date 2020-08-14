# 라이브러리
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import gzip
import logging
import os
import unicodedata
from shutil import copyfile
from transformers import PreTrainedTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# 데이터셋 로드
# 자신의환경에 맞게 경로를 수정하셔야 합니다.
# base dataset 참조 ( https://github.com/naver/nlp-challenge/ )
train = pd.read_csv('./dataset/Colab Data/ner_dataset(LOC update).csv',index_col=0)
weather = pd.read_csv('./dataset/ner_weather.csv',index_col=0)
news = pd.read_csv('./dataset/ner_news.csv', index_col=0)
food = pd.read_csv('./dataset/ner_food.csv',index_col=0)
train = pd.concat([train,weather,news,food])

# 데이터를 전처리
# 한글, 영어, 소문자, 대문자, . 이외의 단어들을 모두 제거
train['src'] = train['src'].str.replace("．", ".", regex=False)
train.loc[train['src']=='.']
train['src'] = train['src'].astype(str)
train['tar'] = train['tar'].astype(str)
train['src'] = train['src'].str.replace(r'[^ㄱ-ㅣ가-힣0-9a-zA-Z.]+', "", regex=True)

# 데이터를 (인덱스, 단어, 개체) 로 이루어 진 리스트로 변환.
# 인덱스가 1,2,3,4,5.. 이렇게 이어지다가 다시 1,2,3,4, 로 바뀌는데 숫자가 바뀌기 전까지가 한 문장을 의미합니다.
data = [list(x) for x in train[['index', 'src', 'tar']].to_numpy()]

# ner 사전 생성
label = train['tar'].unique().tolist()
label_dict = {word:i for i, word in enumerate(label)}
label_dict.update({"[PAD]":len(label_dict)})
index_to_ner = {i:j for j, i in label_dict.items()}

# 생성한 사전 pickle로 저장
with gzip.open('./vocab/label_dict.pickle', 'wb') as f:
    pickle.dump(label_dict, f)
with gzip.open('./vocab/index_to_ner.pickle', 'wb') as f:
    pickle.dump(index_to_ner, f)

# 데이터를 문장들과 개체들로 분리
tups = []
temp_tup = []
sentences = []
targets = []
for i, j, k in data:

    if i != 1:
        temp_tup.append([j, label_dict[k]])
    if i == 1:
        if len(temp_tup) != 0:
            tups.append(temp_tup)
            temp_tup = []
            temp_tup.append([j, label_dict[k]])

for tup in tups:
  sentence = []
  target = []
  sentence.append("[CLS]")
  target.append(label_dict['-'])
  for i, j in tup:
    sentence.append(i)
    target.append(j)
  sentence.append("[SEP]")
  target.append(label_dict['-'])
  sentences.append(sentence)
  targets.append(target)


######################################################
# tokenizer(sentencepiece)
# monologg님의 소스를 활용했습니다.
######################################################
logger = logging.getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer_78b3253a26.model",
                     "vocab_txt": "vocab.txt"}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "monologg/kobert": "./vocab/tokenizer_78b3253a26.model",
    },
    "vocab_txt": {
        "monologg/kobert": "./vocab/vocab.txt",
    }
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "monologg/kobert": 512,
}
PRETRAINED_INIT_CONFIGURATION = {
    "monologg/kobert": {"do_lower_case": False},
}
SPIECE_UNDERLINE = u'▁'

class KoBertTokenizer(PreTrainedTokenizer):
    """
        SentencePiece based tokenizer. Peculiarities:
            - requires `SentencePiece <https://github.com/google/sentencepiece>`_
    """
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
        print(self.token2idx)
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
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
            logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
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

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
######################################################
######################################################

# 문장을 토크나이징하고 개체를 토크나이징 한 문장에 맞추는 함수선언
def tokenize_and_preserve_labels(sentence, text_labels):
  tokenized_sentence = []
  labels = []

  for word, label in zip(sentence, text_labels):

    tokenized_word = tokenizer.tokenize(word)
    n_subwords = len(tokenized_word)

    tokenized_sentence.extend(tokenized_word)
    labels.extend([label] * n_subwords)

  return tokenized_sentence, labels


tokenized_texts_and_labels = [
                              tokenize_and_preserve_labels(sent, labs)
                              for sent, labs in zip(sentences, targets)]

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

# 문장의 길이가 상위 2.5%(72) 인 지점을 기준으로 문장의 길이 지정
# 만약 문장의 길이가 72보다 크면 문장이 잘리게 되고, 길이가 72보다 작다면 패딩이 되어 모든 문장의 길이가 72로 정해지게 됩니다.
print(np.quantile(np.array([len(x) for x in tokenized_texts]), 0.975))
max_len = 72
bs = 32

# 버트에 인풋으로 들어갈 train 데이터를 만들도록 하겠습니다.
# 버트 인풋으로는
# input_ids : 문장이 토크나이즈 된 것이 숫자로 바뀐 것,
# attention_masks : 문장이 토크나이즈 된 것 중에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 마스킹
# [input_ids, attention_masks]가 인풋으로 들어갑니다.
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype = "int", value=tokenizer.convert_tokens_to_ids("[PAD]"), truncating="post", padding="post")

# 정답(target)에 해당하는 개체 생성
tags = pad_sequences([lab for lab in labels], maxlen=max_len, value=label_dict["[PAD]"], padding='post',\
                     dtype='int', truncating='post')

# 어텐션 마스크 생성
# 실제 문장은 1 아닌부분은 0으로 생성
attention_masks = np.array([[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in input_ids])

# validation 데이터 생성
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


# 모델 생성
# TPU 작동을 위해 실행
# colab에서 사용시 주석해제
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)

SEQ_LEN = max_len
def create_model():
    model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True, num_labels=len(label_dict),
                                        output_attentions=False,
                                        output_hidden_states=False)

    token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')  # 토큰 인풋
    mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')  # 마스크 인풋

    bert_outputs = model([token_inputs, mask_inputs])
    bert_outputs = bert_outputs[0]  # shape : (Batch_size, max_len, 30(개체의 총 개수))
    nr = tf.keras.layers.Dense(31, activation='softmax')(bert_outputs)  # shape : (Batch_size, max_len, 30)

    nr_model = tf.keras.Model([token_inputs, mask_inputs], nr)

    nr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00002),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics=['sparse_categorical_accuracy'])
    nr_model.summary()
    return nr_model

# TPU를 활용하기 위해 context로 묶어주기
# colab tpu 사용시
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
# with strategy.scope():
#   nr_model = create_model()
#   nr_model.fit([tr_inputs, tr_masks], tr_tags, validation_data=([val_inputs, val_masks], val_tags), epochs=3, shuffle=False, batch_size=bs)

# 만약 TPU를 사용하지 않고 GPU를 사용한다면
nr_model = create_model()
nr_model.fit([tr_inputs, tr_masks], tr_tags, validation_data=([val_inputs, val_masks], val_tags), epochs=3, shuffle=False, batch_size=bs)

# 모델 가중치 저장
# 자신의환경에 맞게 경로를 수정하셔야 합니다.
nr_model.save_weights('./model/ner_class_model_weight.h5')

# 예측 결과 확인
y_predicted = nr_model.predict([val_inputs, val_masks])
f_label = [i for i, j in label_dict.items()]
val_tags_l = [index_to_ner[x] for x in np.ravel(val_tags).astype(int).tolist()]
y_predicted_l = [index_to_ner[x] for x in np.ravel(np.argmax(y_predicted, axis=2)).astype(int).tolist()]
f_label.remove("[PAD]")
print(classification_report(val_tags_l, y_predicted_l, labels=f_label))

# 실제 데이터로 확인하기위한 함수 선언
def ner_inference(test_sentence):
    # tokenized_sentence = np.array([tokenizer.encode(test_sentence, max_length=max_len, pad_to_max_length=True)])
    tokenized_sentence = np.array(
        [tokenizer.encode(test_sentence, max_length=max_len, pad_to_max_length=True, truncation=True)])
    tokenized_mask = np.array([[int(x != 1) for x in tokenized_sentence[0].tolist()]])
    ans = nr_model.predict([tokenized_sentence, tokenized_mask])
    ans = np.argmax(ans, axis=2)

    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, ans[0]):

        if (token.startswith("▁")):
            new_labels.append(index_to_ner[label_idx])
            new_tokens.append(token[1:])
        elif (token == '[CLS]'):
            pass
        elif (token == '[SEP]'):
            pass
        elif (token == '[PAD]'):
            pass
        elif (token != '[CLS]' or token != '[SEP]'):
            new_tokens[-1] = new_tokens[-1] + token

    for token, label in zip(new_tokens, new_labels):
        print("{}\t{}".format(label, token))


# 예시
ner_inference('7월 둘째주 날씨 알려줄수있어?')
# 결과 예시
# DAT_B	7월
# DAT_I	둘째주
# WTH_B	날씨
# -	알려줄수있어?
# 질의 문장의 각 개체에 대해 태깅이 되는 모습입니다.