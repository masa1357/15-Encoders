import pandas as pd
from pathlib import Path
from logging import getLogger, StreamHandler, Formatter
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import re
from typing import Dict, List

#---
import MeCab
import re
from gensim.models import Word2Vec
import numpy as np
import logging
from tqdm import tqdm

class GetSentenceVector:
    def __init__(self, args:Dict[str, object], corpus_df:pd.DataFrame):
        self.LABEL_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
        self.logger = args["logger"].getChild(__name__)
        MODEL_PATH = args["MODEL_PATH"]
        self.MODEL_PATH = MODEL_PATH/"word2vec_testcase.model"
        self.tagger = MeCab.Tagger("-Owakati")
        # word2vec parameters
        self.num_features = 200
        self.min_word_count = 5
        self.num_workers = 40
        self.context = 10
        self.downsampling = 1e-3
        self.model_name = "word2vec_testcase.model"
        self.corpus = []

        #INFO: 最初にモデルを準備しておく
        # コーパス読み込み
        for doc in tqdm(corpus_df["text"]):
            self.corpus.append(self.make_wakati(doc))
        self.train_model()

    def __call__(self, plot_df:pd.DataFrame,key:str) -> pd.DataFrame:

        # print(len(plot_df[key]))
        self.logger.debug(f"processing {key} ...")
        X, Y, texts = [], [], []
        for doc, category in tqdm(zip(plot_df[key], plot_df["label"])):
            wakati = self.make_wakati(doc)
            docvec = self.wordvec2docvec(wakati)
            X.append(list(docvec))
            Y.append(category)
            texts.append(doc)
        data_X = pd.DataFrame(X, columns=["X" + str(i + 1) for i in range(self.num_features)])
        data_Y = pd.DataFrame(Y, columns=["category_id"])
        # data_texts = pd.DataFrame(texts, columns=["text"])
        # data_X["text"] = text
        # data = pd.concat([plot_df, data_X, data_Y], axis=1)
        return data_X, data_Y, texts


#-------------------------------------------------------------------------------------
# common functions
#-------------------------------------------------------------------------------------
    def train_model(self):
        # word2vecモデルの作成＆モデルの保存
        # logging.basicConfig(
        #     format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        # )

        if self.MODEL_PATH.is_file():
            self.logger.info("loading word2vec model ...")
            self.model = Word2Vec.load(str(self.MODEL_PATH))
            self.logger.info("model loaded.")
            return
        else:
            self.logger.info("cleating word2vec model ...")
            self.model = Word2Vec(
                self.corpus,
                workers=self.num_workers,
                hs=1,
                sg=1,
                negative=10,
                epochs=25,
                vector_size=self.num_features,
                min_count=self.min_word_count,
                window=self.context,
                sample=self.downsampling,
                seed=1,
            )
            self.model.save(self.model_name)
            self.logger.info("Done.")


    def make_wakati(self, sentence):
        sentence = self.tagger.parse(sentence)
        sentence = re.sub(r"[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+", " ", sentence)
        sentence = re.sub(
            r"[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=＝)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+",
            "",
            sentence,
        )
        wakati = sentence.split(" ")
        wakati = list(filter(("").__ne__, wakati))
        return wakati
    
    def wordvec2docvec(self, sentence):
        # 文章ベクトルの初期値（0ベクトルを初期値とする）
        docvecs = np.zeros(self.num_features, dtype="float32")

        # 文章に現れる単語のうち、モデルに存在しない単語をカウントする
        denomenator = len(sentence)

        # 文章内の各単語ベクトルを足し合わせる
        for word in sentence:
            try:
                temp = self.model.wv[word]
                # docvecs += temp
            except:
                denomenator -= 1
                # print(f"{word}はモデルに存在しません。")
                continue
            docvecs += temp

        # 文章に現れる単語のうち、モデルに存在した単語の数で割る
        if denomenator > 0:
            docvecs = docvecs / denomenator

        return docvecs
    



