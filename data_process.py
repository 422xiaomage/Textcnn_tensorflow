import pandas as pd
import jieba
from gensim import models
import numpy as np
from collections import Counter

class Data_process(object):
    def __init__(self, path_input, path_stopword, path_word2vec_model, embedding_size=300,
                 max_length=300, min_counter=10, rate=0.8):
        self.path_input = path_input
        self.path_stopword = path_stopword
        self.max_length = max_length
        self.path_word2vec_model = path_word2vec_model
        self.min_counter = min_counter
        self.embedding_size = embedding_size
        self.rate = rate
    # 读取csv文件中的数据，并做好分词
    def read_data(self,path):
        fb = pd.read_csv(path)
        reviews = fb["Sentence"].tolist()
        labels = fb["Emotion"].tolist()
        reviews_word = [jieba.lcut(review.strip()) for review in reviews]
        return reviews_word, labels

    def read_stopword(self):
        with open(self.path_stopword, "r", encoding="utf-8") as f:
            lines = f.read()
            stop_word = lines.splitlines()
            self.stopWordDict = dict(zip(stop_word, list(range(len(stop_word)))))

    def get_vocabulary_embedding(self,reviews_word):
        allwords = [word for review in reviews_word for word in review]
        subwords = [word.strip() for word in allwords if word.strip() not in self.stopWordDict]
        wordcounter = Counter(subwords)
        sort_wordcounter = sorted(wordcounter.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sort_wordcounter if item[1] >= self.min_counter]
        vocab = []
        wordembedding = []
        vocab.append("PAD")
        vocab.append("UNK")
        wordembedding.append(np.zeros(self.embedding_size))
        wordembedding.append(np.random.randn(self.embedding_size))
        word2vec = models.Word2Vec.load(self.path_word2vec_model)
        for word in words:
            try:
                vector = word2vec.wv[word]
                vocab.append(word)
                wordembedding.append(vector)
            except:
                print(word+"\t"+"不在训练的词向量中")
        embedding = wordembedding
        self.wordToindex = dict(zip(vocab, list(range(len(vocab)))))
        return np.array(embedding)

    def data_process(self, review,wordToindex):
        reviewVec = np.zeros((self.max_length))
        sequenceLen = self.max_length
        if len(review) < self.max_length:
            sequenceLen = len(review)
        for i in range(sequenceLen):
            if review[i] in wordToindex:
                reviewVec[i] = wordToindex[review[i]]
            else:
                reviewVec[i] = wordToindex["UNK"]
        return reviewVec
    def get_train_evadata(self,x,y,rate):
        reviews_vector = []
        labels = []
        for i in range(len(x)):
            reviewvec = self.data_process(x[i],self.wordToindex)
            reviews_vector.append(reviewvec)
            labels.append([y[i]])
        trainIndex = int(len(x) * rate)
        trainReviews = np.asarray(reviews_vector[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews_vector[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化停用词
        self.read_stopword()

        # 初始化数据集
        reviews, labels = self.read_data(self.path_input)

        # 初始化词汇-索引映射表和词向量矩阵
        embedding = self.get_vocabulary_embedding(reviews)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self.get_train_evadata(reviews, labels, self.rate)
        return trainReviews, trainLabels, evalReviews, evalLabels, embedding

    def nextBatch(self, x, y, batchSize):
        """
        生成batch数据集，用生成器的方式输出
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        numBatches = len(x) // batchSize

        for i in range(numBatches):
            start = i * batchSize
            end = start + batchSize
            batchX = np.array(x[start: end], dtype="int64")
            batchY = np.array(y[start: end], dtype="int64")

            yield batchX, batchY