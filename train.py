import tensorflow as tf
from data_process import Data_process
from textcnn_model import TextCNN
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score

flags = tf.app.flags
flags.DEFINE_boolean("is_train",       True,      "clean train folder")
flags.DEFINE_integer("batch_size",    128,        "batch_size")
flags.DEFINE_integer("epoch",    20,        "epoch")
flags.DEFINE_integer("word2vev_size",    300,        "length of word2vec")
flags.DEFINE_integer("numfilter",    128,        "numfilter")

# 若文本长度大于300则截断
flags.DEFINE_integer("max_length",    300,        "text max_length")
# 取词频大于10的
flags.DEFINE_integer("min_counter",    10,        "min_counter")
flags.DEFINE_integer("numclass",    3,        "numclass")
flags.DEFINE_string("path_input",   "./data/train_label.csv",    "input file path")
flags.DEFINE_string("path_stopword",   "./data/stopword.txt",    "stopword file path")
flags.DEFINE_string("path_model",   "./model/model.ckpt",    "model file path")
flags.DEFINE_string("path_word2vec_model",   "./word2vec/word2vec_words_final.model",    "word2vec_model file path")
flags.DEFINE_float("learnrate",          0.0001,          "rate")
# 将文本分为训练集核验证集的比例
flags.DEFINE_float("rate",          0.8,          "rate")
flags.DEFINE_float("dropoutrate",          0.5,          "dropout")
FLAGS = tf.app.flags.FLAGS

def train():
    dataprocess = Data_process(FLAGS.path_input,FLAGS.path_stopword,FLAGS.path_word2vec_model,
                               FLAGS.word2vev_size,FLAGS.max_length,FLAGS.min_counter,FLAGS.rate)
    trainReviews, trainLabels, evalReviews, evalLabels, wordembedding = dataprocess.dataGen()
    with tf.Graph().as_default():
        cnn = TextCNN(vocab_size=len(wordembedding),Filter_size=[2,3,4],embedding=wordembedding,
                      numFilters=FLAGS.numfilter,max_length=FLAGS.max_length,dropoutKeepProb=FLAGS.dropoutrate,
                      numClass=FLAGS.numclass)
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learnrate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器，对参数进行更新
        saver = tf.train.Saver()
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            recall_max = 0
            for i in range(FLAGS.epoch):
                for batch in dataprocess.nextBatch(trainReviews,trainLabels,FLAGS.batch_size):

                    feed_dict = {
                        cnn.input_X: batch[0],
                        cnn.input_Y: batch[1]
                    }
                    predictions,loss,_,ouput,step = sess.run([cnn.predictions,cnn.loss,trainOp,cnn.output,globalStep],
                                                        feed_dict)
                    acc = accuracy_score(batch[1], ouput)
                    precision = precision_score(batch[1], ouput, average='weighted')
                    recall = recall_score(batch[1], ouput, average='micro')
                    timeStr = datetime.datetime.now().isoformat()
                    print("{}, iter: {}, step: {}, loss: {},acc: {}, precision: {}, recall: {}"
                          .format(timeStr, i, step, loss, acc, precision, recall))
                acces = []
                precisiones = []
                recalles = []
                for batch_eva in dataprocess.nextBatch(evalReviews, evalLabels, FLAGS.batch_size):

                    loss, output = sess.run([cnn.loss, cnn.output], feed_dict={
                        cnn.input_X: batch_eva[0],
                        cnn.input_Y: batch_eva[1]
                    })
                    acc = accuracy_score(batch_eva[1], ouput)
                    precision = precision_score(batch_eva[1], ouput, average='weighted')
                    recall = recall_score(batch_eva[1], ouput, average='micro')
                    acces.append(acc)
                    precisiones.append(precision)
                    recalles.append(recall)
                acc = sum(acces)/len(acces)
                precision = sum(precisiones)/len(precisiones)
                recall = sum(recalles)/len(recalles)
                print("验证集结果：")
                print("{}, iter: {}, loss: {},acc: {}, precision: {}, recall: {}"
                      .format(timeStr, i, loss, acc, precision, recall))
                if recall > recall_max:
                    recall_max = recall
                    print("正在保存模型")
                    saver.save(sess, FLAGS.path_model, global_step=step)

if __name__ == "__main__":
    train()