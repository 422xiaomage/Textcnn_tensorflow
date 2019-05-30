import tensorflow as tf

class TextCNN(object):
    def __init__(self, vocab_size,Filter_size,embedding=None, embedding_size=300, numFilters=128, max_length=300,
                 dropoutKeepProb=0.5,numClass=3):
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.Filter_size = Filter_size
        self.numFilters = numFilters
        self.max_length = max_length
        self.dropoutKeepProb = dropoutKeepProb
        self.vocab_size = vocab_size
        self.numClass = numClass


        # 词嵌入层
        self.input_X = tf.placeholder(tf.int32, [None, self.max_length], name="input_X")
        self.input_Y = tf.placeholder(tf.int32, [None, 1], name="input_Y")
        with tf.name_scope("embedding"):
            if self.embedding is not None:
                # 这里将词向量设置成了变量，所以在训练过程中，也会对原始词向量进行微调，如果不需要，
                # 可以将这一句修改为注释掉的代码段
                self.W = tf.Variable(tf.cast(self.embedding, dtype=tf.float32,name="word2vec"), name="W")
                # self.W = tf.constant(tf.cast(self.embedding, dtype=tf.float32,name="word2vec"), name="W")
            else:
                self.W = tf.Variable(tf.truncated_normal([self.vocab_size,self.embedding_size],stddev=1,
                                                         dtype=tf.float32), name="W")
                # self.W = tf.constant(tf.truncated_normal([self.vocab_size,self.embedding_size],stddev=1,
                # dtype=tf.float32), name="W")
            # 词序号的向量化操作
            self.embeddingwords = tf.nn.embedding_lookup(self.W, self.input_X)
            # 这里由于卷积层输入的是一个四维的向量，第四维时通道，所以这里要扩展一维，
            # 扩展成[batch_size, width, height, channel]
            self.embeddingwords_expand = tf.expand_dims(self.embeddingwords, -1)
            # 对标签进行onne_hot编码,这里要注意self.input_Y的数据类型必须时int型
            self.input_Y_one_hot = tf.cast(tf.one_hot(self.input_Y, self.numClass, name="Y_onehot"), dtype=tf.float32)
        # 卷积层和池化层
        pooledOutputs = []
        for i, filtersize in enumerate(self.Filter_size):

            with tf.name_scope("conv-maxpool-%s" % filtersize):
                # 卷积层，卷积核的尺寸[filtersize,self.embedding_size],卷积核的个数是self.numFilters,这个是超参
                # 初始化权重矩阵和偏置
                # 第三维1是通道数量，对于文本来说通道一定是1，所以不可以更改
                W = tf.Variable(tf.truncated_normal([filtersize, self.embedding_size, 1, self.numFilters],
                                                    stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1,shape=[self.numFilters]), name="b")
                conv = tf.nn.conv2d(
                    self.embeddingwords_expand,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                covn_plus_b = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    covn_plus_b,
                    ksize=[1, self.max_length-filtersize+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                pooledOutputs.append(pooled)
        # 池化后的维度不变，按照最后的维度channel来concat
        self.hPool = tf.concat(pooledOutputs, 3)
        # cnn输出的长度为卷积核的种类数*每种卷积核的个数
        flat_length = len(self.Filter_size) * self.numFilters
        self.hPoolFlat = tf.reshape(self.hPool, [-1, flat_length])

        # dropout层
        with tf.name_scope("drop_out"):
            self.hPoolFlat_dropout = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)

        # 定义全连接层
        with tf.name_scope("output"):
            output_W = tf.get_variable(
                "output_W",
                shape=[flat_length, self.numClass],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.numClass]), name="output_b")

            self.predictions = tf.nn.xw_plus_b(self.hPoolFlat_dropout, output_W, output_b, name="predictions")

            self.output = tf.cast(tf.arg_max(self.predictions, 1), tf.float32, name="category")
        # 计算三元交叉熵损失
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions,
                                                                               labels=self.input_Y_one_hot))
        # 优化器
