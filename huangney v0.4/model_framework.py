import tensorflow as tf
from tensorflow import keras
import time

def tf_datas(a,batch_size = 8,shuffle = True,epochs = 5):
    dataset = tf.data.Dataset.from_tensor_slices((list(a["inp"]),list(a["outp"])))
    sum_ = 0
    if shuffle:
        dataset = dataset.shuffle(1000)
    for i in dataset.take(-1):
        sum_ += 1
    print(sum_)
    sum_ = 0
    dataset = dataset.repeat(epochs).batch(batch_size,drop_remainder = True)
    if shuffle:
        dataset = dataset.shuffle(1000)
    for i in dataset.take(-1):
        sum_ += 1
    print(sum_)
    sum_ = 0
    return dataset
# 将先前的pandas DataFrame文件变换为tfDataSet

class Encoder(keras.Model):
    def __init__(self, word_size, embedding_dim, encoding_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoding_units = encoding_units

        self.embedding = keras.layers.Embedding(word_size, embedding_dim, trainable=True)
        self.gru = keras.layers.GRU(encoding_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        #         print("这是EMbedding的第一个结果: ",x[0])
        output, state = self.gru(x, initial_state=hidden)

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoding_units))

# 编码器

class Attention(keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.units = units
        self.Dense_encoder_output = keras.layers.Dense(units)
        self.Dense_decoder_hidden = keras.layers.Dense(units)
        self.Dense_gather = keras.layers.Dense(1)

    def call(self, encoder_output, decoder_hidden):
        # decoder的隐藏状态张量是[batch_size,decoder_units]形状，因为其只含有当前这一个步长的输出
        # encoder的输出张量是[batch_size,sequence_length,encoder_units]形状，其包含所有步长的输出
        decoder_hidden_expan = tf.expand_dims(decoder_hidden, 1)

        # 此时gather的形状为[batch_size,length,1]
        gather = self.Dense_gather(
            tf.nn.tanh(
                self.Dense_encoder_output(encoder_output) + self.Dense_decoder_hidden(decoder_hidden_expan)))

        # 获得一个形状为[batch_size,length,1]的张量，每一个句子中的每一个词都有了一个权重
        attention_weights = tf.nn.softmax(gather, axis=1)

        # 获得加权后的句子，能知道那个词语是当前最应被注意到的,shape = [bcsz, length, units]
        context_vec = attention_weights * encoder_output

        # 最终输出中，权重最大的对它影响最多
        context_score = tf.reduce_sum(context_vec, axis=1)

        return context_score, attention_weights
# Attention机制实现

class Decoder(keras.Model):
    def __init__(self, word_size, embedding_dim, decoding_units, batch_size):
        super(Decoder, self).__init__()
        self.decoding_units = decoding_units
        self.batch_size = batch_size
        self.embedding = keras.layers.Embedding(word_size, embedding_dim, trainable=True)

        self.gru = keras.layers.GRU(decoding_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        self.Dense_out = keras.layers.Dense(word_size, activation='softmax')
        self.attention_machanism = Attention(decoding_units)

    def call(self, x, hidden, encoder_output):
        attention_score, attention_weights = self.attention_machanism(encoder_output, hidden)

        # 嵌入之前的X：[batch_size,1]，因为每次只取一个步长的数据进入，即每次只吃了一个词
        # 嵌入之后多一个嵌入维度:[batch_size,1,embedding_dim]
        x = self.embedding(x)

        # 而attention_score的形状为[batch_size,1],是与X对应的，但是没有嵌入扩展维度
        attentioned_x = tf.concat([tf.expand_dims(attention_score, 1), x], axis=-1)

        # 输出形状：[batch_size,1,deconding_units](每一步的，因为其实由于return_sequences,还会含有步长)
        output, state = self.gru(attentioned_x)

        # 先规整形状，在做全连接
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.Dense_out(output)

        return output, state, attention_weights
# 解码器

def one_loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits = False,
                                                         reduction = 'none')
    # 生成掩码
    losses = loss_object(real,pred)
    # 用掩码去遮蔽padding造成的损失
    mask = tf.cast(mask,dtype = losses.dtype)
    losses = losses * mask

    return tf.reduce_mean(losses)
# 定义单步损失函数
#%%


@tf.function
def muti_step_loss(inp, targ, encoding_hidden,encoder,decoder,optimizer):

    # initialize loss
    loss = 0
    # 引入梯度机制
    #     print("这是输入给ENCO的：",inp[0])
    #     print("这是输入给ENCO的隐藏状态：",encoding_hidden[:,:5])
    with tf.GradientTape() as tape:
        encoding_output, encoding_state = encoder(inp, encoding_hidden)

        #         print("encoding的隐藏状态：",encoding_state[:,:5])
        decoding_hidden = encoding_state

        # 损失计算自己推一下，长度为5的句子将会计算4词
        # targ[1]是因为targ.shape:[batch_size,sentence_length],我们需要句子长度-1.
        for step in range(0, targ.shape[1] - 1):
            # 批次中所有句子，但只要当前步长,
            # 而且因为是直接切片,所以targ[:,step].shape:[batch_size],显然维度不对
            decoding_input = tf.expand_dims(targ[:, step], 1)

            # 可以拿给decoder预测了
            predictions, decoding_hidden, attetion_weights = decoder(decoding_input,
                                                                     decoding_hidden,
                                                                     encoding_output)
            #             print("输入给DECO的：",targ[:,step])
            #             if tf.compat.v1.is_nan(one_loss_function(targ[:,step+1],predictions)):
            #                 print("\n\n\n\n\n它们导致了nan: targ: ",targ[:,step+1],"predic: ",predictions,"\n\n\n\n\n")
            #             else:
            loss += one_loss_function(targ[:, step + 1], predictions)

        ave_loss = loss / int(targ.shape[0])  # 防止因批量不同引起total_loss数量级不同

        # 计算梯度
        variables = encoder.trainable_variables + decoder.trainable_variables

        grad = tape.gradient(loss, variables)
        # 执行梯度下降
        optimizer.apply_gradients(zip(grad, variables))

        return ave_loss

# 多步长共同计算损失
#%%

def train(a_data,encoder,decoder,epochs,lenth,optimizer):
    batch_size = encoder.batch_size
    steps_spans_epochs = lenth // batch_size
    print("跨步批量数为:", steps_spans_epochs)

    for epoch in range(epochs):
        start = time.time()

        encoding_hid = encoder.initialize_hidden_state()
        total_loss = 0.0

        for (batch_id, (inp, targ)) in enumerate(a_data.take(steps_spans_epochs)):
            #         print(batch_id,":input:",inp,"targ:",targ)
            batch_loss = muti_step_loss(inp, targ, encoding_hid,encoder,decoder,optimizer)
            total_loss += batch_loss

            if batch_id % 10 == 0:
                print(f'迭代：{epoch + 1},批次：{batch_id},批次损失：{batch_loss.numpy()}')

        print(f"迭代：{epoch + 1}，总均损失：{total_loss / steps_spans_epochs}")
        print(f"本次用时{time.time() - start}秒")