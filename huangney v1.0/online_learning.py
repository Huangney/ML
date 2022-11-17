import numpy.ma
import tensorflow as tf
from tensorflow import keras
import functions
import os
from model_framework import one_loss_function, Encoder, Decoder


def uni_train(inp_senten, outp_senten, items, lr=False):
    print("可以可以")
    steps_loss = 0.0

    origin_word_size, encoder, decoder = items[0], items[1], items[2]
    limit, tok, embedding_dim, word_index = items[3], items[4], items[5], items[6]

    word_size = origin_word_size
    inp_senten, word_size, word_index = functions.splindex_sent_foruni(inp_senten, word_index, word_size)
    inp_senten = keras.preprocessing.sequence.pad_sequences([inp_senten],
                                                            maxlen=limit,
                                                            padding="post")

    outp_senten, new_word_size, word_index = functions.splindex_sent_foruni(outp_senten, word_index, word_size)
    outp_senten = keras.preprocessing.sequence.pad_sequences([outp_senten],
                                                             maxlen=limit,
                                                             padding="post")
    # print("your_new_output: ",outp_senten)

    added_num = new_word_size - origin_word_size
    my_epochs = 3

    if added_num:
        my_epochs = 1
        print("还有新词学啊")
        optimizer = keras.optimizers.Adam(learning_rate=1e-4 * 2)
        decoder_den = decoder.get_layer(index=-2).get_weights()
        print(decoder_den)
        # print("decoder_dense[1].shape:", decoder_den[1].shape)

        # extra_weights_for_dense = numpy.zeros((decoder.decoding_units, added_num))
        extra_weights_for_dense = numpy.random.randn(decoder.decoding_units, added_num)
        extra_weights_for_dense = numpy.clip(extra_weights_for_dense, -5, 5)
        extra_weights_for_dense = extra_weights_for_dense *0.01
        print(extra_weights_for_dense)
        # extra_weights_for_dense.fill(0.05)
        decoder_den[0] = numpy.hstack([decoder_den[0], extra_weights_for_dense])
        decoder_den[1] = numpy.hstack([decoder_den[1], numpy.zeros((added_num))])
        # print("decoder_dense[1].shape:", decoder_den[1].shape)

        # print("decoder_dense.shape:", decoder_den[0].shape)
        # print("it's new word_size", new_word_size)
        # print("decoder_dense:", decoder_den[0])

        decoder.Dense_out = keras.layers.Dense(new_word_size, weights=decoder_den, activation="softmax")

        mimic = numpy.zeros((1, decoder.decoding_units))

        out = decoder.Dense_out(mimic)
        # print("it's the out of :", out)
        # print("dense_shape = ", decoder.Dense_out.variables)

        encoder_embed = encoder.get_layer(index=-1).get_weights()
        # print(encoder_embed[0].shape)
        decoder_embed = decoder.get_layer(index=-2).get_weights()
        print(decoder_embed[0])

        extra_weights_for_embedding = numpy.random.randn(added_num, embedding_dim)
        extra_weights_for_embedding = numpy.clip(extra_weights_for_embedding, -5, 5)
        extra_weights_for_embedding = extra_weights_for_embedding * 0.01

        encoder_embed[0] = numpy.vstack([encoder_embed[0], extra_weights_for_embedding])
        decoder_embed[0] = numpy.vstack([decoder_embed[0], extra_weights_for_embedding])
        # print(encoder_embed[0].shape)
        # print(encoder_embed[0])

        encoder.embedding = keras.layers.Embedding(new_word_size, 300, weights=encoder_embed, trainable=True)
        decoder.embedding = keras.layers.Embedding(new_word_size, 300, weights=decoder_embed, trainable=True)
    else:
        optimizer = keras.optimizers.Adam()
    if lr:
        my_epochs *= 2
    # inp_senten = tf.expand_dims(tf.convert_to_tensor(inp_senten),0)
    inp_senten = tf.convert_to_tensor(inp_senten)
    outp_senten = tf.convert_to_tensor(outp_senten)
    # print(inp_senten)
    # print(outp_senten)

    for i in range(my_epochs):
        with tf.GradientTape() as tape:
            encoder_hidden = tf.zeros((1, encoder.encoding_units))
            encoding_outputs, encoding_hid = encoder(inp_senten, encoder_hidden)

            decoding_hidden = encoding_hid
            sum_ = 0
            for step in range(outp_senten.shape[1] - 1):
                # 老规矩，切片出来只剩一个维度了，少了一个步长维度，做一下expand
                decoder_input = tf.expand_dims(outp_senten[:, step], 1)
                predic, decoder_hidden, attention_weights = decoder(decoder_input,
                                                                    decoding_hidden,
                                                                    encoding_outputs)
                now_loss = one_loss_function(outp_senten[:, step + 1], predic)
                if(now_loss > 0.00001):
                    sum_ += 1
                steps_loss += now_loss

            variables = encoder.trainable_variables + decoder.trainable_variables
            grad = tape.gradient(steps_loss, variables)
            optimizer.apply_gradients(zip(grad, variables))
            steps_loss /= sum_

    steps_loss = steps_loss * 10 / my_epochs
    model_path = "./models"
    enmodel_name = "encoder_weights"
    demodel_name = "decoder_weights"
    en_path = os.path.join(model_path, enmodel_name)
    de_path = os.path.join(model_path, demodel_name)
    encoder.save_weights(en_path)
    decoder.save_weights(de_path)
    if lr:
        print("快速懂起了捏")
    else:
        print("懂起了捏")

    return encoder, decoder, steps_loss, new_word_size, word_index
