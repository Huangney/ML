import pickle
import os
import json
import time

from functions import session
from model_framework import Encoder,Decoder
import jieba

def chat():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    tok_name = "tok.pkl"
    json_name = "params.json"

    path = ".\data"
    json_path = os.path.join(path,json_name)
    tok_path = os.path.join(path,tok_name)

    try:
        tk_file = open(tok_path,'rb')
        tok = pickle.load(tk_file)
        tk_file.close()
    except:
        print("未能打开tok.pkl文件！")
        return

    batch_size = 8
    embedding_dim = 300
    units = 1024
    limit = 24

    try:
        with open(json_path,'r') as diy_setting:
            diy_dicts = json.load(diy_setting)
            batch_size = diy_dicts["batch_size"]
            embedding_dim = diy_dicts["embedding_dim"]
            units = diy_dicts["units"]
            limit = diy_dicts["limit"]
    except:
        print("未能打开params.json配置文件！")
        return

    word_size = len(tok.word_index) + 1 # 因为0是预留位置 所以+1

    encoder = Encoder(word_size,embedding_dim,units,batch_size)
    decoder = Decoder(word_size,embedding_dim,units,batch_size)

    try:
        encoder.load_weights('./models/encoder_weights')
        decoder.load_weights('./models/decoder_weights')
    except:
        print("你的模型在？没读起捏？")
        return

    a = "牛逼的！"
    b = jieba.cut(a)
    a = [i for i in b]
    time.sleep(0.5)

    items = [word_size,encoder,decoder,limit,tok]

    print("已启用CPU！")
    try:
        degree = abs(int(input("请输入随机度(推荐50-100)：")))
    except:
        print("喊你输数字你在干什么？")
        time.sleep(2)
        return
    print(f"当前随机度：{degree}%")
    temp = degree / 100 + 0.01
    result = session(items,temp = temp)