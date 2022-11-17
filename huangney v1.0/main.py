import chatting
import training
import time
from qtrans import q_trans

with open("./icon/name.txt",encoding="utf-8") as names:
      print(names.read())
train_ed = False
mode = 0
while(mode + 1):
      print("\n可选择模式：\n"
            "输入1：对话模式\n"
            "输入2：QQ文件处理\n"
            "输入3：自定义模型模式\n"
            "输入-1：back")

      try:
            mode = int(input())
      except:
            print("你这纯纯非法输入")
            continue

      if mode == 1:
            print("正在进入对话模式")
            time.sleep(1)
            chatting.chat()
      elif mode == 2:
            q_trans()
      elif mode == 3:
            if train_ed == True:
                  print("\n请勿多次调用训练，如需再次训练请重启程序。\n")
                  time.sleep(2)
                  continue
            print("警告：GPU模式可能基于Cudnn和Cuda，请谨慎使用")
            try:
                  if_gpu = int(input("输入0以调用CPU，1以调用GPU\n"))
                  if (if_gpu != 1) and (if_gpu != 0):
                        print("让你乱输了吗？")
                        time.sleep(1)
                        continue
                  else:
                        train_ed =  training.self_training(if_gpu = if_gpu)
            except:
                  print("让你乱输了吗？")
                  time.sleep(1)
                  continue
      elif mode == -1:
            print("别骂了在关了")
            time.sleep(1)
      else:
            print("你别玩了搞什么非法输入？")
            time.sleep(2)