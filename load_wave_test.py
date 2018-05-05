from __future__ import print_function, division
import tflearn
import speech_data
import tensorflow as tf
import numpy as np

data_path=r"D:\python_code\tensorflow_works\testx\pre"
learning_rate = 0.001
batch_size = 1

width = 20  # mfcc features 
height =29  # (max) length of utterance
classes = 10 # digits
batch = word_batch = speech_data.mfcc_batch_generator(batch_size,height,data_path)          #傳入64，返回一個生產器batch，每次只會執行一次，每次使用就算出一次的值。
#print('batch =',batch)
#X, Y = next(batch)        

# Network building
net = tflearn.input_data([None, width, height])

#net = tflearn.lstm(net,256, dropout=0.8)
net = tflearn.lstm(net,2048)
#jie bi jie bi 
net = tflearn.fully_connected(net, classes, activation='softmax')

net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('tflearn.lstm.model')

test_times=476
acc_times=0
acc_rate=0
print("test_times is : ",test_times)
for i in range(test_times):
    X, Y = next(batch)
    y=model.predict(X)
    b=Y[0].tolist()
    b=b.index(max(b))
  #  print("\n","input : ",b,"\n",Y,"\n")
    a=y[0].tolist()
    meens=a.index(max(a))
 #   print(" Output : ",meens,"\n",y,"\n a.max =",max(a),"\n")
    if(meens==b):acc_times+=1
    # if meens==0:  print("Result : Zero")
    # elif meens==1: print("Result : One")
    # elif meens==2: print("Result : Two")
    # elif meens==3: print("Result : Three")
    # elif meens==4: print("Result : Four")
    # elif meens==5: print("Result : Five")
    # elif meens==6: print("Result : Six")
    # elif meens==7: print("Result : Seven")
    # elif meens==8: print("Result : Eight")
    # elif meens==9: print("Result : Nine")
print("Total acc_times is",acc_times)
print("acc_rate is ",(acc_times/test_times)*100,"%")
