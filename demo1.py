from __future__ import division, print_function, absolute_import
import tflearn
import speech_data_test2 as speech_data
import tensorflow as tf
import numpy as np

learning_rate = 0.001
tra_data_path=r"D:\python_code\tensorflow_works\train_10\pre"
val_data_path=r"D:\python_code\tensorflow_works\valx\pre"
training_iters =100  # steps
batch_size = 512

validation_batch_size=236

width = 20  # mfcc features
height = 37  # (max) length of utterance
classes = 10  # digits

tra_batch = speech_data.mfcc_batch_generator(batch_size,height,tra_data_path)
val_batch=speech_data.mfcc_batch_generator(validation_batch_size,height,val_data_path)

# Network building
net = tflearn.input_data([None, width,height])   #20*141
net = tflearn.lstm(net, 2048, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='Softmax')
#adam = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.99)
net = tflearn.regression(net, optimizer='Adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

x, y = next(val_batch)
testX, testY = x, y

model = tflearn.DNN(net, tensorboard_verbose=0)
for times in range(training_iters):
  X, Y = next(tra_batch)
#  X = np.transpose(np.asarray(X),(0, 2, 1))
  trainX, trainY = X, Y
#  x, y = next(val_batch)
#  x = np.asarray(x).reshape([validation_batch_size, height, width])
#  testX, testY = x, y
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY),validation_batch_size=validation_batch_size, show_metric=True,
          batch_size=batch_size)
  
#  print("trainY","\n",trainY)
#  print("testY","\n",testY)
#  y=model.predict(X)
model.save("tflearn.lstm.model")
#print (y)
# ###
# for each in range(training_iters):
#   a=y[each].tolist()
#   print("its the %d  times predict"%each)
#   print("a = ",a)
#   print("a.max =",max(a))
#   meens=a.index(max(a))
#   if meens==0:  print("0")
#   elif meens==1: print("1")
#   elif meens==2: print("2")
#   elif meens==3: print("3")
#   elif meens==4: print("4")
#   elif meens==5: print("5")
#   elif meens==6: print("6")
#   elif meens==7: print("7")
#   elif meens==8: print("8")
#   elif meens==9: print("9")
#   ###