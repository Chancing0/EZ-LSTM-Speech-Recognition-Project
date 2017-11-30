import os
import numpy as np
import librosa
from random import shuffle
def mfcc_batch_generator(batch_size=10,height=10,data_path=None):
  batch_features = []
  labels = []
  files = os.listdir(data_path)
  while True:
    shuffle(files)#將上述2400個文檔進行順序打亂
    for wav in files:
      if not wav.endswith(".wav"): continue   #如果該文檔不是.wav文檔，則跳過本次的循環，換下一個文檔，以下步驟不做，只跳過本次的循環
#      print(wav)
      wave, sr = librosa.load(data_path+"/"+wav, mono=True,sr=None)   #加載一個音頻文檔賦予wave名字,取出他的取樣率賦值給sr,，，，path+檔名，load默認的采樣率是22050，如果需要讀取原始采樣率,需要.load(filename,sr=None)而不是load(filename),mono=True是把音頻處理成單聲道,疑點：音頻本身sr=是8000hz，但這裡使用的是默認值22050
 #     print(sr)
      label=dense_to_one_hot(int(wav[0]),10)    
      labels.append(label)       #labels會把這次的label存到起來，每次迴圈就增加一個lable的值，type是list
      mfcc = librosa.feature.mfcc(wave, sr,n_fft=5000,hop_length=2500,n_mels=30)    #輸入wave形態是numpy.ndarray ,做mfcc處理後返回一個陣列numpy.ndarray 
      mfcc=np.pad(mfcc,((0,0),(0,height-len(mfcc[0]))), mode='constant', constant_values=0)#對大小不一的數據進行填充0，規格固定下來20*80
      batch_features.append(np.array(mfcc))         #type是list
      if len(batch_features) >= batch_size:            #每次都只隨機取barch_size=64個音頻文件來執行上述，
        yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence  
        batch_features = []  # Reset for next batch
        labels = []

def dense_to_one_hot(labels_dense, num_classes=10):      
   """Convert class labels from scalars to one-hot vectors."""
   return np.eye(num_classes)[labels_dense]
