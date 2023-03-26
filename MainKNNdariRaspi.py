#! Myenv/bin/python

from hashlib import new
import sys
import  threading
import bluetooth
import time
import pyaudio
import math
import struct
import wave
import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import warnings
import tensorflow as tf

from BFCCTrain import convertBFCC
from wfcc import convertWFCC
#from GFCC import convertGFCC
from realtime import mainGFCC

os.system('sudo sdptool add --channel=22 SP')
#BLUETOOTH
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
address = ""
while len(address) == 0:
    print('searching')
    sock.bind(("", 22))
    sock.listen(1)
    sock, address = sock.accept()
print(address)


def receiveData():
    global dataFromApp
    while True:
        dataFromApp = sock.recv(1024).decode()
        print("DATA FROM APP=============  ", dataFromApp)
        
        if dataFromApp == '1':
            start = time.time()
            convertBFCC()
            end = time.time()
            print ("Waktu ekstraksi = ", end-start)
            predictBFCC()

        if dataFromApp == '2':
            start = time.time()
            convertWFCC()
            end = time.time()
            print ("Waktu ekstraksi = ", end-start)
            predictWFCC()

        if dataFromApp == '3':
            start = time.time()
            data = mainGFCC()
            end = time.time()
            print ("Waktu ekstraksi = ", end-start)
            predictGFCC(data)

        if dataFromApp == '4':
            print("turning off")
            break
            sys.exit()
        


def StopStream(stream,p):
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("turning off mic")

def GetStream(chunk, stream):
    return stream.read(chunk)




def predictBFCC():

    start = time.time()
    #Klasifikasi KNN
    path1 = "/home/pi/Myenv/BFCCTrain.xlsx"
    path2 = "/home/pi/Myenv/BFCCTest.xlsx"

    dataset1 = pd.read_excel(path1, header=None)
    dataset2 = pd.read_excel(path2, header=None)
    
    x_train = dataset1.iloc[1:, :40].values
    y_train = dataset1.iloc[1:, 40].values

    x_test = dataset2.iloc[1:, :40].values
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    knn.fit(x_train, y_train)
    klasifikasiDataMentah = knn.predict(x_test)
    hasil = np.array_str(klasifikasiDataMentah)

    print (f"Your sadness intensity is {hasil}")
    send = f"Your sadness intensity is {hasil}"
    sock.send(send)
    end = time.time()
    print ("Waktu prediksi = ", end-start)


def predictWFCC():
    start = time.time()
    path1 = '/home/pi/Myenv/WFCCTrain.xlsx' #data training
    path2 = '/home/pi/Myenv/WFCCTest.xlsx' #data test mentah

    dataset1 = pd.read_excel(path1, header=None)
    dataset2 = pd.read_excel(path2, header=None)

    x_train = dataset1.iloc[1:, :40].values
    y_train = dataset1.iloc[1:, 40].values

    x_test = dataset2.iloc[1:, :40].values
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    knn.fit(x_train,y_train)
    klasifikasiDataMentah = knn.predict(x_test)

    hasil = np.array_str(klasifikasiDataMentah)
    print (f"Intensitas emosi Marah anda {hasil}")
    send = f"Intensitas emosi Marah anda {hasil}"
    sock.send(send)
    end = time.time()
    print ("Waktu prediksi = ", end-start)

def predictGFCC(data1):
    start = time.time()
    bob = f"Intensitas emosi ketakuktan anda adalah{str(data1)}"
    print("Intensitas emosi ketakutan anda adalah:", str(data1))
    sock.send(bob)
    end = time.time()
    print ("Waktu prediksi = ", end-start)

if __name__ == '__main__':
    dataFromApp = ''
    # dataApp = sock.recv(1024).decode()
    # stream, p = StartStream()
    #t1 = threading.Thread(target=KeepRecord, args=())
    t2 = threading.Thread(target=receiveData, args=())
    # if dataApp == '1':

    #t1.start()
    t2.start()

    #t1.join()
    t2.join()

    