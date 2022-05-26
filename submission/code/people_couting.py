from ctypes import sizeof
from cv2 import displayStatusBar
import ifxdaq
import processing
import numpy as np
#print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import matplotlib.pyplot as plot
import sys
import json
import time
import calendar
from numpy import dtype, empty, savez_compressed
from time import sleep
import pygame

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as im
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
import csv
import time
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 



def display_text(text):
    screen.fill((0,0,0))
    text_surface = my_font.render(text, False, (220,0,0))

    textRect = text_surface.get_rect()
    textRect.center = (350,250)

    screen.blit(text_surface, textRect)
    pygame.display.update()

text=["0 people", "1 person", "2 people", "3 people"]

pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
my_font = pygame.font.SysFont('Comic Sans MS', 70)
size = (700, 500)
X=700
Y=500
screen = pygame.display.set_mode(size)

pygame.display.update()
np.set_printoptions(threshold=sys.maxsize)

config_file = "radar_configs/RadarIfxBGT60.json"
number_of_frames = 5

f2r = 100

data_epoc = 2

with open(config_file) as json_file:
    c = json.load(json_file)["device_config"]["fmcw_single_shape"]
    chirp_duration = c["num_samples_per_chirp"]/c['sample_rate_Hz']
    frame_duration = (chirp_duration + c['chirp_repetition_time_s']) * c['num_chirps_per_frame']
    print("With the current configuration, the radar will send out " + str(c['num_chirps_per_frame']) + \
          ' signals with varying frequency ("chirps") between ' + str(c['start_frequency_Hz']/1e9) + " GHz and " + \
          str(c['end_frequency_Hz']/1e9) + " GHz.")
    print('Each chirp will consist of ' + str(c["num_samples_per_chirp"]) + ' ADC measurements of the IF signal ("samples").')
    print('A chirp takes ' + str(chirp_duration*1e6) + ' microseconds and the delay between the chirps is ' + str(c['chirp_repetition_time_s']*1e6) +' microseconds.')
    print('With a total frame duration of ' + str(frame_duration*1e3) + ' milliseconds and a delay of ' + str(c['frame_repetition_time_s']*1e3) + ' milliseconds between the frame we get a frame rate of ' + str(1/(frame_duration + c['frame_repetition_time_s'])) + ' radar frames per second.')

raw_data    = []
time_dump=[]

X_train=np.empty(())
Y_train=np.empty(())

Y_test=np.empty(())
X_test=np.empty(())

data_dump=[]
aux=np.empty((64,64),dtype=float)
hit=0

#fig, axs = plot.subplots(f2r, number_of_frames,figsize=(15,10),sharex=True, sharey=  True)
#fig.suptitle('Range-Doppler Plot')

with RadarIfxAvian(config_file) as device:  # Initialize the radar with configurations
    for k in range(0,data_epoc):
        for j in range(0,4):

            display_text("STOP - NEXT: " + text[j])

            time.sleep(6)

            display_text(text[j])

            for i in range(0,f2r):      
                raw_data.clear()
                hit = 0
                for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
                    aux=np.empty((64,64),dtype=float)
                    raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
                    if(i_frame == number_of_frames-1):
                        data = np.asarray(raw_data).astype('complex128')

                        range_data=processing.processing_rangeDopplerData(data,True)
                        #range_data=np.abs(range_data)
                        
                        for n1 in range(3):
                            for n2 in range(number_of_frames):
                                aux = aux + np.abs(range_data)[n2,n1,:,:]; #average of the 10 frames
                        
                        #del data
                        #data_dump.append(np.array(range_data))
                        print(np.shape(aux))#del data
                        #data_dump.append(np.array(range_data, dtype=object).astype("complex128"))
                        data_dump.append((aux/number_of_frames))
                        #axs[j, i_frame].imshow(aux/number_of_frames)
                        #axs[i, j].set_aspect('equal') 
                        #print(np.shape(data_dump))
                        break      
           
                ts=calendar.timegm(time.gmtime())

                hit=hit+1
            
            print("finished frames")

            print(np.shape(Y_train))
            print(np.shape(np.ones((f2r), dtype=float)*j))

            if np.size(Y_train) > 1 :
                Y_train = np.append(Y_train, np.ones((f2r), dtype=float)*j)
            else:
                Y_train = np.ones((f2r), dtype=float)*j


            print("training")

            #Neural Network
     
    X_train=data_dump

    print(np.shape(Y_train))    

    Y_train= np.reshape(Y_train,(f2r*4*data_epoc,1))
    print("AQUI")

    print(Y_train)
    
    model = models.Sequential()
    model.add(layers.Conv2D(128, (4, 4), activation='tanh', input_shape=(64, 64,1)))
    model.add(layers.Conv2D(128, (4, 4), activation='tanh'))
    model.add(layers.AveragePooling2D((4, 4)))
    model.add(layers.Conv2D(64, (4, 4), activation='tanh'))
    model.add(layers.Conv2D(64, (4, 4), activation='tanh'))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='tanh'))
    model.add(layers.Dense(1024, activation='tanh'))
    model.add(layers.Dense(4, activation='softmax'))

    model.summary()

    opt = tf.keras.optimizers.Adam(0.001, clipnorm=1.)

    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    since_train = time.time()

    print("brace urselves")

    print(type(X_train))
    print(type(Y_train))
    #print(Y_train)

    print(np.shape(X_train))
    print(np.shape(Y_train))

    X_train = np.asarray(X_train, dtype=float)


    data_dump.clear()
    for j in range(0,4):

        display_text("STOP - NEXT: " + text[j])

        time.sleep(6)

        display_text(text[j])

        for i in range(0,f2r):      
            raw_data.clear()
            hit = 0
            for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
                aux=np.empty((64,64),dtype=float)
                raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
                if(i_frame == number_of_frames-1):
                    data = np.asarray(raw_data).astype('complex128')

                    range_data=processing.processing_rangeDopplerData(data,True)
                    #range_data=np.abs(range_data)
                    
                    for n1 in range(3):
                        for n2 in range(number_of_frames):
                            aux = aux + np.abs(range_data)[n2,n1,:,:]; #average of the 10 frames
                    

                    print(np.shape(aux))#del data
                    data_dump.append((aux/number_of_frames))

                    break      
        
            ts=calendar.timegm(time.gmtime())

            hit=hit+1
        
        print("finished frames")
        X_test=data_dump
        print(np.shape(Y_test))
        print(np.shape(np.ones((f2r), dtype=float)*j))

        if np.size(Y_test) > 1 :
            Y_test = np.append(Y_test, np.ones((f2r), dtype=float)*j)
        else:
            Y_test = np.ones((f2r), dtype=float)*j


        print("training")


    X_test = np.asarray(X_test, dtype=float)


    history = model.fit(X_train, Y_train, epochs=2, validation_data=(X_test,Y_test))


    time_elapsed = time.time() - since_train
    print(' Time elapsed training: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    filename = 'model_2.sav'
    model.save(filename)

    
    #Neural Network
    while(True):
        data_dump.clear()
        raw_data.clear()

        for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
            aux=np.empty((64,64),dtype=float)
            raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
            if(i_frame == number_of_frames-1):
                data = np.asarray(raw_data).astype('complex128')

                range_data=processing.processing_rangeDopplerData(data,True)
                #range_data=np.abs(range_data)
                
                for n1 in range(3):
                    for n2 in range(number_of_frames):
                        aux = aux + np.abs(range_data)[n2,n1,:,:]; #average of the 10 frames
                
                #del data
                #data_dump.append(np.array(range_data))
                print(np.shape(aux))#del data
                #data_dump.append(np.array(range_data, dtype=object).astype("complex128"))
                data_dump.append((aux/number_of_frames))
                print(np.shape(data_dump))
                break      
        res=model.predict(np.array(data_dump))
        
        print(res[0])
        if(res[0][0]>=res[0][1] and res[0][0]>=res[0][2] and res[0][0]>= res[0][3]):
            print(0)
        elif(res[0][1]>res[0][0] and res[0][1]>res[0][2] and res[0][1]> res[0][3]):
            print(1)
        elif(res[0][2]>res[0][1] and res[0][2]>res[0][0] and res[0][2]> res[0][3]):
            print(2)
        elif(res[0][3]>res[0][1] and res[0][3]>res[0][2] and res[0][3]> res[0][0]):
            print(3)
        print(np.shape(data_dump))
        print(type(np.array(data_dump)))
        #test_eval = model.predict(data_dump , verbose=2)
        data_dump.clear()





