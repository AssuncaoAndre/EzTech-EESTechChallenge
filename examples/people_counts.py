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
from numpy import savez_compressed
from time import sleep
import pygame
from sklearn.svm import SVC




def display_text(text):
    screen.fill((0,0,0))
    text_surface = my_font.render(text, False, (220,0,0))

    textRect = text_surface.get_rect()
    textRect.center = (350,250)

    screen.blit(text_surface, textRect)
    pygame.display.update()

text=["0 people", "1 person", "2 people", "3 people"]
people=[0,1,2,3]

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
number_of_frames = 10


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
data_dump=[]
time_dump=[]

hit=0
with RadarIfxAvian(config_file) as device:  # Initialize the radar with configurations
    for k in range(0,2):
        for j in range(0,3):

            display_text("STOP - NEXT: "+text[j])

            time.sleep(1)

            display_text(text[j])

            for i in range(0,5):      
                for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
                    
                    raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
                    if(i_frame == number_of_frames-1):
                        data = np.asarray(raw_data)
                        range_doppler_map = processing.processing_rangeDopplerData(data)

                        range_data=processing.processing_rangeDopplerData(range_doppler_map,True)
                        #del data
                        break      

                ts=calendar.timegm(time.gmtime())

                data_dump.append(range_data)
                print("aqui "+str(hit))
                hit=hit+1

            print("finished frames")

            print("training")

            #SVM
            
            X_train = data_dump

            Y_data=[]

            for p in range (0,5):
                Y_data.append(people[j])

            print(np.shape(X_train))
            print(np.shape(Y_data))
            
            clf = SVC()
            clf.fit(X_train, Y_data)









