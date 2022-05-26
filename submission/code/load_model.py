import tensorflow as tf
import pygame
import sys
import numpy as np
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import json
import processing

#Display Stuff-------------------------------

def display_text(text):
    screen.fill((0,0,0))
    text_surface = my_font.render(text, False, (220,0,0))

    textRect = text_surface.get_rect()
    textRect.center = (350,250)

    screen.blit(text_surface, textRect)
    pygame.display.update()

pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
my_font = pygame.font.SysFont('Comic Sans MS', 70)
size = (700, 500)
X=700
Y=500
screen = pygame.display.set_mode(size)

pygame.display.update()
np.set_printoptions(threshold=sys.maxsize)

text=["0 people", "1 person", "2 people", "3 people"]


#Radar Stuff-------------------------------

config_file = "radar_configs/RadarIfxBGT60.json"
number_of_frames = 10

f2r = 10

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


#load model
model = tf.keras.models.load_model('models/model_one.sav')

data_dump=[]

#Prediction Cycle
with RadarIfxAvian(config_file) as device:  # Initialize the radar with configurations

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
                #data_dump.append(np.array(range_data, dtype=object).astype("complex128"))
                data_dump.append((aux/number_of_frames))
                break      
        res=model.predict(np.array(data_dump))
        
        print(res[0])
        if(res[0][0]>=res[0][1] and res[0][0]>=res[0][2] and res[0][0]>= res[0][3]):
            display_text("Detected: "+text[0])
            print(0)
        elif(res[0][1]>res[0][0] and res[0][1]>res[0][2] and res[0][1]> res[0][3]):
            display_text("Detected: "+text[1])
            print(1)
        elif(res[0][2]>res[0][1] and res[0][2]>res[0][0] and res[0][2]> res[0][3]):
            display_text("Detected: "+text[2])
            print(2)
        elif(res[0][3]>res[0][1] and res[0][3]>res[0][2] and res[0][3]> res[0][0]):
            display_text("Detected: "+text[3])
            print(3)
        print(np.shape(data_dump))
        print(type(np.array(data_dump)))
        #test_eval = model.predict(data_dump , verbose=2)
        data_dump.clear()


