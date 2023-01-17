# import library and codes needed
import os
import shutil
import sys
import copy
import numpy as np
import time
from detector_v3 import video_splitter, delta_coord_generator, frame_classify, detect_motion
import torch
from dynamic_predict import Dynamic_predict
from static_predict import static_model

f2d = "./splitted_video"
if os.path.exists(f2d):
    shutil.rmtree(f2d)

# specify absolute path of source video // input to the model
source_video = "/tools/home/aistore9/project_final/hazard.mp4" # fixme !
# get the directory of splitted video(video frames)
source_dirname = video_splitter(source_video)

# for stability
time.sleep(1)

# d_coord : delta of coordinate btw adjacent frames (delta_coord)
# f_on : whether it is detected by mediapipe or not (frame_on)
# frame_path : absolute path to each frame (frame_path)
# frame_coord : 42 coordinate belongs to each frame (frame_coord)
# height, width : height and width of frames
# s_points : starting frame of each frame chunks that represents ASL character (start_points)
# e_points : end frame of each frame chunks that represents ASL character (end_points)
# motion : whether each frame chunk represent dynamic/static ASL
# more info on the report....

print("getting coordinate difference from frames...")
d_coord, f_on, frame_path, frame_coord, height, width = delta_coord_generator(source_dirname)

print("classifying frame chunks...")
s_points, e_points = frame_classify(d_coord, f_on)

print("detecting static/dynamic characteristic of each frame chunks...")
motion = detect_motion(d_coord, s_points, e_points)

# # for debugging purpose .. to check whether detector splitted the video properly
# print(f"starting points: {s_points}")
# print(f"end points: {e_points}")
# print(f"object state: {motion}")

# list to save output
output = []

# model 1 : static classifier model
model1 = static_model()
model1.load_state_dict(torch.load("/tools/home/aistore9/project/static.pt")) #load weight .. fix me!

# model 2 : dynamic classifier model
model2 = Dynamic_predict()
model2.load_state_dict(torch.load("/tools/home/aistore9/project/last_model.pt")) #load weight .. fix me!

# global_idx : variable for saving result in proper order
global_idx = 0

print("producing output...")

for i in range(len(motion)):
    # if the frame chunk is detected as static ASL
    if motion[i] == 's':
        
        # code for stability .. to ensure bbox is obtained from mediapipe-recognizable frame
        if (s_points[i] > e_points[i]):
            s_points[i], e_points[i] = e_points[i], s_points[i]
        target_frame_coord = frame_coord[s_points[i]]
        target_frame = frame_path[s_points[i]]
        for f_idx in range(s_points[i], e_points[i]+1):
            if not frame_coord[f_idx]:
                continue
            else:
                target_frame_coord = frame_coord[f_idx]
                target_frame = frame_path[f_idx]
                break
            
        # prepare input to model 1 : tensor shape (,42)
        temp = []
        for item in target_frame_coord:
          temp.append(item['x'])
        for item in target_frame_coord:
          temp.append(item['y'])
        np.tmp = np.array(temp)
        tensor_input = torch.from_numpy(np.tmp).float()
        
        # increment global index
        global_idx += 1

        # feed out input to model 1(static classifier)
        tmp = model1.test_print(tensor_input)
        
        # append the output
        output.append(tmp)
        
    # if the frame chunk is detected as Dynamic ASL
    elif motion[i] == 'd':
        # get the coordinates of dynamic frames
        target_frame_coord_list = copy.deepcopy(frame_coord[s_points[i]:e_points[i]+1])
        
        # list_save : input to dynamic classifier
        list_save = []
        for item_list in target_frame_coord_list:
            if (len(item_list) == 0):
                continue
            temp = []
            for item_dict in item_list:
                temp.append(item_dict['x'])
            for item_dict in item_list:
                temp.append(item_dict['y'])
            list_save.append(temp)
        # change it to numpy -> tensor to feed in to model
        np_temp = np.array(list_save)
        tensor_input = torch.from_numpy(np_temp).float()
        
        # increase the global index
        global_idx += 1
        
        # get and append output
        output.append(model2.test_print(tensor_input))
        
    else:
        print("label can only be s or d!")
        sys.exit()   

# print the final output
print("\n")
print("ans : " + "".join(output))
print("\n")
