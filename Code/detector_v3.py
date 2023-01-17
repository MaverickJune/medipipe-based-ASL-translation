import cv2
import mediapipe as mp
import sys
import os
import copy
import re
import glob
import pickle
import time
import argparse

############################ sub logics ############################
def get_delta(prev_f_coord, curr_f_coord, is_detect=True):
    if is_detect == False: # if hand is not detected at current frame
        return 0
    if len(prev_f_coord) == 0 or len(curr_f_coord) == 0: # if prev frame or current frame does not contain hand
        return 0
    if len(prev_f_coord) != len(curr_f_coord): # if prev and curr frame both have hand, but they have diff lenght -> unexpected!
        print("fatal error!, some coordinates are not identified!")
        sys.exit()

    sum_delta = 0
    for i in range(len(curr_f_coord)): # len(curr_f_coord) = 21 (if not there is problem)
        delta_x = abs(curr_f_coord[i]['x'] - prev_f_coord[i]['x'])
        delta_y = abs(curr_f_coord[i]['y'] - prev_f_coord[i]['y'])
        sum_delta = sum_delta + delta_x + delta_y

    return sum_delta  
    
# inner function to get files from directory in order
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts 

# get bounding box from each frame ... (depreciated at final version)
def get_bbox(frame_coord, height, width):
    
    if not frame_coord:
        print("this frame is not identified by mediapipe!")
        sys.exit()
    
    x_max = 0
    y_max = 0
    x_min = width
    y_min = height
    
    for item in frame_coord:
        x, y = int(item['x'] * width), int(item['y'] * height)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
    
    if x_min > 30:
        x_min -= 30
    if y_min > 30:
        y_min -= 30
    if x_max < width - 30:
        x_max += 30
    if y_max < height - 30:
        y_max += 30  
    
    return x_min, y_min, x_max, y_max  

# function for splitting the video input 
def video_splitter(input_file):
    # configure path to source and dst
    in_filename = input_file
    video_name = in_filename.split("/")[-1]
    out_filename = "frame"
    out_dirname = f"{os.getcwd()}/splitted_video/"+video_name[:-4]

    # create output directory if there isn't
    isExist = os.path.exists(out_dirname)
    if not isExist:
        os.makedirs(out_dirname)

    # read video
    print("Reading "+in_filename)
    cap = cv2.VideoCapture(in_filename)

    ## To count the number of frames
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in video : {length}")

    ## split the video into frames
    cnt = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(out_dirname+"/"+out_filename+"_"+str(cnt)+".jpg",frame)
            cnt += 1
        else:
            print("Split process Ended!")
            break

    cap.release()
    return out_dirname 

####################################################################


# 1. frame classifier algorithm
def frame_classify(delta_coord, frame_on):
    # list that stores start and end point of detected subvideos
    start_points = []
    end_points = []
    pre_start_points = []
    post_end_points = []
    
    # variable indicating that subvideo detection is on
    detect_on = False
    first_detect = True
    
    # threshold setting
    stationary_threshold = 0.3
    
    ## pre_start and post_end points detection logic
    for i in range(1,len(frame_on)):
        
        ## pre_start_point detection logic
        if frame_on[i] == True and frame_on[i-1] == False and detect_on == False and first_detect == True:  ## at first object detection ... just compare i and i-1 th frames
            pre_start_points.append(i)
            first_detect = False
            detect_on = True
        elif frame_on[i] == True and frame_on[i-1] == False and detect_on == False:
            flag = 1
            for j in range(min(10, i-1)): ## check previous 10 frames
                if frame_on[i-1-j] != False:
                    flag = 0
            if flag == 1:
                pre_start_points.append(i)
                detect_on = True
                
        ## post_end_point detection logic
        if frame_on[i] == True and frame_on[i+1] == False and detect_on == True:
            flag = 1
            for j in range(10):
                if (i+1+j) >= len(frame_on)-1:
                    break
                if frame_on[i+1+j] != False:
                    flag = 0
            if flag == 1:
                post_end_points.append(i)
                detect_on = False
                    
    ## start points determination logic
    for item in pre_start_points:
        f_count = 0
        s_count = 0
        idx = item
        while True:
            if f_count >= 12 or s_count >= 8:
                start_points.append(idx)
                break
            if frame_on[idx] == True and delta_coord[idx] <= stationary_threshold:
                s_count += 1
            if idx == len(frame_on) - 1:
                start_points.append(idx)
                break
            idx += 1    
            f_count += 1
            
    ## end points determination logic
    for item in post_end_points:
        f_count = 0
        s_count = 0
        idx = item
        while True:
            if f_count >= 12 or s_count >= 8:
                end_points.append(idx)
                break
            if frame_on[idx] == True and delta_coord[idx] <= stationary_threshold:
                s_count += 1
            if idx == 0:
                end_points.append(idx)
                break
            
            idx -= 1    
            f_count += 1
            
    ## for debugging purpose
    # print(f"pre_starting points : {pre_start_points}")
    # print(f"post_end points : {post_end_points}")
    
    return start_points, end_points


# 2. delta_coordinate and frame_on generating logic
def delta_coord_generator(source_dir):
    ## create mediapipe class object to extract coordinate from pictures
    mpHands = mp.solutions.hands
    my_hands = mpHands.Hands(max_num_hands=1, static_image_mode=True)

    ## source directory path
    source_dirname = source_dir

    ## list to save coordinate of previous frame
    prev_frame_coord = []

    ## list to save delta of coordinates between consecutive frames
    delta_coord = []
    
    ## list to save hand detection on each frame is successful or not
    frame_on = []
    
    ## list of frame paths
    frame_path = []
    
    ## list of each frame coordinate
    frame_coord = []
    
    ## control variable
    cnt = 0
    h = 0; w = 0
    
    ## baseline algorithm
    for filename in sorted(glob.glob(source_dirname+'/*.jpg'), key=numericalSort):
        frame = os.path.join(source_dirname, filename)

        if os.path.isfile(frame):
            frame_path.append(frame)
            frame_img = cv2.imread(frame, cv2.IMREAD_COLOR)
            if frame_img is None:
                print("image load failed!")
                sys.exit()
                
            if cnt == 0:
                h, w, _ = frame_img.shape
                cnt += 1
                
            img_RGB = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

            results = my_hands.process(img_RGB)

            if results.multi_hand_landmarks is None:
                dt = get_delta([], prev_frame_coord, is_detect=False)
                delta_coord.append(dt)
                prev_frame_coord = []
                frame_on.append(False)
                frame_coord.append([])
                continue

            # else (results.multi_hand_landmarks is not None)
            curr_frame_coord = []
            for handLms in results.multi_hand_landmarks:
                for i in range(21):
                    curr_frame_coord.append({'x':handLms.landmark[i].x, 'y':handLms.landmark[i].y})
                ## if there are more than two hands .. just discard it
                break

            dt = get_delta(curr_frame_coord, prev_frame_coord, is_detect=True)
            delta_coord.append(dt)
            prev_frame_coord = copy.deepcopy(curr_frame_coord)
            frame_on.append(True)
            frame_coord.append(copy.deepcopy(curr_frame_coord))
            
    return delta_coord, frame_on, frame_path, frame_coord, h, w


## 3. determine splited object is static or dynamic
def detect_motion(delta_coord, start_points, end_points):
    if len(start_points) != len(end_points):
        print("detection error!")
        sys.exit()
    
    ## set threshold (empirically detemined threshold)
    dynamic_threshold = 0.3
    
    ## return array
    motion = []
    
    for i in range(len(start_points)):
        sp = start_points[i]
        ep = end_points[i]
        
        dynamic_cnt = 0
        for idx in range(sp, ep+1):
            if delta_coord[idx] > dynamic_threshold:
                dynamic_cnt += 1
        
        # if there are more than 15 frames .. determine it as dynamic
        if dynamic_cnt >= 15:
            motion.append("d")
            ## for debugging purpose
            # print(f"dynamic count for {i} : {dynamic_cnt}")
        else:
            motion.append("s")
            
    return motion

# 4. final data processing for inference .. (for simulation and empirical callibration purpose)
def final_process(start_points, end_points, motion, frame_path, frame_coord, height, width):
    global_idx = 0
    for i in range(len(motion)):
        if motion[i] == 's':
            
            # if start point is larger than end point, swap them
            if start_points[i] > end_points[i]:
                start_points[i], end_points[i] = end_points[i], start_points[i]
            
            # set the target frame, which is identified by mediapipe
            target_frame_coord = frame_coord[start_points[i]]
            target_frame = frame_path[start_points[i]]
            for f_idx in range(start_points[i], end_points[i]+1):
                if not frame_coord[f_idx]:
                    continue
                else:
                    target_frame_coord = frame_coord[f_idx]
                    target_frame = frame_path[f_idx]
                    break
            
            ## get bounding box and crop image
            x_min, y_min, x_max, y_max = get_bbox(target_frame_coord, height, width)
            image = cv2.imread(target_frame, cv2.IMREAD_COLOR)
            cropped_image = copy.deepcopy(image[y_min:y_max, x_min:x_max])
            
            ## create directory and save the cropped image
            # create output directory if there isn't
            r_dirname = f'{os.getcwd()}/detected_objects/s_object_{global_idx}'
            isExist = os.path.exists(r_dirname)
            if not isExist:
                os.makedirs(r_dirname)
            cv2.imwrite(r_dirname+"/sampled_img.jpg", cropped_image)
            global_idx += 1
            
        elif motion[i] == 'd':
            target_frame_coord_list = copy.deepcopy(frame_coord[start_points[i]:end_points[i]+1])
            r_dirname = f'{os.getcwd()}/detected_objects/d_object_{global_idx}'
            isExist = os.path.exists(r_dirname)
            if not isExist:
                os.makedirs(r_dirname)
            with open(r_dirname+"/dynamic_seq", 'wb') as fp:
                pickle.dump(target_frame_coord_list, fp)
                     
            global_idx += 1
        
        else:
            print("label can only be s or d!")
            sys.exit()      
            
# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/Users/zenius77/Desktop/deep_learning_project/detector_test_env/videosample/classify_test_1.mp4")
    args = parser.parse_args()

    source_video = args.video_path
    source_dirname = video_splitter(source_video)

    ## for stability
    time.sleep(1)

    d_coord, f_on, frame_path, frame_coord, height, width = delta_coord_generator(source_dirname)
    s_points, e_points = frame_classify(d_coord, f_on)
    motion = detect_motion(d_coord, s_points, e_points)
    print(f"starting points : {s_points}")
    print(f"end points : {e_points}")
    print(f"object state : {motion}")
    final_process(s_points, e_points, motion, frame_path, frame_coord, height, width)

           
''' main startpoint '''
if __name__ == "__main__":
    main()
    