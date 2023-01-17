import os
import numpy as np

from static_image_handtracker import hand_to_cordinate

""" 
input : directory (including A~Z directory)
outcome : save .npy for (42 ndarray)
"""

dataset_path = r"C:\Users\bell9\Desktop\deep\static_dataset\static_data5\data5"     # data  폴더

flipped = False
padding = True
"""
if hand is on right side of image (left hand) => filpped = True
elif hand is on left side of image (right hand) => flipped = False
"""


for dir in os.listdir(dataset_path):
  # make directory
  os.makedirs(os.path.join(dataset_path + "np", dir), exist_ok=True)           
  i = 0
  for img in os.listdir(os.path.join(dataset_path, dir)):
      img_path = os.path.join(dataset_path, dir, img)
      # get coodinate
      cord = hand_to_cordinate(img_path, padding=padding, flipped = flipped)   
      if cord is None :
        continue
      save_path = os.path.join(dataset_path + "np", dir, str(i))
      np.save(save_path, cord)
      i+=1
  print(dir)
