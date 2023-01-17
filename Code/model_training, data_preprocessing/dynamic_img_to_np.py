import os
import numpy as np

from dynamic_image_handtracker import hand_to_cordinate


dataset_path = r"C:\Users\bell9\OneDrive\바탕 화면\deep\dynamic_dataset_np"     # data를 저장할 폴더
img_path = r"C:\Users\bell9\OneDrive\바탕 화면\dynamic_trial1"                  # J,Z를 포함하는 이미지 폴더


J_idx = 128
Z_idx = 125

flipped = False
"""
if hand is on right side of image (left hand) => filpped = True
elif hand is on left side of image (right hand) => flipped = False
"""


def dir_npconverter(dir_path, idx, save_dir):

  """
  input : directory (including image set directory. note below)
  outcome : save .npy for (n * 42 ndarray)

  """
  set_list = os.listdir(dir_path)

  for path in set_list:
    sub_j_path = os.path.join(dir_path, path)
    print(sub_j_path)
    np_z = []
    for img in os.listdir(sub_j_path):
      img_path = os.path.join(sub_j_path, img)

      # get coordinate for image set
      cord = hand_to_cordinate(img_path, flipped)
      if cord is None :
        continue
      np_z.append(cord)

    # handling for empty folder
    if len(np_z) == 0 :
      print("@@@@@ empty folder @@@@@")
      continue

    #np stack > n * 42 ndarray
    npz = np.vstack(np_z)
    save_path = os.path.join(dataset_path, save_dir, str(idx)) + ".npy"
    if os.path.exists(save_path):
      print("@@@@@ file already exist!!!!!! @@@@@")
      exit(0)
    np.save(save_path, npz)
    print(save_path)
    idx += 1

def main():
  # make directory
  os.makedirs(dataset_path + r"/np_J", exist_ok=True)
  os.makedirs(dataset_path + r"/np_Z", exist_ok=True)
  
  # path for J, Z
  J_path = os.path.join(img_path, "J")
  Z_path = os.path.join(img_path, "Z")

  # save .npy for each J, Z set
  dir_npconverter(J_path, J_idx, "np_J")
  dir_npconverter(Z_path, Z_idx, "np_Z")

if __name__ == "__main__":
  main()
  print("@@@@@ done!!! @@@@@")

############################
# Image dataset
#   ├── J  # must be J
#   │   ├── J1
#   │   │    ├── --1.jpg
#   │   │    ├── --2.jpg
#   │   │    └── --n.jpg
#   │   └── J2
#   │        ├── --1.jpg
#   │        ├── --2.jpg
#   │        └── --n.jpg
#   │
#   │
#   ├── Z  # must be Z
#   │   ├── Z1
#   │   │    ├── --1.jpg
#   │   │    ├── --2.jpg
#   │   │    └── --n.jpg
#   │   └── Z2
#   │        ├── --1.jpg
#   │        ├── --2.jpg
#   │        └── --n.jpg
############################