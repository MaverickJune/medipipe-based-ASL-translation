import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def hand_to_cordinate(img_path, filped = False):

  """
  input : a image path
  output : 
            if one hand is detected  >>  min-max normalized coordinate of 42 hand points
            else (two hand or no hand)  >>>  None
  """

  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.2) as hands:
    
    # Read an image
    img_array = np.fromfile(img_path, np.uint8)       
    
    # flip condition
    if filped == True:
      image = cv2.flip(cv2.imdecode(img_array, cv2.IMREAD_COLOR), 1)
    else : 
      image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert the BGR image to RGB
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    # hand exception handling
    if results.multi_hand_landmarks == None:
      print("error occured : NO HAND")
      return None
    num_hands = len(results.multi_hand_landmarks)
    if num_hands >= 2:
      print("error occured : TWO HAND")
      return None

    # cordinate processing
    cordinate_x = np.zeros((21))
    cordinate_y = np.zeros((21))
    hand_landmarks = results.multi_hand_landmarks[0]
    i = 0
    for point in mp_hands.HandLandmark:
      cordinate_x[i] = hand_landmarks.landmark[point].x
      cordinate_y[i] = hand_landmarks.landmark[point].y
      i += 1
  
  # coordination normalization (min-max)  
  cordinate_x = (cordinate_x - np.min(cordinate_x)) / (np.max(cordinate_x) - np.min(cordinate_x))
  cordinate_y = (cordinate_y - np.min(cordinate_y)) / (np.max(cordinate_y) - np.min(cordinate_y))
  cordinate = np.concatenate((cordinate_x, cordinate_y))
  return cordinate


if __name__ == "__main__":
  img = r"C:\Users\bell9\Desktop\deep\l.PNG"
  print(hand_to_cordinate(img))