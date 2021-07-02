import cv2
import face_recognition as fr
import os
import pandas as pd

ENCODING_PATH_GREEN = "./encodings/encodings_green.csv"
ENCODING_PATH_RED = "./encodings/encodings_red.csv"

IMAGE_ROOT = "./images/Red/"

if not os.path.exists(ENCODING_PATH_RED):
    encodings_red_df = pd.DataFrame({'Name':[],'Encoding':[]})
    encodings_red_df.to_csv(ENCODING_PATH_RED,index=False)
else:
    encodings_red_df = pd.read_csv(ENCODING_PATH_RED)


names = [x[1] for x in os.walk(IMAGE_ROOT)][0]  # get sub-directories i.e. names of people
for name in names:
    for imagename in os.listdir(IMAGE_ROOT+name+'/'):
        image_path = IMAGE_ROOT+name+'/'+imagename
        print(image_path)
        img = cv2.imread(image_path)
        face_loc = fr.face_locations(img, model="hog")
        face_enc = fr.face_encodings(img,face_loc)

        face_enc_df = pd.DataFrame({'Name':name,'Encoding':face_enc})
        encodings_red_df = pd.concat([encodings_red_df, face_enc_df])

encodings_red_df.to_csv(ENCODING_PATH_RED,index=False)
############################

IMAGE_ROOT = "./images/Green/"

if not os.path.exists(ENCODING_PATH_GREEN):
    encodings_df_green = pd.DataFrame({'Name':[],'Encoding':[]})
    encodings_df_green.to_csv(ENCODING_PATH_GREEN,index=False)
else:
    encodings_df_green = pd.read_csv(ENCODING_PATH_GREEN)


names = [x[1] for x in os.walk(IMAGE_ROOT)][0]  # get sub-directories i.e. names of people
for name in names:
    for imagename in os.listdir(IMAGE_ROOT+name+'/'):
        image_path = IMAGE_ROOT+name+'/'+imagename
        print(image_path)
        img = cv2.imread(image_path)
        face_loc = fr.face_locations(img, model="hog")
        face_enc = fr.face_encodings(img,face_loc)

        face_enc_df = pd.DataFrame({'Name':name,'Encoding':face_enc})
        encodings_df_green = pd.concat([encodings_df_green, face_enc_df])


encodings_df_green.to_csv(ENCODING_PATH_GREEN,index=False)



