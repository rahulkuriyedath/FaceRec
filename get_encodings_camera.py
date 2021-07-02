import cv2
import face_recognition as fr
import os
import time
import pandas as pd

ENCODING_PATH = "./encodings/encodings.csv"
if not os.path.exists(ENCODING_PATH):
    encodings_df = pd.DataFrame({'Name':[],'Encoding':[]})
    encodings_df.to_csv(ENCODING_PATH,index=False)
else:
    encodings_df = pd.read_csv(ENCODING_PATH)

name = input('Enter name of the person: ')
# PATH = ENCODING_PATH + '/' + name + '/'
#
# if os.path.exists(ENCODING_PATH):
#     print(f'Directory with name "{name}" already exists')
# else:
#     os.mkdir(PATH)

print('Please look at the camera for 3 seconds')
time.sleep(3)
video_cap = cv2.VideoCapture(0)
ret, frame = video_cap.read()

cv2.imshow('img',frame)
cv2.waitKey(1000)
cv2.destroyAllWindows()

face_loc = fr.face_locations(frame, model="hog")
face_enc = fr.face_encodings(frame,face_loc)

face_enc_df = pd.DataFrame({'Name':name,'Encoding':face_enc})
encodings_df = pd.concat([encodings_df, face_enc_df])
print(encodings_df['Encoding'])
encodings_df.to_csv(ENCODING_PATH,index=False)




