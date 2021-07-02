import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def processEncoding(enc):
    enc = enc.replace('\n', '')
    enc = enc.replace('[', '')
    enc = enc.replace(']', '')
    enc = enc.split()
    for i in range(len(enc)):
        enc[i] = float(enc[i])
    return enc


ENCODING_GREEN_PATH = "./encodings/encodings_green.csv"
ENCODING_RED_PATH = "./encodings/encodings_red.csv"

# ENCODING_PATH = "./encodings/encodings.csv"

encodings_green_df = pd.read_csv(ENCODING_GREEN_PATH)
encodings_green_df['Encoding_formatted'] = encodings_green_df['Encoding'].apply(lambda x: processEncoding(x))
known_face_encodings_green = list(encodings_green_df['Encoding_formatted'])
known_face_names_green = list(encodings_green_df['Name'])

encodings_red_df = pd.read_csv(ENCODING_RED_PATH)
encodings_red_df['Encoding_formatted'] = encodings_red_df['Encoding'].apply(lambda x: processEncoding(x))
known_face_encodings_red = list(encodings_red_df['Encoding_formatted'])
known_face_names_red = list(encodings_red_df['Name'])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
timestamps = dict()
timeout = 5
process_this_frame = True

for i in known_face_names_green:
    timestamps[i] = None

for i in known_face_names_red:
    timestamps[i] = None

print(timestamps)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/friends_video.mp4')

# while True:
while cap.isOpened():

    ret, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        categories = []
        for face_encoding in face_encodings:
            match_found = False
            name = "Unknown"
            category = "Unknown"

            # See if face matches any known red-list face
            matches = face_recognition.compare_faces(known_face_encodings_red, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings_red, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names_red[best_match_index]
                category = "Red"
                match_found = True
                if timestamps[name] == None:
                    timestamps[name] = datetime.now()
                    print('Intruder detected: ', name, timestamps[name])
                elif datetime.now() > timestamps[name] + timedelta(0, timeout):
                    timestamps[name] = datetime.now()
                    print('Intruder detected: ', name, timestamps[name])

            if not match_found:
                # See if face matches any known green-list face
                matches = face_recognition.compare_faces(known_face_encodings_green, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings_green, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names_green[best_match_index]
                    category = "Green"
                    match_found = True

                    if timestamps[name] == None:
                        timestamps[name] = datetime.now()
                        print(name, timestamps[name])
                    elif datetime.now() > timestamps[name] + timedelta(0, timeout):
                        timestamps[name] = datetime.now()
                        print(name, timestamps[name])

            face_names.append(name)
            categories.append(category)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name, category in zip(face_locations, face_names, categories):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if category == "Red":
            color = (0, 0, 255)
        elif category == "Green":
            color = (0, 150, 0)
        else:
            color = (255, 0, 0)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
