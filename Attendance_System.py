import cv2
import face_recognition
import numpy as np
import os
from datetime import date


def bin_search(roll, nums):
    l = 0
    u = len(nums) - 1
    mid = l + ((u - l) // 2)
    while l <= u:
        sec_list = nums[mid].split(',')
        if int(sec_list[0]) == roll:
            return sec_list
        elif int(sec_list[0]) < roll:
            l = mid + 1
        else:
            u = mid - 1
        mid = l + ((u - l) // 2)
    return [-1, -1]


def findencoding(images1):
    encodelist = []
    for img in images1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodelist.append(face_recognition.face_encodings(img)[0])
    return encodelist


def presentf(name):
    t = date.today()
    s = f"Attendance_{t}.csv"
    try:
        with open(s, 'r+') as f:
            f.readlines()
            with open('Data.csv', 'r') as g:
                g.readline()
                d = g.readlines()
                r, n = bin_search(name, d)
                if r != -1:
                    f.writelines(f'{r},{n}\n')
                    return True
        return False
    except:
        with open(s, 'w') as f:
            f.writelines('RollNo,Name\n')
            with open('Data.csv', 'r') as g:
                g.readline()
                d = g.readlines()
                r, n = bin_search(name, d)
                if r != -1:
                    f.writelines(f'{r},{n}')
                    return True
        return False

path = 'ImageData'
images = []
classNames = []
myList = os.listdir(path)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
encodel = findencoding(images)
print('Encoding Completed')
cap = cv2.VideoCapture(0)
present = []
stop = False
while not (stop):
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facesCurrframe = face_recognition.face_locations(imgs)
    encodingofcurr = face_recognition.face_encodings(imgs, facesCurrframe)
    
    for face, loc in zip(encodingofcurr, facesCurrframe):
        matches = face_recognition.compare_faces(encodel, face)
        dist = face_recognition.face_distance(encodel, face)
        matched = np.argmin(dist)
        if dist[matched] > 0.45:
            matched = -1
        y1, x2, y2, x1 = loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        if matches[matched] and matched >= 0:
            name = classNames[matched]
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            if name not in present:
                if presentf(int(name)):
                    present.append(name)
                    print("Roll Number : ", name, " Present!")
        else:
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow('Webcam', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        stop = True
print('The Roll Numbers that are present are : ')
for i in range(len(present)):
    if i == len(present) - 1:
        print(present[i])
    else:
        print(present[i], end=',')
