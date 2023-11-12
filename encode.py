import cv2
import face_recognition
import pickle
import os

folderPath = 'images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
personIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    personIds.append(os.path.splitext(path)[0])

print(personIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("encoding started")
encodeListKnown = findEncodings(imgList)
encodeListKnownIds = [encodeListKnown, personIds]
print("encoding complete")

file = open('EncodeFile.p', 'wb')
pickle.dump(encodeListKnownIds, file)
file.close()
print('File saved')
