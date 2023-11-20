import cv2
import cvzone
import torch
import numpy as np
import logging
import pickle
import face_recognition

from cvzone.FaceMeshModule import FaceMeshDetector


def operator(frame, operatorMenu):
    imgStacked = cvzone.stackImages([frame, operatorMenu], 2, 1)
    cv2.imshow('operator', imgStacked)


def load_encodings():
    file = open('EncodeFile.p', 'rb')
    encodeListKnownIds = pickle.load(file)
    file.close()
    encodeListKnown, personIds = encodeListKnownIds
    return encodeListKnown


class FaceDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.face_detector = FaceMeshDetector(maxFaces=1)
        load_encodings()

    def process_frame(self, frame):
        operatorMenu = np.zeros_like(frame)
        frame, faces = self.face_detector.findFaceMesh(frame, draw=False)

        if faces:
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faceCurrentFrame = face_recognition.face_locations(imgS)
            encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

            for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
                matches = face_recognition.compare_faces(load_encodings(), encodeFace)
                faceDis = face_recognition.face_distance(load_encodings(), encodeFace)

                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    cv2.putText(operatorMenu, 'Person Detected', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = x1, y1, x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, bbox, rt=0)
                else:
                    cv2.putText(operatorMenu, 'Auth denied', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = x1, y1, x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, bbox, rt=0)

        return frame, operatorMenu

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            assert ret

            frame, operatorMenu = self.process_frame(frame)

            operator(frame, operatorMenu)

            # Press Esc to stop
            if cv2.waitKey(5) & 0xFF == 27:
                print("The program has stopped")
                break

        cap.release()
        cv2.destroyAllWindows()


detector = FaceDetection(capture_index=0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        detector()
    except KeyboardInterrupt:
        print("Forced stop")
        pass
