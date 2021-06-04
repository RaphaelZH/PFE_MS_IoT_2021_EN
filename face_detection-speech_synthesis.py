import cv2
import argparse
import os
from datetime import datetime
import time
import pyttsx3

DEFAULT_CASCADE_INPUT_PATH = 'haarcascade_frontalface_alt.xml'
DEFAULT_OUTPUT_PATH = 'FaceCaptureImages/'


class VideoCapture:

    def __init__(self):
        self.count = 0
        self.argsObj = Parse()
        self.faceCascade = cv2.CascadeClassifier(self.argsObj.input_path)
        self.videoSource = cv2.VideoCapture(0)

    def CaptureFrames(self):
        while True:

            # Create a unique number for each frame
            frameName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Capture frame-by-frame
            ret, frame = self.videoSource.read()

            # Set screen color to gray, so the haar cascade can easily detect edges and face
            screenColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Customize how the cascade detects your face
            faces = self.faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            # Display the resulting frame
            cv2.imshow('Video', screenColor)

            # If length of faces is 0, there have been no faces detected
            if len(faces) == 0:
                pass

            # If a face is detected, faces returns 1 or more depending on amount of faces detected
            if len(faces) > 0:
                print('Face Detected')
                # Graph the face and draw a rectangle around it
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                cv2.imwrite(DEFAULT_OUTPUT_PATH + frameName + '.png', frame)

                print("Hello, what can I do for you? Recording starts…")
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                engine.setProperty("rate", 200)
                engine.setProperty("voice", voices[0].id)
                engine.say("Hello, what can I do for you? Recording starts…")
                engine.runAndWait()
                break

            # If 'esc' is hit, the video is closed. We only wait for a fraction of a second per loop
            if cv2.waitKey(1) == 27:
                break

        # When everything is done, release the capture
        self.videoSource.release()
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.waitKey(500)


def Parse():
    parser = argparse.ArgumentParser(
        description='Cascade Path for face detection')
    parser.add_argument('-i', '--input_path', type=str,
                        default=DEFAULT_CASCADE_INPUT_PATH, help='Cascade input path')
    parser.add_argument('-o', '--output_path', type=str,
                        default=DEFAULT_OUTPUT_PATH, help='Output path for pictures taken')
    args = parser.parse_args()
    return args


def ClearImageFolder():
    if not (os.path.exists(DEFAULT_OUTPUT_PATH)):
        os.makedirs(DEFAULT_OUTPUT_PATH)

    else:
        for files in os.listdir(DEFAULT_OUTPUT_PATH):
            filePath = os.path.join(DEFAULT_OUTPUT_PATH, files)
            if os.path.isfile(filePath):
                os.unlink(filePath)
            else:
                continue


def main():
    ClearImageFolder()

    # Instantiate Class object
    faceDetectImplementation = VideoCapture()

    # Call CaptureFrames from class to begin face detection
    faceDetectImplementation.CaptureFrames()


if __name__ == '__main__':
    main()
