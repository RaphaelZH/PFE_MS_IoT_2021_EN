import cv2
import argparse
import os
from datetime import datetime
import time
from pygame import mixer
import speech_recognition as sr
from gtts import gTTS
# quiet the endless 'insecurerequest' warning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

                self.SpeechSynthesis()
                self.SpeechRecognition()
                break

            # If 'esc' is hit, the video is closed. We only wait for a fraction of a second per loop
            if cv2.waitKey(1) == 27:
                break

        # When everything is done, release the capture
        self.videoSource.release()
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.waitKey(500)

    def SpeechSynthesis(self):
        mixer.init()
        tts = gTTS(
            text="Hello, what can I do for you? Recording starts…", lang='en')
        tts.save("question.mp3")
        mixer.music.load('question.mp3')
        mixer.music.play()

    def SpeechRecognition(self):
        mixer.init()
        while (True == True):
            # obtain audio from the microphone
            r = sr.Recognizer()
            with sr.Microphone() as source:
                #print("Please wait. Calibrating microphone...")
                # listen for 1 second and create the ambient noise energy level
                r.adjust_for_ambient_noise(source, duration=1)
                print("Hello, what can I do for you? Recording starts…")
                audio = r.listen(source, phrase_time_limit=5)

        # recognize speech using Sphinx/Google
            try:
                #response = r.recognize_sphinx(audio)
                response = r.recognize_google(audio)
                print("I think you said '" + response + "'")
                tts = gTTS(text="I think you said " + str(response), lang='en')
                tts.save("response.mp3")
                mixer.music.load('response.mp3')
                mixer.music.play()
                pass

            except sr.UnknownValueError:
                print("Sphinx could not understand audio")
            except sr.RequestError as e:
                print("Sphinx error; {0}".format(e))


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
