from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

import cv2
import os

# Window

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(500, 500, 500, 500)
        self.setWindowTitle("GUI Face Recognition App")
        self.initUI()

    def initUI(self):
        self.button = QtWidgets.QPushButton(self)
        self.button.setText("SCAN")
        self.button.clicked.connect(self.clicked)
        self.button.move(200,100)

    def clicked(self):
        self.cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)

        screen = cv2.VideoCapture(0)

        while True:
            ret, frames = screen.read()
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Video', frames)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        screen.release()
        cv2.destroyAllWindows()



# Window Function

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()