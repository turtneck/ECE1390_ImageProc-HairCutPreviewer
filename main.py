from JonahBelback.JEB382_YOLOv8 import YOLO_model_v1 #Jonah;  Need this here so it can imported in CameraScreen.py

import sys
sys.path.append('.')
from PyQt5.QtWidgets import QApplication
from Frontend.MainWindow import *

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()