from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2 as cv
from JonahBelback.JEB382_YOLOv8 import YOLO_model_v1 #Jonah

class CameraScreen(QWidget):
    def __init__(self, parent):
        '''
        This class is a QWidget that displays the camera feed and allows the user to capture a photo.

        :param parent: The parent widget
        '''
        super().__init__()
        self.parent = parent
        self.capture = None
        self.image = None
        self.layout = QVBoxLayout()
        self.label = QLabel("Camera View")
        self.label.setAlignment(Qt.AlignCenter)
        self.camera_view = QLabel()
        self.camera_view.setFixedSize(640, 480)
        self.camera_view.setStyleSheet("background-color: black;")
        self.capture_btn = QPushButton("Capture Photo")
        self.capture_btn.clicked.connect(self.capture_photo)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.camera_view)
        self.layout.addWidget(self.capture_btn)
        self.setLayout(self.layout)
        self.start_camera()
        
        self.box=None
        self.JEBModel = YOLO_model_v1(model_path='JonahBelback/FaceModel.pt')

    def start_camera(self):
        '''
        This method starts the camera feed.
        '''
        self.capture = cv.VideoCapture(1)
        self.timer = self.startTimer(30)

    def timerEvent(self, event):
        '''
        This method is called when the timer event is triggered. It reads the camera feed and displays it on the screen.

        :param event: The timer event
        '''
        if( not self.parent.stacked_widget.currentWidget() == self ): return #doesnt take camera image if not on the window
        
        ret, frame = self.capture.read()
        if ret:
            frame = cv.flip(frame, 1)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.image=cv.cvtColor(frame.copy(), cv.COLOR_RGB2BGR)
            
            #Adjust Window size
            height, width = frame.shape[:2]
            self.camera_view.setFixedSize(width, height)
            
            #Jonah- Model Detection
            results = self.JEBModel.run_model(frame)
            #No results
            if len(results)==0: self.box=None
            #One result, dont look for max size
            elif len(results)==1:
                self.box= [  [int(results[0][1][0][0]), int(results[0][1][0][1])  ], [  int(results[0][1][1][0]), int(results[0][1][1][1])  ]]
                cv.rectangle(frame,self.box[0], self.box[1],(0, 255, 0),2)
            #multiple results, look for max size
            elif len(results)>1:
                #get biggest
                resmax=0;idx=0
                for i,Res in enumerate(results):
                    if (Res[1][0][1]-Res[1][0][0])*(Res[1][1][1]-Res[1][0][1]) > resmax:
                        resmax = (Res[1][0][1]-Res[1][0][0])*(Res[1][1][1]-Res[1][0][1])
                        idx=i
                self.box= [  [int(results[idx][1][0][0]), int(results[idx][1][0][1])  ], [  int(results[idx][1][1][0]), int(results[idx][1][1][1])  ]]
                cv.rectangle(frame,self.box[0], self.box[1],(0, 255, 0),2)
                
                #highlights less faces in diff color
                for i,Res in enumerate(results):
                    if i==idx:continue
                    cv.rectangle(frame,[int(results[1][0][0]), int(results[1][0][1])  ], [  int(results[1][1][0]), int(results[1][1][1])  ],(71, 198, 252),2)
            
            #display image
            q_img = QImage(frame.data, width, height, 3*width, QImage.Format_RGB888).scaled(self.camera_view.size(), Qt.KeepAspectRatio)
            self.camera_view.setPixmap(QPixmap.fromImage(q_img))

    def capture_photo(self):
        '''
        This method captures a photo from the camera feed and saves it to a file.
        '''
        if not (self.image is None):  #if theres a detection
            cv.imwrite("captured_photo.jpg", self.image)
            self.killTimer(self.timer)
            self.capture.release()
            self.parent.BoundingBox_edit_screen.inital_set_image(self.image,self.box)
            self.parent.stacked_widget.setCurrentWidget(self.parent.BoundingBox_edit_screen)

    def closeEvent(self, event):
        '''
        This method is called when the widget is closed. It releases the camera feed.

        :param event: The close event
        '''
        if self.capture and self.capture.isOpened():
            self.capture.release()
        event.accept()
