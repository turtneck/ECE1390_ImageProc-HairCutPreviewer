import sys
sys.path.append('.')
from Frontend.HomeScreen import *
from Frontend.CameraScreen import *
from Frontend.ViewScreen import *
from Frontend.BoxEditScreen import *
from PyQt5.QtWidgets import QWidget, QVBoxLayout,QStackedWidget

class MainWindow(QWidget):
    def __init__(self):
        '''
        This class is the MainWindow class which is a QWidget.
        '''
        super().__init__()
        self.setWindowTitle("just a LITTLE off the top")
        self.setGeometry(100, 100, 800, 600)
        self.stacked_widget = QStackedWidget()
        self.layout = QVBoxLayout()
        self.home_screen = HomeScreen(self)
        self.camera_screen = CameraScreen(self)
        self.view_edit_screen = ViewScreen(self)
        self.BoundingBox_edit_screen = BoxEditScreen(self)
        self.stacked_widget.addWidget(self.home_screen)
        self.stacked_widget.addWidget(self.camera_screen)
        self.stacked_widget.addWidget(self.view_edit_screen)
        self.stacked_widget.addWidget(self.BoundingBox_edit_screen)
        self.layout.addWidget(self.stacked_widget)
        self.setLayout(self.layout)
