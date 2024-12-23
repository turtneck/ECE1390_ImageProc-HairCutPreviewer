from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt

class HomeScreen(QWidget):
    def __init__(self, parent):
        '''
        This class is the HomeScreen class which is a QWidget.

        :param parent: The parent widget to which this widget belongs.
        '''
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout()
        self.title = QLabel("just a LITTLE off the top")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 30px;")
        self.take_photo_btn = QPushButton("Capture Photo")
        self.upload_photo_btn = QPushButton("Upload Photo")
        self.take_photo_btn.clicked.connect(self.open_camera)
        self.upload_photo_btn.clicked.connect(self.upload_photo)
        self.layout.addWidget(self.title)
        self.layout.addStretch()
        self.layout.addWidget(self.take_photo_btn)
        self.layout.addWidget(self.upload_photo_btn)
        self.layout.addStretch()
        self.setLayout(self.layout)

    def open_camera(self):
        '''
        Open the camera screen.
        '''
        self.parent.stacked_widget.setCurrentWidget(self.parent.camera_screen)

    def upload_photo(self):
        '''
        Upload a photo from the file system.
        '''
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", "", "Images (*.png *.jpg *.jpeg)", options=options
        )
        if file_path:
            self.parent.BoundingBox_edit_screen.inital_set_image(file_path,box=None)
            self.parent.stacked_widget.setCurrentWidget(self.parent.BoundingBox_edit_screen)
