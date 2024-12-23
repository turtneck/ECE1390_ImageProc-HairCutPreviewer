from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2 as cv
import numpy as np
from ApplyHairstyles import *

class ViewScreen(QWidget):
    def __init__(self, parent):
        '''
        This class displays the image and provides controls to apply hairstyles and filters.
        '''
        super().__init__()
        self.parent = parent
        self.current_image_path = None
        self.layout = QHBoxLayout()
        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.controls_layout = QVBoxLayout()
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Select Hairstyle", "Bald", "Buzz", "Mullet", "Waves"])
        self.apply_btn = QPushButton("Apply Style")
        self.apply_btn.clicked.connect(self.apply_hairstyle)
        self.new_photo_btn = QPushButton("Capture New Photo")
        self.new_photo_btn.clicked.connect(self.take_new_photo)
        self.upload_new_photo_btn = QPushButton("Upload New Photo")
        self.upload_new_photo_btn.clicked.connect(self.upload_new_photo)
        
        # Filter buttons
        self.grayscale_btn = QPushButton("Grayscale")
        self.grayscale_btn.clicked.connect(self.apply_grayscale_filter)
        self.sepia_btn = QPushButton("Sepia")
        self.sepia_btn.clicked.connect(self.apply_sepia_filter)
        self.negative_btn = QPushButton("Negative")
        self.negative_btn.clicked.connect(self.apply_negative_filter)
        
        self.controls_layout.addWidget(QLabel("Choose Hairstyle:"))
        self.controls_layout.addWidget(self.combo_box)
        self.controls_layout.addWidget(self.apply_btn)
        self.controls_layout.addStretch()
        self.controls_layout.addWidget(self.new_photo_btn)
        self.controls_layout.addWidget(self.upload_new_photo_btn)
        
        # Add filter buttons to layout
        self.controls_layout.addWidget(self.grayscale_btn)
        self.controls_layout.addWidget(self.sepia_btn)
        self.controls_layout.addWidget(self.negative_btn)
        
        self.layout.addWidget(self.image_label)
        self.layout.addLayout(self.controls_layout)
        self.setLayout(self.layout)

    def set_image(self, image_path):
        '''
        Set the image to be displayed on the screen.

        :param image_path: The path to the image file.
        '''
        self.current_image_path = image_path
        pixmap = QPixmap(image_path).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def apply_hairstyle(self):
        '''
        Apply the selected hairstyle to the image.
        '''
        hairstyle = self.combo_box.currentText()
        output_image_path = "boxed_photo.jpg"
        if hairstyle == "Select Hairstyle":
            pass
        elif hairstyle == "Bald":
            apply_bald_hairstyle(output_image_path)
        elif hairstyle == "Buzz":
            apply_buzz_hairstyle(output_image_path)
        elif hairstyle == "Mullet":
            apply_mullet_hairstyle(output_image_path)
        elif hairstyle == "Waves":
            apply_waves_hairstyle(output_image_path)

        if output_image_path:
            #cv.imwrite(output_image_path, cv.imread(self.current_image_path))
            self.set_image(output_image_path)

    def take_new_photo(self):
        '''
        Switch to the camera screen to capture a new photo.
        '''
        self.parent.camera_screen.start_camera()
        self.parent.stacked_widget.setCurrentWidget(self.parent.camera_screen)

    def upload_new_photo(self):
        '''
        Upload a new photo from the file system.
        '''
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", "", "Images (*.png *.jpg *.jpeg)", options=options
        )
        if file_path:
            #go back to bounding box edit
            self.parent.BoundingBox_edit_screen.inital_set_image(file_path,box=None)
            self.parent.stacked_widget.setCurrentWidget(self.parent.BoundingBox_edit_screen)

    def apply_grayscale_filter(self):
        '''
        Apply grayscale filter to the image.
        '''
        if self.current_image_path:
            image = cv.imread(self.current_image_path)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            output_image_path = "grayscale_image.jpg"
            cv.imwrite(output_image_path, gray_image)
            self.set_image(output_image_path)

    def apply_sepia_filter(self):
        '''
        Apply sepia filter to the image.
        '''
        if self.current_image_path:
            image = cv.imread(self.current_image_path)
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_image = cv.transform(image, sepia_filter)
            sepia_image = np.clip(sepia_image, 0, 255)
            output_image_path = "sepia_image.jpg"
            cv.imwrite(output_image_path, sepia_image)
            self.set_image(output_image_path)

    def apply_negative_filter(self):
        '''
        Apply negative filter to the image.
        '''
        if self.current_image_path:
            image = cv.imread(self.current_image_path)
            negative_image = cv.bitwise_not(image)
            output_image_path = "negative_image.jpg"
            cv.imwrite(output_image_path, negative_image)
            self.set_image(output_image_path)
