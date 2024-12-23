# Written by Jonah Earl Belback

# Adjust the Bounding box of the face detection
# If was import, just draws general box


#-------
import sys
sys.path.append('.')
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QRect
import cv2 as cv
import numpy as np
from scipy.ndimage import label
from anchorPoints import findingAnchorPoints

class BoxEditScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout()
        #---
        self.label = QLabel("Please Adjust Bounding Box\nClick and Drag Corners :D")
        self.label.setAlignment(Qt.AlignCenter)
        #---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        #---
        self.ExpanRate=0
        self.EXPslider = QSlider(Qt.Horizontal)
        self.EXPslider.setMinimum(0);self.EXPslider.setMaximum(100);self.EXPslider.setValue(0)
        self.EXPslider.valueChanged.connect(self.AdjustSlider)
        #---
        self.capture_btn = QPushButton("Capture New Photo")
        self.capture_btn.clicked.connect(self.take_new_photo)
        #---
        self.confirm_btn = QPushButton("Confirm Bounding box")
        self.confirm_btn.clicked.connect(lambda: self.confirmed_box())
        #---
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.EXPslider)
        self.layout.addWidget(self.capture_btn)
        self.layout.addWidget(self.confirm_btn)
        self.setLayout(self.layout)
        
        #Drag Boxes
        self.crn_factor=10  #adjustable value, size of corners (pixels)
        self.blur_k = 15    #adjustable value, kernel size of blur
        self.mouse_start = None
        self.dragging = False
        self.expanding = False
        self.dragged_crn = None
        self.overlay_img = cv.imread('JonahBelback/omgFace.png', cv.IMREAD_UNCHANGED) #face overlay
            
    
    
    # ========================================
    def AdjustSlider(self):
        self.ExpanRate = self.EXPslider.value()/100
        self.update_image()
    
    
    # ========================================
    # ========================================
    #gives corners of box, used in self.mousePressEvent
    def corners(self):
        return [
            [ self.box[0][0],self.box[0][1] ],  #TL
            [ self.box[1][0],self.box[0][1] ],  #TR
            [ self.box[0][0],self.box[1][1] ],  #BL
            [ self.box[1][0],self.box[1][1] ]   #BR
        ]

    #goes off per click
    def mousePressEvent(self, event):
        pos = event.pos()-self.image_label.pos()
        # print(f'[{pos.x()}, {pos.y()}]\t\t{self.box}')

        #on corner, within factor
        for idx, corner in enumerate(self.corners()):
            if (    abs(pos.x()-corner[0]) <= self.crn_factor and
                    abs(pos.y()-corner[1]) <= self.crn_factor ):
                self.expanding = True
                self.dragged_crn = idx
                self.mouse_start = pos
                return
            
        #inside box
        if (pos.x() >= self.box[0][0] and pos.x() <= self.box[1][0] and
                pos.y() >= self.box[0][1] and pos.y() <= self.box[1][1] ):
            self.dragging = True
            self.mouse_start = pos
            
    
    
    #---------------------------------------
    #goes off whenever the mouse moves
    def mouseMoveEvent(self, event):
        # #only proceed if mouse is in the module, saves self.expanding
        # if not self.image_label.rect().contains(event.pos()): return
        
        pos = event.pos()-self.image_label.pos()
        
        #Resize self.box to drag_crn
        if self.expanding:
            if self.dragged_crn == 0:
                #Top Left, increase x1,y1
                self.box[0][0] = pos.x()
                self.box[0][1] = pos.y()
            elif self.dragged_crn == 1:
                #Top Right, increase x2,y1
                self.box[1][0] = pos.x()
                self.box[0][1] = pos.y()
            elif self.dragged_crn == 2:
                #Bot Left, increase x1,y2
                self.box[0][0] = pos.x()
                self.box[1][1] = pos.y()
            elif self.dragged_crn == 3:
                #Bot Right, increase x2,y2
                self.box[1][0] = pos.x()
                self.box[1][1] = pos.y()
        
        #Move the square
        elif self.dragging:
            dist = pos - self.mouse_start
            #check box wouldnt be outside widget, allow sliding
            #x's
            if self.box[0][0]+dist.x()<0:
                self.box[1][0]-= self.box[0][0]
                self.box[0][0]=0
            elif self.box[1][0]+dist.x()>self.pure_w:
                self.box[0][0]=self.pure_w-(self.box[1][0]-self.box[0][0])
                self.box[1][0]=self.pure_w
            else:
                self.box[0][0]+= dist.x()
                self.box[1][0]+= dist.x()
                
            #y's
            if self.box[0][1]+dist.y()<0:
                self.box[1][1]-= self.box[0][1]
                self.box[0][1]=0
            elif self.box[1][1]+dist.y()>self.pure_h:
                self.box[0][1]=self.pure_h-(self.box[1][1]-self.box[0][1])
                self.box[1][1]=self.pure_h
            else:
                self.box[0][1]+= dist.y()
                self.box[1][1]+= dist.y()
                
            self.mouse_start = pos

        self.update_image()
            
    
    #---------------------------------------
    #goes off whenever the mouse is released
    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.expanding = False
        self.dragged_crn = None
            
    
    
    # ========================================
    # ========================================
    def update_image(self):
        #------------
        #if box is beyond 
        
        try:
            #------------
            setting_image=self.pure_image.copy()
            
            
            #------------
            #Draw overlay!!!!
            overlay_resize = cv.resize(   self.overlay_img, ( abs(self.box[0][0]-self.box[1][0]), abs(self.box[0][1]-self.box[1][1]) )   )
            png_rgb = overlay_resize[:, :, :3] #RGB
            alpha_channel = overlay_resize[:, :, 3] #alpha (transparency)
            want = setting_image[ self.box[0][1]:self.box[1][1], self.box[0][0]:self.box[1][0], :  ]
            mask = np.stack([alpha_channel == 255] * 3, axis=-1)
            want[mask] = png_rgb[mask]
            setting_image[ self.box[0][1]:self.box[1][1], self.box[0][0]:self.box[1][0], :  ] = want
            
            #------------
            #get outside area of overlay!!!!!
            blurmask = np.zeros( (self.pure_h+2,self.pure_w+2), dtype=bool)
            blurmask[ self.box[0][1]+1:self.box[1][1]+1, self.box[0][0]+1:self.box[1][0]+1  ] = np.any(mask, axis=2)
            false_labels, num_false_components = label(~blurmask) #get components of mask
            # cv.imshow('~blurmask', (blurmask * 255).astype(np.uint8) )
            # print(f'false_labels:\t{num_false_components}')
            #   combo of all the masks not in the middle of the overlay (inside) and not the background:  mask of outside the overlay
            # print(setting_image.shape[:2],end='   ')
            # print(f'{blurmask.shape[:2]}, {num_false_components}<{np.unique(false_labels)}',end='   ')
            # print(  [abs(self.box[0][0]-self.box[1][0]), abs(self.box[0][1]-self.box[1][1])],       [(self.box[0][0]+self.box[1][0])//2,  (self.box[0][1]+self.box[1][1])//2]  )
            # blurmask1 =   (false_labels != false_labels[ ( self.box[0][0] +  self.box[1][0])//2, ( self.box[0][1] +  self.box[1][1])//2 ]) & (false_labels != 0)
            # blurmask2 = (false_labels == false_labels[ ( self.box[0][0] +  self.box[1][0])//2, ( self.box[0][1] +  self.box[1][1])//2 ]) & (false_labels != 0)
            # blurmask3 = (false_labels != false_labels[ ( self.box[0][0] +  self.box[1][0])//2, ( self.box[0][1] +  self.box[1][1])//2 ]) & ~blurmask
            # if self.box[0][0] == 0: blurmask = (false_labels != false_labels[ 0, 0 ]) & (~blurmask)
            # else: blurmask = false_labels == false_labels[ 0, 0 ]
            blurmask = (false_labels == false_labels[ 0, 0 ])[1:-1,1:-1]
            #   blur image at mask
            # setting_image = setting_image[blurmask] = cv.medianBlur(setting_image, self.blur_k)
            setting_image = np.where(blurmask[:,:,np.newaxis], self.blur_image, setting_image)
            
            #test
            # cv.circle(setting_image, ((self.box[0][0]+self.box[1][0])//2,  (self.box[0][1]+self.box[1][1])//2), 5, (255, 0, 255), -1)
            # cv.imshow('blurmask1', (blurmask1 * 255).astype(np.uint8) )
            # cv.imshow('blurmask2', (blurmask2 * 255).astype(np.uint8) )
            # blurmask3_img = (blurmask * 255).astype(np.uint8)
            # cv.circle(blurmask3_img, ((self.box[0][0]+self.box[1][0])//2,  (self.box[0][1]+self.box[1][1])//2), 5, (255, 0, 255), -1)
            # cv.imshow('blurmask3', blurmask3_img )

            
            #------------
            #Draw Expanded bounding box (blue)
            if self.ExpanRate>0:
                exp_box = self.expand_box()
                cv.rectangle(setting_image, exp_box[0], exp_box[1], (0, 0, 255),2)
            #------------
            #Draw bounding box (red)
            cv.rectangle(setting_image, self.box[0], self.box[1], (255, 0, 0),2)
            #------------
            #Draw Corner Circle
            for corner in self.corners(): cv.circle(setting_image, corner, self.crn_factor, (255, 0, 0), -1)
        
        except Exception as e:
            # print(f"ERROR\t{self.box}\n{e}")
            return
        #------------
        #display
        pixmap = QPixmap.fromImage(QImage(setting_image.data, self.pure_w, self.pure_h, 3*self.pure_w, QImage.Format_RGB888)).scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    #---------------------------------------
    #make box square, even scaling on all sides
    def square_box(self,box=None):
        #------------
        #use self.box
        if box is None:
            CTRx = (self.box[0][0] + self.box[1][0])//2
            CTRy = (self.box[0][1] + self.box[1][1])//2
            half_side = max(abs(self.box[1][0]-self.box[0][0]), abs(self.box[1][1]-self.box[0][1]))//2
            self.box = [    [CTRx-half_side, CTRy-half_side]  ,  [CTRx+half_side, CTRy+half_side]    ]
        #------------
        #use given box
        else:
            CTRx = (box[0][0] + box[1][0])//2
            CTRy = (box[0][1] + box[1][1])//2
            half_side = max(abs(box[1][0]-box[0][0]), abs(box[1][1]-box[0][1]))//2
            return [    [CTRx-half_side, CTRy-half_side]  ,  [CTRx+half_side, CTRy+half_side]    ]

    #---------------------------------------
    #expand box by percent
    def expand_box(self):
        box=self.square_box(self.box)#specify to return
        Exp = abs(box[1][0]-box[0][0]) * self.ExpanRate//2 #RAW space between new_box and old
        
        #Temper Expansion, find smallest infraction; if none then min is regExpand
        Exp = min(  Exp, box[0][0], box[0][1], self.pure_w-box[1][0], self.pure_h-box[1][1]  )

        # Adjust the bounding box coordinates to expand the box
        return [    [int( box[0][0]-Exp ),int( box[0][1]-Exp )],  [int( box[1][0]+Exp ),int( box[1][1]+Exp )]    ]
            

    #---------------------------------------
    #setting image on display from other modules
    def inital_set_image(self, image, box):        
        #------------
        #image path
        if isinstance(image, str): self.pure_image=cv.cvtColor(cv.imread(image), cv.COLOR_RGB2BGR)
        else: self.pure_image=cv.cvtColor(image.copy(), cv.COLOR_RGB2BGR)
        self.pure_h, self.pure_w =  self.pure_image.shape[:2]
        self.image_label.setFixedSize(self.pure_w, self.pure_h)
        self.blur_image = cv.medianBlur(self.pure_image, self.blur_k)
        
        #make new box thats just area on the image
        if box is None:
            side = min(self.pure_w,self.pure_h)
            self.box = [[int(side*.2), int(side*.2)],[int(side*.8), int(side*.8)]]
        #------------
        #given box
        else: self.box=box
        
        #------------
        #display
        self.update_image()
            
    
    # ========================================
    #Move to CameraScreen
    def take_new_photo(self):
        self.parent.camera_screen.start_camera()
        self.parent.stacked_widget.setCurrentWidget(self.parent.camera_screen)

    #---------------------------------------
    #Move to ViewEdit
    def confirmed_box(self,square=True):
        #------------
        #make bounding box square
        if square: self.square_box()
        
        #------------
        #crop then expand image if rate given
        #   Expan_rate= 0  :  no expansion
        #   Expan_rate= 0.7:  expansion by 70%
        exp_box = self.expand_box()
        setting_image = self.pure_image.copy()[ exp_box[0][1]:exp_box[1][1], exp_box[0][0]:exp_box[1][0], :  ]

        #------------
        #crop photo to bounding box, export, move on
        cv.imwrite("boxed_photo.jpg", cv.cvtColor(setting_image, cv.COLOR_RGB2BGR))
        self.parent.view_edit_screen.set_image("boxed_photo.jpg")
        #anchor point function
        anchorPoints = findingAnchorPoints("boxed_photo.jpg")
        #this prints out the (x,y) coordinates of the anchor points
        #print("Anchor Points: ", anchorPoints)
        self.parent.stacked_widget.setCurrentWidget(self.parent.view_edit_screen)
