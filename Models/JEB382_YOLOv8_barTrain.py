# Written by Jonah Earl Belback

# YOLOv8n stable module container


#-------
#YOLO
#https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb#scrollTo=X58w8JLpMnjH
import ultralytics
from ultralytics import YOLO
from ultralytics import settings
#onnx
import cv2

#------------------------
import os,sys,time,shutil
import numpy as np

#------------------------

    
global JEB_YOLOHome
# JEB_YOLOHome = (os.getcwd()+'/').replace('\\','/')
JEB_YOLOHome = (os.getcwd()+'/JonahBelback/').replace('\\','/')
print(f"JEB_YOLOHome:\t\t<{JEB_YOLOHome}>")

#------------------------




#===============================================================================
'''
vers: type of YOLOv8 model
    - n(nano)
    - s(mall), larger
    - else raises error
    
modelpath: either:
    - directpath to a YOLOv8 model
    or
    - its training folder: find latest 'best.pt' file
    
verbose: passed YOLOv8 attribute, deals with YOLO printouts

classes: list of classifier names
    - only used in .onnx

'''

class YOLO_model_v1:
    def __init__(self, vers='n', model_path=None,verbose=False,model_type='classify',classes=['Cardboard','glass','metal','paper','plastic']):
        self.verbose=verbose
        self.model_type=model_type
        self.classes=classes
        
        #check library
        ultralytics.checks()
        
        #---------------------------------------
        # NOTE: Model Creation
        
        #make new model
        if model_path == None:
            print("Creating new model")
            #load version
            if vers.lower() == 'n': self.model = YOLO(JEB_YOLOHome+"dataset/default_models/yolov8n.pt",verbose=self.verbose)#,task='classify')
            elif vers.lower() == 's': self.model = YOLO(JEB_YOLOHome+"dataset/default_models/yolov8s.pt",verbose=self.verbose)#,task='classify')
            else: raise ValueError(f"YOLOv8 version not found: <_{vers}_>")
            self.pretrain=False
            self.full_model = True
            
            print("SUCCESS: MODEL CREATED")
            
        #load latest model from a directory
        else:
            raise KeyError("CANT LOAD")
        
        #---------------------------------------    
        #disable bad API
        os.environ['WANDB_MODE'] = 'disabled'
        settings.update({"wandb": False})
        
        
        #---------------------------------------
        print("SUCCESS: MODEL INIT PASS")
            
    
    
    # ========================================
    def save_model(self,dir_path,imgsz=None):
        if not self.full_model: raise TypeError(f"Loaded Model is not full (.onnx not .pt): Cannot Save and no reason to")
        
        if imgsz==None: path = self.model.export(format='onnx',verbose=False)
        else: path = self.model.export(format='onnx',verbose=False,imgsz=imgsz)
    
        sp = dir_path.split('.')
        if len(sp)>1:
            if sp[-1] != 'onnx':
                #.(not onnx) file_path
                export = shutil.copyfile(path, '.'.join(sp[:-1])+'.onnx')
            else:
                #.onnx file_path (correct)
                export = shutil.copyfile(path, dir_path)
        else:
            #folder file_path
            export = shutil.copyfile(path, f'{dir_path}/Unnamed_save.onnx')
        #delete old file
        os.remove(path)
        
        print("SUCCESS: MODEL SAVED")
        return export
    
    
    
    '''
    opt: optimizer (honestly dont know the difference)
        - None: let YOlO decide which on to use
        - SGD:
        - Adam:
        - AdamW:
        - NAdam:
        - RAdam:
        - RMSProp:
    
    pretrained: if false, resets weights to random values
    
    imgsize: YOLOv8 uses a square image (if rect false), resizes. set this to you highest dimension in the image
    
    rect: if image is rectangular
    '''
    # ========================================
    #imgsize=[512,384]?
    def train_model(self,data_path,iter=1,opt=None,imgsize=None,rect=True,verbose=False, workers=6):
        
        if not self.full_model: raise TypeError(f"Loaded Model is not full (.onnx not .pt): Cannot *Train*, can only *Run*")
        
        # check optimizer is valid
        if opt not in [None,'SGD','Adam','AdamW','NAdam','RAdam','RMSProp']: raise ValueError(f"Optimizer not found: {opt}")
        
        # train the model
        if opt==None:
            if imgsize == None:
                train_obj = self.model.train(
                    data=data_path,
                    epochs=iter,
                    # optimizer=opt,
                    pretrained=self.pretrain,
                    # imgsz=imgsize,
                    rect=rect,
                    verbose=self.verbose or verbose,
                    workers=workers
                    )
            else:
                train_obj = self.model.train(
                    data=data_path,
                    epochs=iter,
                    # optimizer=opt,
                    pretrained=self.pretrain,
                    imgsz=imgsize,
                    rect=rect,
                    verbose=self.verbose or verbose,
                    workers=workers
                    )
        else:
            if imgsize == None:
                train_obj = self.model.train(
                    data=data_path,
                    epochs=iter,
                    optimizer=opt,
                    pretrained=self.pretrain,
                    # imgsz=imgsize,
                    rect=rect,
                    verbose=self.verbose or verbose,
                    workers=workers
                    )
            else:
                train_obj = self.model.train(
                    data=data_path,
                    epochs=iter,
                    optimizer=opt,
                    pretrained=self.pretrain,
                    imgsz=imgsize,
                    rect=rect,
                    verbose=self.verbose or verbose,
                    workers=workers
                    )
        
        self.pretrain=True
        return train_obj.save_dir


#==========================================================
#helper funcs specific to YOLO Models





#==========================================================
#Test Cases



    
print("RUNTYPE 3")
test_model = YOLO_model_v1(vers='s')
#training
start_time = time.time()
save_dir = test_model.train_model(
                                JEB_YOLOHome+'dataset/face detection.v1i.yolov8/data.yaml',
                                rect=True,
                                iter=20,
                                verbose=True
                                )
end_time = time.time()
print(f'model2 pretrain:\t{test_model.pretrain}')
print(f'model2 Train Time:\t{end_time-start_time}')
print(f'model2 Train output:\t{save_dir}')


#saving
start_time = time.time()
result = test_model.save_model(JEB_YOLOHome+'dataset/TEST_initModelS.onnx')
end_time = time.time()
print(f'model2 Save Time:\t{end_time-start_time}')