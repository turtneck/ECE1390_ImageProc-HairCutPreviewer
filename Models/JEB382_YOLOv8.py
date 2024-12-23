# Written by Jonah Earl Belback

# YOLOv8n stable module container


#-------
#YOLO
#https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb#scrollTo=X58w8JLpMnjH
import ultralytics
from ultralytics import YOLO
from ultralytics import settings
#onnx
import onnxruntime,cv2,math
from PIL import Image

#------------------------
import os,sys,time,shutil
import matplotlib.pyplot as plt
import numpy as np

#------------------------

    
global JEB_YOLOHome
JEB_YOLOHome = (os.getcwd()+'/JonahBelback/').replace('\\','/')
# print(f"JEB_YOLOHome:\t\t<{JEB_YOLOHome}>")

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
            self.load_model(model_path)
            self.pretrain=True
        
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
    
    
    # ========================================
    def load_model(self,model_path):
        #load model but cant train
        if model_path[-5:] =='.onnx':
            print('From File:\t'+model_path)
            self.model = YOLO(model_path,verbose=self.verbose,task=self.model_type)
            
            # onnx runs differently
            #https://alimustoofaa.medium.com/how-to-load-model-yolov8-onnx-runtime-b632fad33cec
            self.opt_session = onnxruntime.SessionOptions()
            self.opt_session.enable_mem_pattern = False
            self.opt_session.enable_cpu_mem_arena = False
            self.opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list) #use this object to predict
            
            model_inputs = self.ort_session.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
            self.input_shape = model_inputs[0].shape

            model_output = self.ort_session.get_outputs()
            self.output_names = [model_output[i].name for i in range(len(model_output))]
            
            self.full_model = False
            print("SUCCESS: MODEL LOADED (.onnx)")
                
        #load model from file
        elif model_path[-3:] =='.pt':
            print('From File:\t'+model_path)
            self.model = YOLO(model_path,verbose=self.verbose)#,task='classify')
            self.full_model = True
            print("SUCCESS: MODEL LOADED (.pt)")
            
        #try to load from a directory
        else:
            print("Loading latest")
            latest_model = YOLOv8_find_latest(model_path)
            
            if latest_model == None: raise LookupError(f"Cannot find .pt in folder or best.pt in subfolders: {model_path}")
            
            try:
                #load model from file
                print('From Directory:\t'+latest_model)
                self.model = YOLO(latest_model,verbose=self.verbose)#,task='classify')
                self.full_model = True
                print("SUCCESS: MODEL LOADED (.pt from directory)")
            
            except Exception as e:
                print(str(e))
                os._exit()
            
    
    # ========================================
    # 'data' can filepath to an image file or a cv2 object ( such as from getFeed() )
    def run_model(self,data,conf_thres=0.8, type_comp=3, PIX_tol=10, PRC_tol=0.8, verbose=False):
        
        #if not onnx model
        if self.full_model:
            results = self.model(data,verbose=self.verbose or verbose)  # predict on an image file
            arr = []       
            for obj in results:
                if obj.boxes.xyxy.shape[0] == 0: continue #catch empty results
                xyxy = list(obj.boxes.xyxy.numpy()[0])
                arr.append( [   obj.names[ int(obj.boxes.cls.numpy()[0]) ],  [xyxy[:2],xyxy[2:]]   ] )
            
            #list of list of individual object properties [ classification, [Top right BB corod, Bottom Left] ]
            return arr
        
        #onnx model
        else:
            image_height, image_width = data.shape[:2]
            # data.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

            input_height, input_width = self.input_shape[2:]
            image_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(image_rgb, (input_width, input_height))

            # Scale input pixel value to 0 to 1
            input_image = resized / 255.0
            input_image = input_image.transpose(2,0,1)
            input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
            input_tensor.shape
            
            outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
            predictions = np.squeeze(outputs).T
            # Filter out object confidence scores below threshold
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > conf_thres, :]
            scores = scores[scores > conf_thres]
            
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            boxes = predictions[:, :4] # Get bounding boxes for each object

            #rescale box
            input_shape = np.array([input_width, input_height, input_width, input_height])
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([image_width, image_height, image_width, image_height])
            boxes = boxes.astype(np.int32)
            
            '''
            NOTE: at this point there are a number of predictions of detected objects above a certain confidence, !!! SOME ARE DUPLICATE !!!
            
            Removes "duplicate" detections based on tracing simualrity of bounding boxes:
              - typeR/type_comp = 0: boxes are simular if each corner is less then < PIX_tol > pixels from eachother:  SimBox_Corner(tolerance= PIX_tol)
                    - default
              - typeR/type_comp = 1: boxes are simular if the intersecting area is on average less then < PIX_tol >% of the area of box1 and box2:  SimBox_Area(tolerance= PRC_tol)
                    - make sure your using PRC_tol not PIX_tol
              - typeR/type_comp = 2: SimBox_Corner(tolerance= PIX_tol)      AND     SimBox_Area(tolerance= PRC_tol)
              - typeR/type_comp = 3: SimBox_Corner(tolerance= PIX_tol)      OR      SimBox_Area(tolerance= PRC_tol)
            
            does not check if simular bounding boxes are detecting the same class
            # '''
            reduced = [ find_list_in_LoL(boxes,non_sim) for non_sim in ReduceList(work=boxes,typeR=type_comp,PIX_tol=PIX_tol,PRC_tol=PRC_tol)  ]
            
            
            results=[]
            for ele in reduced:
                results.append(   [  self.classes[class_ids[ele]], [boxes[ele][:2].tolist(),boxes[ele][2:].tolist()]  ]   )
            return results
    
    
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
    def train_model(self,data_path,iter=1,opt=None,imgsize=None,rect=True,verbose=False):
        
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
                    verbose=self.verbose or verbose
                    )
            else:
                train_obj = self.model.train(
                    data=data_path,
                    epochs=iter,
                    # optimizer=opt,
                    pretrained=self.pretrain,
                    imgsz=imgsize,
                    rect=rect,
                    verbose=self.verbose or verbose
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
                    verbose=self.verbose or verbose
                    )
            else:
                train_obj = self.model.train(
                    data=data_path,
                    epochs=iter,
                    optimizer=opt,
                    pretrained=self.pretrain,
                    imgsz=imgsize,
                    rect=rect,
                    verbose=self.verbose or verbose
                    )
        
        self.pretrain=True
        return train_obj.save_dir


#==========================================================
#helper funcs specific to YOLO Models

#Finding Model File help
def YOLOv8_find_latest(folder_path):
    folder_list = os.listdir(folder_path)
    file_list = [ i for i in folder_list if i[-3:] == '.pt']
    folder_list.reverse()
    file_list.reverse()
    # print( folder_list )
    # print( file_list )

    if file_list: return f'{folder_path}/{file_list[0]}'

    for folder in folder_list:
        if os.path.isfile(f'{folder_path}/{folder}/weights/best.pt'): return f'{folder_path}/{folder}/weights/best.pt'
    return None

#ONNX help
def SimBox_Corner(box1,box2,tolerance=10):
    # print('a',box1,box2)
    check = abs(box1[0] - box2[0])<= tolerance
    check *= abs(box1[1] - box2[1])<= tolerance
    check *= abs(box1[2] - box2[2])<= tolerance
    check *= abs(box1[3] - box2[3])<= tolerance
    return check

def SimBox_Area(box1,box2,tolerance=0.9):
    L_inter = max(box1[0], box2[0])
    T_inter = max(box1[1], box2[1])
    R_inter = min(box1[2], box2[2])
    B_inter = min(box1[3], box2[3])
    
    if R_inter < L_inter or B_inter < T_inter: return False

    area3= (R_inter-L_inter)*(B_inter-T_inter)
    area1 = abs(box1[0]-box1[2]) * abs(box1[1]-box1[3])
    area2 = abs(box2[0]-box2[2]) * abs(box2[1]-box2[3])
    perc = ((area3/area2)+(area3/area1))/2 #avg percent simularity of 1-2 & 2-1
    
    print(perc,tolerance)
    return perc >= tolerance
    

def ReduceList(work, typeR=3, PIX_tol=10, PRC_tol=0.9):
    working=work.tolist()
    cnt=0;cnt2=cnt+1
    while cnt<len(working):
        while cnt2<len(working):
            if   typeR==0: del_bool = SimBox_Corner(   working[cnt], working[cnt2], PIX_tol   )
            elif typeR==1: del_bool = SimBox_Area  (   working[cnt], working[cnt2], PRC_tol   )
            elif typeR==2: del_bool = SimBox_Corner(   working[cnt], working[cnt2], PIX_tol   ) and SimBox_Area(   working[cnt], working[cnt2], PRC_tol   )
            elif typeR==3: del_bool = SimBox_Corner(   working[cnt], working[cnt2], PIX_tol   ) or  SimBox_Area(   working[cnt], working[cnt2], PRC_tol   )
            else: raise ValueError(f"typeR error: Must be 0,1,2,3\t typeR = <{typeR}>")
            
            if del_bool: working.pop(cnt2)
            else: cnt2+=1
        cnt+=1;cnt2=cnt+1
    return working

#find list in list of lists
def find_list_in_LoL(LoL,targ):
    for i, row in enumerate(LoL):
        if np.array_equal(row, targ):
            return i
    return -1


def reduce_filepath(file_path, coords, output_path, Expan_rate=0.7, Compress_rate=10):
    #read
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    #crop with expansion
    if Expan_rate>1: Expan_rate=Expan_rate/100 #if not floating, assume is percent
    I_height,I_width = img.shape[:2]
    delta_w = abs(coords[1][0]-coords[0][0]) * Expan_rate
    delta_h = abs(coords[1][1]-coords[0][1]) * Expan_rate
    new_x_min = int(max(0, coords[0][0] - delta_w // 2))
    new_y_min = int(max(0, coords[0][1] - delta_h // 2))
    new_x_max = int(min(I_width, coords[1][0] + delta_w // 2))
    new_y_max = int(min(I_height, coords[1][1] + delta_h // 2))
    img = img[ new_y_min:new_y_max, new_x_min:new_x_max ]
    
    #compress
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Compress_rate]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decoded_img = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    
    cv2.imwrite(output_path, decoded_img )
    
def reduce_ImgObj(img, coords, Expan_rate=0.7, output_path=None):
    #crop with expansion
    if Expan_rate>1: Expan_rate=Expan_rate/100 #if not floating, assume is percent
    I_height,I_width = img.shape[:2]
    delta_w = abs(coords[1][0]-coords[0][0]) * Expan_rate
    delta_h = abs(coords[1][1]-coords[0][1]) * Expan_rate
    new_x_min = int(max(0, coords[0][0] - delta_w // 2))
    new_y_min = int(max(0, coords[0][1] - delta_h // 2))
    new_x_max = int(min(I_width, coords[1][0] + delta_w // 2))
    new_y_max = int(min(I_height, coords[1][1] + delta_h // 2))
    img = img[ new_y_min:new_y_max, new_x_min:new_x_max ]
    
    #make square
    I_height,I_width,layers = img.shape
    size=max(I_height,I_width)
    blackness = np.zeros((size,size,layers), np.uint8)
    blackness[    (size-I_height)//2:((size-I_height)//2)+I_height, (size-I_width)//2:((size-I_width)//2)+I_width    ] = img
    
    
    if output_path: cv2.imwrite(output_path, blackness)
    return blackness


def square_img(img):
    I_height,I_width,layers = img.shape
    size=max(I_height,I_width)
    offsetX=(size-I_width)//2
    offsetY=(size-I_height)//2
    return img[     offsetX:(offsetX+I_width),  offsetY:(offsetY+I_height)     ]
    


#==========================================================
#Test Cases

'''
0: testcases
1: webcam
2: webcam with model
3: Training
'''
RunType=2


if __name__ == "__main__":
    if RunType==0:
        print("RUNTYPE 0")
        import time
        
        
        
        # init model outside container -----------------------------------------------
        print( "\n\ninit model outside container -----------------------------------------------" )
        test_model = YOLO(JEB_YOLOHome+"dataset/default_models/yolov8s.pt")
        
        
        
        
        
        # init model (s) -----------------------------------------------
        print( "\n\ninit model (s) -----------------------------------------------" )
        test_model = YOLO_model_v1(vers='s')
        print(f'model2 pretrain:\t{test_model.pretrain}')
        print(f'model2 full_model:\t{test_model.full_model}')
        # #training
        # start_time = time.time()
        # save_dir = test_model.train_model(JEB_YOLOHome+'dataset/face detection.v1i.yolov8/data.yaml',imgsize=[512,384])
        # end_time = time.time()
        # print(f'model2 pretrain:\t{test_model.pretrain}')
        # print(f'model2 Train Time:\t{end_time-start_time}')
        # print(f'model2 Train output:\t{save_dir}')
        #running
        start_time = time.time()
        result = test_model.run_model(JEB_YOLOHome+'dataset/TEST_example.jpg')
        end_time = time.time()
        print(f'model2 Runtime:\t{end_time-start_time}')
        print(f'model2 Run output:\t{result}')
        #running square
        start_time = time.time()
        result = test_model.run_model(JEB_YOLOHome+'dataset/TEST_example_square.jpg')
        end_time = time.time()
        print(f'model2 Sq Runtime:\t{end_time-start_time}')
        print(f'model2 Sq Run output:\t{result}')
        #saving
        start_time = time.time()
        result = test_model.save_model(JEB_YOLOHome+'dataset/TEST_initModelS.onnx')
        end_time = time.time()
        print(f'model2 Savetime:\t{end_time-start_time}')
        
        
        
        
        
        # loading model (.pt) -----------------------------------------------
        print( "\n\nloading model (.pt) -----------------------------------------------" )
        test_model = YOLO_model_v1(model_path=JEB_YOLOHome+'dataset/example_model.pt')
        print(f'model3 pretrain:\t{test_model.pretrain}')
        print(f'model3 full_model:\t{test_model.full_model}')
        # #training
        # start_time = time.time()
        # save_dir = test_model.train_model(JEB_YOLOHome+'dataset/face detection.v1i.yolov8/data.yaml',imgsize=[512,384])
        # end_time = time.time()
        # print(f'model3 pretrain:\t{test_model.pretrain}')
        # print(f'model3 Train Time:\t{end_time-start_time}')
        # print(f'model3 Train output:\t{save_dir}')
        #running
        start_time = time.time()
        result = test_model.run_model(JEB_YOLOHome+'dataset/TEST_example.jpg')
        end_time = time.time()
        print(f'model3 Runtime:\t{end_time-start_time}')
        print(f'model3 Run output:\t{result}')
        #running square
        start_time = time.time()
        result = test_model.run_model(JEB_YOLOHome+'dataset/TEST_example_square.jpg')
        end_time = time.time()
        print(f'model3 Sq Runtime:\t{end_time-start_time}')
        print(f'model3 Sq Run output:\t{result}')
        #saving
        start_time = time.time()
        result = test_model.save_model(JEB_YOLOHome+'dataset/TEST_LoadModPT.onnx',imgsz=[512,384])
        end_time = time.time()
        print(f'model3 Savetime:\t{end_time-start_time}')
        
        
        
        
        
        # loading model (.onnx) -----------------------------------------------
        print( "\n\nloading model (.onnx) -----------------------------------------------" )
        test_model = YOLO_model_v1(model_path=JEB_YOLOHome+'dataset/TEST_LoadModPT.onnx')
        print(f'model4 pretrain:\t{test_model.pretrain}')
        print(f'model4 full_model:\t{test_model.full_model}')
        #training (not allowed)
        try:
            save_dir = test_model.train_model(JEB_YOLOHome+'dataset/face detection.v1i.yolov8/data.yaml',imgsize=[512,384])
            raise KeyError("model4 NOT SUPPOSED TO BE ABLE TO TRAIN")
        except:
            print(f'model4 SUCCESS: cant train .onnx model')
        #running
        start_time = time.time()
        result = test_model.run_model(JEB_YOLOHome+'dataset/TEST_example.jpg')
        end_time = time.time()
        print(f'model4 Runtime:\t{end_time-start_time}')
        print(f'model4 Run output:\t{result}')
        #running square
        start_time = time.time()
        result = test_model.run_model(JEB_YOLOHome+'dataset/TEST_example_square.jpg')
        end_time = time.time()
        print(f'model4 Sq Runtime:\t{end_time-start_time}')
        print(f'model4 Sq Run output:\t{result}')
        #saving (not allowed)
        try:
            result = test_model.save_model(JEB_YOLOHome+'dataset/TEST_LoadModONNX.onnx')
            raise KeyError("model4 NOT SUPPOSED TO BE ABLE TO SAVE")
        except:
            print(f'model4 SUCCESS: cant save .onnx model')
        
        
        
        
        
        # loading model (folder) -----------------------------------------------
        print( "\n\nloading model (folder) -----------------------------------------------" )
        test_model = YOLO_model_v1(model_path=JEB_YOLOHome+'dataset')
        #demo of the compression
        result = test_model.run_model(JEB_YOLOHome+'dataset/TEST_example.jpg')
        print(f'model5 Run output:\t{result}')
        for i,res in enumerate(result):
            reduce_filepath(
                file_path=  JEB_YOLOHome+'dataset/TEST_example.jpg',
                coords=     res[1],
                output_path=    JEB_YOLOHome+f'dataset/Reduce_{i}__{res[0]}.jpg'
            )
    
    
    #==========================================================
    if RunType==1:
        print("RUNTYPE 1")
        
        capture = cv2.VideoCapture(1)
        
        while True:
            ret, frame = capture.read()
            if not ret: raise KeyError("Can't receive frame (stream end?)")
            if cv2.waitKey(1) == ord('q'): break
            
            cv2.imshow('frame; <q key> to quit', reduce_ImgObj(frame,[ [100,100], [500,420]]))
    
    
    #==========================================================
    if RunType==2:
        print("RUNTYPE 2")
        
        #load model
        Load_modelpath='FaceModel.pt'
        # Load_modelpath='FaceModel.onnx'
        
        test_model = YOLO_model_v1(model_path=JEB_YOLOHome+Load_modelpath)
        print(f'model3 pretrain:\t{test_model.pretrain}')
        print(f'model3 full_model:\t{test_model.full_model}')
        
        print('Video Capturing')
        capture = cv2.VideoCapture(1)
        while True:
            ret, frame = capture.read()
            if not ret: raise KeyError("Can't receive frame (stream end?)")
            if cv2.waitKey(1) == ord('q'): break
            
            # frame = square_img(frame)
            result = test_model.run_model(frame)
            for i,res in enumerate(result):
                # print( (res[1][0][0], res[1][0][1]), (res[1][1][0], res[1][1][1]) )
                cv2.rectangle(frame,     (  int(res[1][0][0]), int(res[1][0][1])  ), (  int(res[1][1][0]), int(res[1][1][1])  ),    (0, 255, 0),    2)
            
            # if result: cv2.imshow('frame; <q key> to quit', reduce_ImgObj(frame, result[0][1]) )
            cv2.imshow('frame; <q key> to quit', frame )
    
    
    #==========================================================
    if RunType==3:
        print("RUNTYPE 3")
        test_model = YOLO_model_v1(vers='s')
        #training
        start_time = time.time()
        save_dir = test_model.train_model(
                                        JEB_YOLOHome+'dataset/face detection.v1i.yolov8/data.yaml',
                                        rect=True,
                                        iter=20
                                        )
        end_time = time.time()
        print(f'model2 pretrain:\t{test_model.pretrain}')
        print(f'model2 Train Time:\t{end_time-start_time}')
        print(f'model2 Train output:\t{save_dir}')