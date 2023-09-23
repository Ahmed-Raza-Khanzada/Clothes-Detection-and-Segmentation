import os,shutil
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import tqdm
import glob
from ultralytics import YOLO
def detect(img_path,model_path="detection_v8_models/40epochsbest.pt"):
    # load pre-trained model
    # detection_model = YOLO("/kaggle/input/yolo-trained-model2/yolov8n.pt")
    model = YOLO(model_path)
    img = img_path
    if os.path.exists("runs/detect/predict"):
        shutil.rmtree("runs/detect/predict")
    image = cv2.imread(img)
    results=model.predict(source=img, conf=0.5, save=True, line_thickness=2, hide_labels=False)
    m_boxes = {"shirt":[],"jacket":[],"dress":[],"nodetect":[]}
    result = results[0]
    print("*"*40)
    if len(result.boxes) != 0:
        for poss,box in enumerate(result.boxes):
                c = result.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                # cords = [round(x) for x in cords]

                conf = round(box.conf[0].item(), 2)
                x,y,x2,y2 = cords[0],cords[1],cords[2],cords[3]
                x,y,x2,y2 = int(x),int(y),int(x2),int(y2)
                thresh = 40
                if x-thresh>=0:
                    x-=thresh-( (thresh//2)//2)
                if y-thresh>=0:
                    y-=thresh
                if x2+thresh+10<=image.shape[1]:
                    x2+=thresh+10
                elif x2+thresh<=image.shape[1]:
                     x2+=thresh
                if y2+(thresh-(thresh//2))<=image.shape[0]:
                    y2+=thresh-(thresh//2)
                img1 = image[y:y2, x:x2]
                # if not os.path.exists("runs/detect/predict"):
                #     os.mkdir("runs/detect/predict")
                cv2.imwrite(f'runs/detect/predict/{img.split("/")[-1].split(".")[0]}{poss}.jpg',img1)
                m_boxes[c].append(f'runs/detect/predict/{img.split("/")[-1].split(".")[0]}{poss}.jpg')     
                print(f"Detected {c} : {conf}\nDetections Results saved at: ",f'runs/detect/predict/{img.split("/")[-1].split(".")[0]}{poss}.jpg')
    else:
          m_boxes["nodetect"].append(img_path)     
    print("*"*40)
    return m_boxes
    # im = plt.imread(f'runs/detect/predict/{img.split("/")[-1]}')
    # plt.figure(figsize=(20,10))
    # plt.axis('off')
    # plt.imshow(im)