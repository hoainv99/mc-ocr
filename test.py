import re
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from utils import rotate_box,align_box,get_idx
from modules.text_detect.predict import test_net,net,refine_net,poly
from modules.image_clf.predict import Predictor_image
from PIL import Image
# from modules.text_recognition.predict import text_recognizer
from modules.text_clf.svm import predict_svm
from modules.text_clf.phoBert import predict_phoBert
from modules.text_clf.regex import date_finder
import torch

from modules.image_segmentation.predict import  segment_single_images

import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import joblib

def test(image):
    image = segment_single_images(image)
    image_copy = image.copy()
    bboxes, polys, score_text = test_net(net, image_copy, 0.7, 0.4, 0.4, True, poly, refine_net)
    if bboxes!=[]:
        
        bboxes_xxyy = []
        ratios = []
        degrees = []
        for box in bboxes:
            x_min = min(box, key=lambda x: x[0])[0]
            x_max = max(box, key=lambda x: x[0])[0]
            y_min = min(box, key=lambda x: x[1])[1]
            y_max = max(box, key=lambda x: x[1])[1]
            if (x_max-x_min) > 20:
                ratio = (y_max-y_min)/(x_max-x_min)
                ratios.append(ratio)
            

        mean_ratio = np.mean(ratios) 
        if mean_ratio>=1:   
            image,bboxes = rotate_box(image,bboxes,None,True,False)
        
        if predict_image(image) == 1:
            image,bboxes = rotate_box(image,bboxes,None,False,True)

        bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, True, poly, refine_net)

        image,check = align_box(image,bboxes,skew_threshold=0.9)
        
        if check:
            bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, True, poly, refine_net)
        h,w,c = image.shape

        for box in bboxes:

            x_min = max(int(min(box, key=lambda x: x[0])[0]),1)
            x_max = min(int(max(box, key=lambda x: x[0])[0]),w-1)
            y_min = max(int(min(box, key=lambda x: x[1])[1]),3)
            y_max = min(int(max(box, key=lambda x: x[1])[1]),h-2)    
            bboxes_xxyy.append([x_min-1,x_max,y_min-1,y_max])

        img_copy = image.copy()
        for b in bboxes_xxyy:
            cv2.rectangle(img_copy, (b[0],b[2]),(b[1],b[3]),(255,0,0),1)
        plt.figure(figsize=(10,10))
        plt.imshow(img_copy)
        texts = []
        probs = []
        for box in bboxes_xxyy:
            x_min,x_max,y_min,y_max = box
            img = image[y_min:y_max,x_min:x_max,:]
            img = Image.fromarray(img)
            s,prob = text_recognizer.predict(img,return_prob = True)
            texts.append(s)
            probs.append(prob)
            
        out,score = predict_svm(texts[10:])
        out_bert = predict_phoBert(texts[:10])

        rs_text = ""
        t_seller = None

        if len(np.where(out_bert==0)[0])!=0:
            seller_idx = np.where(out_bert==0)[0][0].item()
            text_seller = texts[seller_idx]
            rs_text += text_seller+"|||"
        else:
            rs_text += "|||"
        if len(np.where(out_bert==1)[0])!=0:
            add_idx = np.where(out_bert==1)[0]
            txt_address=""
            for idx in range(len(add_idx)):
                txt_address += texts[add_idx[idx].item()]
                if idx < len(add_idx)-1:
                    txt_address += " "
            rs_text += txt_address+ "|||"
        else:
            rs_text += "|||"
        date_str = None
        for idx,string in enumerate(texts):
    #                     if re.search("Ngày",string):
    #                         start_idx = re.search("Ngày",string).start() 
    #                         date_str = string[start_idx:]
                if date_finder(string):
                    date_str = string
                    date_idx = idx
                    if len(list(string)) > 30:
                        if re.search("Ngày",string):
                            start_idx = re.search("Ngày",string).start() 
                            date_str = date_str[start_idx:]
        for idx,string in enumerate(texts):
            if re.search("Ngay",string):
                start_idx = re.search("Ngay",string).start() 
                date_str = string[start_idx:]
        for idx,string in enumerate(texts):
            if re.search("Ngày",string):
                start_idx = re.search("Ngày",string).start() 
                date_str = string[start_idx:]
        if date_str:

            rs_text += date_str+"|||"
        else:
            rs_text += "|||"

        cost_idx = get_idx(out,score,0)

        if cost_idx!=None:
            txt_cst = texts[cost_idx]                  
            rs_text += txt_cst

            cst1_xmin,cst1_xmax,cst1_ymin,cst1_ymax = bboxes_xxyy[cost_idx]
            cst1_ycenter = (cst1_ymin+cst1_ymax)/2

            for box in bboxes_xxyy:
                if box == bboxes_xxyy[cost_idx]:
                    continue
                x_min,x_max,y_min,y_max = box
                if abs(cst1_ycenter-(y_max+y_min)/2)<13:
                    img = image[y_min:y_max,x_min:x_max,:]
                    img = Image.fromarray(img)
                    s,prob = text_recognizer.predict(img,return_prob = True)
                    rs_text +=" "+ s
        inp_task1 = []
        for prob in probs:
            if np.isnan(prob):
                continue
            inp_task1.append(prob)
        inp_task1 = sorted(inp_task1)
        if len(inp_task1)>100:
            inp1 = np.array(inp_task1[:100])
        else:
            inp1 = np.concatenate((inp_task1,np.zeros(100 - len(inp_task1), dtype=np.float32)))
            
        out_task1 = model_task1.predict([inp1])  
        out_task1 = out_task1.item()
    else:
        out_task1 = 0.2
        rs_text = "|||||||||"
    return rs_text,out_task1

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='MC-OCR inference')
    parser.add_argument('--folder_test', type=str, help='path to folder')
    args = parser.parse_args()


    predict_svm = predict_svm()
    predict_phoBert = predict_phoBert()
    predict_image = Predictor_image()
    #recognition
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'weights/transformerocr.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False
    text_recognizer = Predictor(config)
    model_task1 = joblib.load("weights/model_task1.sav")
    with open('results.csv', mode='w') as csv_file:
        fieldnames = ['img_id', 'anno_image_quality', 'anno_texts']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
        writer.writeheader()
        for file_name in os.listdir(args.folder_test):
            image = cv2.imread(os.path.join(args.folder_test,file_name))
            rs_text,out_task1 = test(image)
            writer.writerow({'img_id': file_name, 'anno_image_quality': out_task1, 'anno_texts': rs_text})




