# requirements: pip install ultralytics opencv-python pytesseract imutils
import cv2,torch,pytesseract,numpy as np
from ultralytics import YOLO

m=YOLO('yolov8n.pt')          # your trained plate detector
cap=cv2.VideoCapture(0)    # or path to video
def warp_plate(img,box):
    x1,y1,x2,y2=[int(v) for v in box]
    w=h=max(80,x2-x1)
    crop=img[y1:y2,x1:x2]
    return cv2.resize(crop,(w,h))
def ocr_plate(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g=cv2.bilateralFilter(g,9,75,75)
    _,th=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cfg='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(th,config=cfg).strip()
while True:
    ret,frame=cap.read()
    if not ret: break
    res=m(frame,stream=True)          # detector inference
    for r in res:
        for box in r.boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2=box
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            plate=warp_plate(frame,box)
            text=ocr_plate(plate)
            cv2.putText(frame,text,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    cv2.imshow('LPR',frame)
    if cv2.waitKey(1)==27: break
cap.release();cv2.destroyAllWindows()
