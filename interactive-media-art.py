import cv2
from ultralytics import YOLO
import serial
import mediapipe as mp
import time

ser = serial.Serial('COM7', 9600)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness = 1, color = (0,0,255))

face_mesh = mp_face_mesh.FaceMesh(
    # max_num_faces=100,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)


start = time.time()

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model.train(data="coco128.yaml", epochs=3)  # train the model
model.predict(classes=0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    x = 0
    y = 0
    #사람 수 세기
    # num_objects = len(results[0].boxes.xyxy)

    # print("Number of recognized objects:", num_objects)
    x1 = int(results[0].boxes.xyxy[0][0])
    x2 = int(results[0].boxes.xyxy[0][2])

    y1 = int(results[0].boxes.xyxy[0][1])
    y2 = int(results[0].boxes.xyxy[0][3])

    x = int((results[0].boxes.xyxy[0][2] + results[0].boxes.xyxy[0][0])/2)
    y = int((results[0].boxes.xyxy[0][3] + results[0].boxes.xyxy[0][1])/2)
    
    #Unreal x, y의 값
    unreal_x = int(round((x-320)/320, 2)*50)
    unreal_y = int(round(((y-240)/240)*(-1), 2)*50)

    mes_x = str(unreal_x)+':\n'
    mes_y = str(unreal_y)+':\n'
    ser.write(mes_x.encode())
    ser.write(mes_y.encode())

    #특징 추출
    mediaPipe_results = face_mesh.process(frame)

    if mediaPipe_results.multi_face_landmarks:
        print('웃기')

    end = time.time()

    #고개 돌리기
    if end-start>=3:
        print("고개 돌리기")

    # frame = results[0].plot()

    #화면에 표시
    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()