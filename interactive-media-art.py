import cv2
from ultralytics import YOLO
import mediapipe as mp
import time
import serial

ser = serial.Serial('COM7', 9600)

mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness = 1, color = (0,0,255))

face_mesh = mp_face_mesh.FaceMesh(
    # max_num_faces=100,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model.train(data="coco128.yaml", epochs=3)  # train the model
model.predict(classes=0, conf=0.8)

start = time.time()

#고개 돌리는 거 임의 값
count=0

while True:
    ret, frame = cap.read()


    if not ret:
        break

    yolo_results = model(frame)

    #사용자 인식 있음
    if yolo_results[0]:
        
        end = time.time()

        #인식한 수 세기
        num_objects = len(yolo_results[0].boxes.xyxy)

        #면적 계산(가장 가까운 사람일수록 면적이 크다)
        max_area = 0
        max_box = []
        x = 0
        y = 0

        for i in yolo_results[0].boxes.xyxy:
            x1 = i[0]
            y1 = i[1]
            x2 = i[2]
            y2 = i[3]
            area = (int(x2)-int(x1))*(int(y2)-int(y1))
            
            if area > max_area:
                max_area = area
                max_box.append(x1)
                max_box.append(y1)

                max_box.append(x2)
                max_box.append(y2)
            
            
            if len(max_box)!=0:
                x1 = int(max_box[0])
                x2 = int(max_box[2])

                y1 = int(max_box[1])
                y2 = int(max_box[3])

                x = int((x2+x1)/2)
                y = int(y1)
                # y = int((y2+y1)/2)

                mes_x = str(x)+':\n'
                mes_y = str(y)+':\n'
                ser.write(mes_x.encode())
                ser.write(mes_y.encode())

                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #특징 추출
        mediaPipe_results = face_mesh.process(frame)

        #고개 돌리기
        if end-start>=3:
            #고개 돌리는 거 표시
            count=1
            #언리얼로 해결해야 할 점 값(값을 하나 지속적으로 보내면 되지 않을까)
            print("고개 돌리기")
            message = "neck:\n"
            ser.write(message.encode())

        #웃기
        if mediaPipe_results.multi_face_landmarks:
            # if count==1:
                print('웃기')
                message = "smile:\n"
                ser.write(message.encode())

    
    #사용자 인식 없음
    else:
        start = time.time()

        print("정지 이미지")
        message = "stop:\n"
        ser.write(message.encode())
        continue

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
ser.close()
cap.release()
cv2.destroyAllWindows()