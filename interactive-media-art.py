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
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
# model.train(data="coco128.yaml", epochs=3)  # train the model
model.predict(classes=0, conf=0.8)

start = time.time()

#고개 돌리는 거 임의 값
count=0

user_eye_x = 0
user_eye_y = 0

user_neck_x = 0
user_neck_y = 0

#이전 위치값
pre_x = 0
pre_y = 0


while True:

    #message 초기화
    message = ''

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if not ret:
        break

    yolo_results = model(frame)

    #사용자 인식 있음
    if yolo_results[0]:
        
        end = time.time()

        #인식한 수 세기
        num_objects = len(yolo_results[0].boxes.xyxy)

        #면적 계산(가장 가까운 사람일수록 면적이 크다)
        min_distance = 100
        max_box = []
        x = 0
        y = 0
        for i in yolo_results[0].boxes.xyxy:
            x1 = i[0]
            y1 = i[1]
            x2 = i[2]
            y2 = i[3]

            now_x = int((((x2+x1)/2)-320)/320*100)
            now_y = int((((y2+y1)/2)-240)/240*(-100)+20)
            distance = int(((now_x-pre_x)**2 + (now_y-pre_y)**2)**(1/2))

            if distance < min_distance:
                min_distance = distance
                max_box=[]
                max_box.append(x1)
                max_box.append(y1)

                max_box.append(x2)
                max_box.append(y2)


        if len(max_box)!=0:
            x1 = int(max_box[0])
            x2 = int(max_box[2])

            y1 = int(max_box[1])
            y2 = int(max_box[3])

            x = int((((x2+x1)/2)-320)/320*100)
            # y = int(y1)
            y = int((((y2+y1)/2)-240)/240*(-100)+20)

            while True:
                #고개 동작할 때
                if end-start>=10:
                    #눈동자 동작
                    if int(x/10)==int(user_eye_x/10)==int(user_neck_x/10) and int(y/10)==int(user_eye_y/10)==int(user_neck_y/10):
                        user_eye_x = x
                        user_eye_y = y
                        user_neck_x = x
                        user_neck_y = y
                        message = f"{user_eye_x},{user_eye_y}.{user_neck_x},{user_neck_y}.1\n"
                        ser.write(message.encode())
                        break
                        
                    else:
                        #눈동자 동작
                        if int(x/10)==int(user_eye_x/10):
                            user_eye_x = x
                        elif x>user_eye_x:
                            user_eye_x += 5
                        elif x<user_eye_x:
                            user_eye_x -= 5
                        
                        if int(y/10)==int(user_eye_y/10):
                            user_eye_y = y
                        elif y>user_eye_y:
                            user_eye_y += 5
                        elif y<user_eye_y:
                            user_eye_y -= 5

                        #고개 동작
                        if int(x/10)==int(user_neck_x/10):
                            user_neck_x = x
                        elif x>user_neck_x:
                            user_neck_x += 1
                        elif x<user_neck_x:
                            user_neck_x -= 1
                        
                        if int(y/10)==int(user_neck_y/10):
                            user_neck_y = y
                        elif y>user_neck_y:
                            user_neck_y += 1
                        elif y<user_neck_y:
                            user_neck_y -= 1

                        message = f"{user_eye_x},{user_eye_y}.{user_neck_x},{user_neck_y}.1\n"
                        ser.write(message.encode())
                #고개 동작 안 할 때
                else:
                    #눈동자 동작
                    if int(x/10)==int(user_eye_x/10) and int(y/10)==int(user_eye_y/10):
                        user_eye_x = x
                        user_eye_y = y
                        message = f"{user_eye_x},{user_eye_y}.{user_neck_x},{user_neck_y}.0\n"
                        ser.write(message.encode())
                        break
                        
                    else:
                        #눈동자 동작
                        if int(x/10)==int(user_eye_x/10):
                            user_eye_x = x
                        elif x>user_eye_x:
                            user_eye_x += 1
                        elif x<user_eye_x:
                            user_eye_x -= 1
                        
                        if int(y/10)==int(user_eye_y/10):
                            user_eye_y = y
                        elif y>user_eye_y:
                            user_eye_y += 1
                        elif y<user_eye_y:
                            user_eye_y -= 1

                        message = f"{user_eye_x},{user_eye_y}.{user_neck_x},{user_neck_y}.0\n"
                        ser.write(message.encode())
                    
            cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)-20), 3, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    #사용자 인식 없음
    else:
        x = 0
        y = 0
        while True:
            #눈동자 목 모두 일치되면 정지
            if int(x/10)==int(user_eye_x/10)==int(user_neck_x/10) and int(y/10)==int(user_eye_y/10)==int(user_neck_y/10):
                user_eye_x = x
                user_eye_y = y
                user_neck_x = x
                user_neck_y = y
                message = f"{user_eye_x},{user_eye_y}.{user_neck_x},{user_neck_y}.0\n"
                ser.write(message.encode())
                break
                
            else:
                if int(x/10)==int(user_eye_x/10):
                    user_eye_x = x
                elif x<user_eye_x:
                    user_eye_x -= 1
                elif x>user_eye_x:
                    user_eye_x += 1
                
                if int(y/10)==int(user_eye_y/10):
                    user_eye_y = y
                elif y<user_eye_y:
                    user_eye_y -= 1
                elif y>user_eye_y:
                    user_eye_y += 1

                if int(x/10)==int(user_neck_x/10):
                    user_neck_x = x
                elif x<user_neck_x:
                    user_neck_x -= 1
                elif x>user_neck_x:
                    user_neck_x += 1
                
                if int(y/10)==int(user_neck_y/10):
                    user_neck_y = y
                elif y<user_neck_y:
                    user_neck_y -= 1
                elif y>user_neck_y:
                    user_neck_y += 1
                
                message = f"{user_eye_x},{user_eye_y}.{user_neck_x},{user_neck_y}.0\n"
                ser.write(message.encode())
    

        start = time.time()
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
ser.close()
cap.release()
cv2.destroyAllWindows()