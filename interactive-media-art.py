import cv2
from ultralytics import YOLO
# import serial
# ser = serial.Serial('COM7', 9600)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    print(unreal_x, unreal_y)

    # frame = results[0].plot()

    #화면에 표시
    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()