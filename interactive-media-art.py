import cv2
from ultralytics import YOLO

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


    #사람 수 세기
    # num_objects = len(results[0].boxes.xyxy)

    # print("Number of recognized objects:", num_objects)

    x = int(results[0].boxes.xyxy[0][2]) - int(results[0].boxes.xyxy[0][0])
    y = int(results[0].boxes.xyxy[0][3]) - int(results[0].boxes.xyxy[0][1])
    
    #Unreal x, y의 값
    unreal_x = round((x-320)/320, 1)
    unreal_y = round(((y-240)/240)*(-1), 1)

    # print(unreal_x, unreal_y)

    annotated_frame = results[0].plot()

    cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()