# USAGE in terminal: python3 yolo_opencv.py -i image.jpeg -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
import cv2
import argparse
import numpy as np
#ounces of CO2
carbonFootprint = {
    "person": 17280000, #https://www.pawprint.eco/eco-blog/average-carbon-footprint-globally
    "bicycle": 8480, #https://slate.com/technology/2011/08/how-soon-does-a-bike-pay-back-its-initial-carbon-footprint.html#:~:text=Independent%20analysts%20have%20used%20a,of%20greenhouse%20gases%20(PDF).
    "car": 1135822,#https://www.forbes.com/sites/jimgorzelany/2018/01/12/the-long-haul-15-vehicles-owners-keep-for-at-least-15-years/?sh=285e96cd6237  https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle#:~:text=typical%20passenger%20vehicle%3F-,A%20typical%20passenger%20vehicle%20emits%20about%204.6%20metric%20tons%20of,around%2011%2C500%20miles%20per%20year.
    "motorcycle":238095, #https://www.thrustcarbon.com/insights/how-to-calculate-motorbike-co2-emissions
    "airplane": 381487.9, #https://www.statista.com/statistics/829300/average-flight-hours-worldwide-business-aviation/#:~:text=Global%20business%20aviation%2D%20quarterly%20flight%20hours%20per%20aircraft%202014%2D2021&text=This%20statistic%20shows%20the%20average,an%20average%20of%2030.9%20hours. https://www.carbonindependent.org/22.html
    "bus": 881849.04874, #https://www.liveabout.com/buses-and-other-transit-lifetime-2798844 https://www.carbonindependent.org/20.html
    "toothbrush": 182.014,
    "hair drier": 9120,
    "teddy bear": 152.384,
    "scissors": 98.76,
    "vase": 172,
    "clock": 'N/A',
    "book": 35.27,
    "refrigerator": 60283.201,
    "sink": 32000,
    "traffic light":
    "fire hydrant":
    "stop sign":
    "parking meter":
    "bench":
    'bird':
    'cat':
    'dog':
    'horse':
    "sheep":
    'cow':
    'elephant':
    'bear':
    'zebra':
    'giraffe':
    'backpack':
    'umbrella':
    'handbag':
    'tie':
    'suitcase':
    'frisbee':
    'skis':
    'snowboard':
    'sports ball':
    'kite':
    'baseball bat':
    'baseball glove':
    'skateboard':
    'surfboard':
    'tennis racket':
    'bottle':
    'wine glass':
    'cup':
    'fork':
    'knife':
    'spoon':
    'bowl':
    'banana':
    'apple':
    'sandwich':
    'orange':
    'broccoli':
    'carrot':
    'hot dog':
    'pizza':
    'donut':
    'cake':
    'chair':
    'couch':
    'potted plant':
    'bed':
    'dining table':
    'toilet':
    'tv':
    'laptop':
    'mouse':
    'remote':
    'keyboard':
    'cell phone':
    'microwave':                 
    "oven":
    "toaster":







    



}

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(
    image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(
        x), round(y), round(x + w), round(y + h))

cv2.imshow("object detection", image)
cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
