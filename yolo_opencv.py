# USAGE in terminal: python3 yolo_opencv.py -i image.jpeg -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
import cv2
import argparse
import numpy as np
import qrcode

alternative_items = {
    'toothbrush': (182, 'bamboo toothbrush', "https://www.independent.co.uk/extras/indybest/fashion-beauty/best-bamboo-toothbrushes-plastic-pollution-biodegradable-bistles-dental-care-eco-friendly-a8411536.html"),
    'car': (1135822, 'public transport', 'https://www.visitphilly.com/getting-around/'),
    'motorcycle': (238095,'public transport', 'https://www.visitphilly.com/getting-around/'),
    'bus': (881849,'public transport', 'https://www.visitphilly.com/getting-around/'),
    'train': (0, 'public transport', 'https://www.visitphilly.com/getting-around/'),
    'cup': (15, 'compostable cups', 'https://citizensustainable.com/eco-friendly-cups/'),
    'refrigerator': (60283,'earthen pot', 'https://en.wikipedia.org/wiki/Pot-in-pot_refrigerator'),
    'person': (17280000,'reducing your footprint' ,'https://europa.eu/youth/get-involved/sustainable-development/how-reduce-my-carbon-footprint_en'),
    'bottle': (3, 'reusable water bottle', 'https://www.mindbodygreen.com/articles/philanthropic-water-bottles'),
    'cell phone': (2400,'easy to repair cell phones', 'https://www.fairphone.com/en/'),
    'airplane': (381487,'reducing flight footprint', 'https://carbonfund.org/how-to-offset-the-carbon-footprint-of-flying/'),
    'sports ball': (30,'Donate to freecycle', 'https://www.freecycle.org/'),
    'book': (35,'Donate to freecycle', 'https://www.freecycle.org/'),
    'clock': (122, 'Donate to freecycle', 'https://www.freecycle.org/'),
    'vase': (172,'Donate to freecycle', 'https://www.freecycle.org/'),
    'scissors': (98,'Donate to freecycle', 'https://www.freecycle.org/'),
    'teddy bear': (152,'Donate to freecycle', 'https://www.freecycle.org/'),
    'keyboard': (360,'Donate to freecycle', 'https://www.freecycle.org/'),
    'mouse': (280,'Donate to freecycle', 'https://www.freecycle.org/'),
    'remote': (200,'Donate to freecycle', 'https://www.freecycle.org/'),
    'chair' : (1500, 'Donate to freecycle', 'https://www.freecycle.org/'),
    'couch' : (3000,'Donate to freecycle', 'https://www.freecycle.org/'),
    'backpack' : (600,'Donate to freecycle', 'https://www.freecycle.org/'),
    'microwave': (27513,'Recycling', 'https://www.phila.gov/programs/recycling-program/'),
    'oven': (45150, 'Recycling', 'https://www.phila.gov/programs/recycling-program/'),
    'toaster': (5940,'Recycling', 'https://www.phila.gov/programs/recycling-program/'),
    'laptop': (14938,'Recycling', 'https://www.phila.gov/programs/recycling-program/'),
}

ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
ap.add_argument('-c', '--config', required=False, default='yolov3.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=False, default='yolov3.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=False, default='yolov3.txt',
                help='path to text file containing class names')
args = ap.parse_args()


all_labels = []
total_footprint = 0

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, num):
    label = str(classes[class_id])
    global all_labels, total_footprint
    for item in list(alternative_items.keys()):
        if item == label and item not in all_labels:
            all_labels.append(label)
            total_footprint += alternative_items[item][0]
            qr_pil = qrcode.make(alternative_items[item][2])
            qr_pil.save('qr.png')
            qr = cv2.resize(cv2.imread('qr.png'),(100,100))
            x_offset = 30
            y_offset = 80 + 100 * num
            x_end = x_offset + qr.shape[1]
            y_end = y_offset + qr.shape[0]
            white_background[y_offset:y_end, x_offset:x_end] = qr
            text = item + ": " + alternative_items[item][1]
            cv2.putText(white_background, text, (x_offset+110, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cap = cv2.VideoCapture(0)
white_background = np.zeros([720,520,3],dtype=np.uint8)
white_background.fill(255)
while True:
    isTrue, image = cap.read()
    final = np.concatenate((image, white_background), axis=1)
    cv2.imshow('Video', final)
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
cap.release()
cv2.destroyAllWindows()

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

for num, i in enumerate(indices):
    i = i
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(
        x), round(y), round(x + w), round(y + h), num)

cv2.putText(white_background, "Total Carbon Footprint: " + str(total_footprint) + " ounces of CO2", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
final = np.concatenate((image, white_background), axis=1)
cv2.imshow("object detection", final)
cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
