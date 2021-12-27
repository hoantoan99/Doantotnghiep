import time
import cv2
import argparse
import numpy as np
import glob
import os
from keras.models import load_model
import io
from PIL import Image
from tool.config import Cfg
import torch
from tool.predictor import Predictor

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default="yolov4-tiny-custom.cfg", required=False,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default="weights/yolov4-tiny-custom_best.weights", required=False,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes',  default="yolo.names", required=False,
                help='path to text file containing class names')
ap.add_argument('-mo', '--model_orientation',  default="weights/classify_orientation.h5", required=False,
                help='path to classify orientation weights')
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def pred_info(net, image, classes):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

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
    results = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        img = image[int(y):int(y+h), int(x):int(x+w)]
        class_id = class_ids[i]
        label = str(classes[class_id])
        results.append((img, label))
    return results

def resize_pad(im, img_size):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

def pred_orientation(img, model, img_size):
    img = resize_pad(img,img_size).astype(np.float32)/255
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.argmax(pred)
    return pred

def main(image):
    # load model recor
    config = Cfg.load_config_from_name('vgg_transformer')
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"&\'()+,-./:;= '
    config['weights'] = 'weights/transformerocr.pth'
    # config['weights'] = '/home/v000354/Downloads/transformerocr_ben.pth'
    config['device'] = 'cpu'
    config['predictor']['beamsearch']=False
    device = config['device']
    detector = Predictor(config)

    # load model yolo
    net = cv2.dnn.readNet(args.weights, args.config)
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    results_info = pred_info(net, image, classes)

    #load model classify orientation
    model_orientation = load_model(args.model_orientation)
    classes_orientation = ["rotate_0", "rotate_90", "rotate_180", "rotate_270"]
    ft_extracts = []
    for result in results_info:
        img, label = result
        orientation = classes_orientation[pred_orientation(img, model_orientation, img_size=96)]
        # print(label, orientation)
        if orientation == "rotate_90":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == "rotate_270":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == "rotate_180":
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            img = img
        img = Image.fromarray(img)
        s = detector.predict(img)
        ft_extracts.append(label, s)
        # print(s, orientation)
        # img.show()
    s_seller ="SELLER: "
    s_address ="ADDRESS: "
    s_timestamp ="TIMESTAMP: "
    s_total ="TOTAL_COST: "
    for label, text in ft_extracts:
        if label=="SELLER":
            s_seller += text+"\n"
        if label=="ADDRESS":
            s_address += text+"\n"
        if label=="TIMESTAMP":
            s_timestamp += text+"\n"
        if label=="TOTAL_COST":
            s_total += text +"\n"

    final_results = s_seller+"\n"+s_address+"\n"+s_timestamp+"\n"+s_total
    print(final_results)

image = cv2.imread("E:\KLTN\yolo\data/mcocr_public_145013bldqx.jpg")
main(image)