import numpy as np
import cv2
import io
import os
from google.cloud import vision
from firebase_admin import storage

# LOAD YOLO MODEL
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_suppression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)).flatten()

    return boxes_np, confidences_np, index

def save_and_extract_text_from_plate(image, bbox, filename):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return None, None
    
    # Guardar la imagen de la placa en Firebase
    plate_filename = f'plate_{filename}'
    plate_url = upload_image_to_firebase(roi, plate_filename, 'plates')
    
    return plate_url, extract_text_from_image(roi)

def extract_text_from_image(image):
    client = vision.ImageAnnotatorClient.from_service_account_json('./my-project-91282-1695850580211-e3e78d970b84.json')

    _, encoded_image = cv2.imencode('.png', image)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        raw_text = texts[0].description.strip()
        filtered_text = ''.join([char for char in raw_text if char.isalnum()])
        limited_text = filtered_text[:6]
    else:
        limited_text = ''

    return limited_text

def drawings(image, boxes_np, confidences_np, index, filename):
    text_list = []
    plate_urls = []
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        plate_url, license_text = save_and_extract_text_from_plate(image, boxes_np[ind], filename)
        
        if plate_url:
            plate_urls.append(plate_url)

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y-30), (x+w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y+h), (x+w, y+h+30), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y+h+27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        text_list.append(license_text)

    return image, text_list, plate_urls

def yolo_predictions(img, net, filename):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    result_img, text, plate_urls = drawings(img, boxes_np, confidences_np, index, filename)
    return result_img, text, plate_urls

def object_detection(image, filename):
    image = np.array(image, dtype=np.uint8)
    result_img, text_list, plate_urls = yolo_predictions(image, net, filename)
    
    # Guardar la imagen procesada en Firebase
    result_filename = f'{filename}'
    result_url = upload_image_to_firebase(result_img, result_filename, 'results')
    
    return text_list, result_url, plate_urls

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def upload_image_to_firebase(image, filename, folder):
    bucket = storage.bucket()
    _, encoded_image = cv2.imencode('.png', image)
    content = encoded_image.tobytes()

    blob = bucket.blob(f'{folder}/{filename}')
    blob.upload_from_string(content, content_type='image/png')
    blob.make_public()
    return blob.public_url
