from ultralytics import YOLO
import cv2
import numpy as np
import os


def detection_gauge_face(img, img_path, model_path, box_index=-1):
    model = YOLO(model_path)  # load model

    results = model(img)  # run inference

    #     print(results[0].boxes)

    boxes = results[0].boxes

    # Check if any detections were found
    if len(boxes) == 0:
        raise ValueError(f"No gauge faces detected in image: {img_path}")

    # Check if the requested box_index exists
    if box_index >= len(boxes):
        raise IndexError(f"box_index {box_index} out of range. Only {len(boxes)} detections found.")

    # assert len(boxes)>0, f"no bbox detected for image {img_path}"
    #     print(f"{len(boxes)} boxes were found")
    if box_index >= 0:
        m_box = boxes[box_index]
    else:
        m_box = boxes[0]
    return m_box.xyxy[0].int(), boxes


def crop_image(img, box):
    img = np.copy(img)
    cropped_img = img[box[1]:box[3],
                  box[0]:box[2], :]  # image has format [y, x, rgb]

    height = int(box[3] - box[1])
    width = int(box[2] - box[0])

    print(f"Height is {height}, Width is {width}")
    # want to preserve aspect ratio but make image square, so do padding
    if height > width:
        delta = height - width
        left, right = delta // 2, delta - (delta // 2)
        top = bottom = 0
    else:
        delta = width - height
        top, bottom = delta // 2, delta - (delta // 2)
        left = right = 0

    pad_color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return new_img


def process_image(img_path, model_path, box_index=-1):
    image = cv2.imread(img_path)

    # Gauge detection
    box, boxes = detection_gauge_face(image, img_path, model_path, box_index=box_index)

    # crop image to only gauge face
    cropped_img = crop_image(image, box)

    resolution = (448, 448)
    resized_img = cv2.resize(cropped_img, resolution, interpolation=cv2.INTER_LINEAR)

    return resized_img, boxes, image


def get_files_from_folder(folder):
    filenames = []
    for filename in os.listdir(folder):
        filenames.append(filename)
    return filenames


def crop_and_save_img(filename, src_dir, dest_dir, model_path, box_index=-1):
    img_path = src_dir + '/' + filename

    cropped_img, boxes, image = process_image(img_path, model_path, box_index)

    new_file_path = os.path.join(dest_dir, 'cropped_' + filename)
    cv2.imwrite(new_file_path, cropped_img)


image_directory = r'C:\Users\Admin\OneDrive\SecretDocs\HCMUT\Indefol\gauge.v3i.yolov8\valid\images'
new_image_directory = r'C:\Users\Admin\OneDrive\SecretDocs\HCMUT\Indefol\Gauge_Detection.v8i.yolov8\images_cropped_new_2'
model_path = r'C:\Users\Admin\OneDrive\SecretDocs\HCMUT\Indefol\Gauge_Detection.v8i.yolov8\runs\best.pt'


test_file_names = get_files_from_folder(image_directory)

os.makedirs(new_image_directory, exist_ok=True)

for filename in test_file_names:
    try:
        crop_and_save_img(filename, image_directory, new_image_directory, model_path)
        print(f"Successfully processed: {filename}")
    except (ValueError, IndexError) as e:
        print(f"Skipping {filename}: {str(e)}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
