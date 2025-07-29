from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np
import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GaugeDetector:
    def __init__(self, model_path='best.pt'):
        """Initialize the gauge detector"""
        try:
            self.model = YOLO(model_path)
            self.result_buffer = deque(maxlen=7)
            self.camera_readings = {}  # Store readings for each camera
            self.camera_locks = {}     # Thread locks for each camera
            self.last_detections = {}  # Store last detection results with bounding boxes
            print("GaugeDetector initialized successfully")
        except Exception as e:
            print(f"Error initializing GaugeDetector: {e}")
            self.model = None

    def preprocess_frame(self, frame):
        """Improve image quality for better detection"""
        if frame is None:
            return None
            
        # Convert to grayscale for better contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Convert back to BGR
        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    def crop_center(self, frame, crop_width=600, crop_height=300):
        """Crop center of frame for better detection"""
        if frame is None:
            return None, None
            
        h, w = frame.shape[:2]
        
        # Calculate center crop coordinates
        center_x, center_y = w // 2, h // 2
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        x2 = min(w, x1 + crop_width)
        y2 = min(h, y1 + crop_height)
        
        # Adjust if crop goes out of bounds
        if x2 - x1 < crop_width:
            x1 = max(0, x2 - crop_width)
        if y2 - y1 < crop_height:
            y1 = max(0, y2 - crop_height)
            
        cropped = frame[y1:y2, x1:x2]
        crop_coords = (x1, y1, x2, y2)
        
        return cropped, crop_coords

    def filter_detections_by_size(self, detections):
        """Filter detections based on reasonable size"""
        if not detections:
            return detections

        # Calculate average size of number detections
        number_detections = [det for det in detections if det['class'] != '.']
        if not number_detections:
            return detections

        avg_width = np.mean([det['width'] for det in number_detections])
        avg_height = np.mean([det['height'] for det in number_detections])

        filtered_detections = []
        for det in detections:
            if det['class'] == '.':
                # Dot should be smaller than numbers but not too small
                if (det['width'] > avg_width * 0.1 and det['width'] < avg_width * 0.8 and
                        det['height'] > avg_height * 0.1 and det['height'] < avg_height * 0.6):
                    filtered_detections.append(det)
            else:
                # Numbers should have reasonable size
                if (det['width'] > avg_width * 0.3 and det['width'] < avg_width * 2.0 and
                        det['height'] > avg_height * 0.5 and det['height'] < avg_height * 1.5):
                    filtered_detections.append(det)

        return filtered_detections

    def smart_dot_filtering(self, detections):
        """Smart filtering for dots"""
        dots = [det for det in detections if det['class'] == '.']
        numbers = [det for det in detections if det['class'] != '.']

        if len(dots) <= 1:
            return detections

        # Sort by confidence first, then by position
        dots.sort(key=lambda x: (-x['confidence'], x['center_x']))

        filtered_dots = []
        for dot in dots:
            # Check if too close to already selected dot
            too_close = False
            for existing_dot in filtered_dots:
                distance = abs(dot['center_x'] - existing_dot['center_x'])
                if distance < 40:
                    too_close = True
                    break

            if not too_close:
                filtered_dots.append(dot)
                # Keep only one dot
                break

        return numbers + filtered_dots

    def validate_digit_sequence(self, detections):
        """Validate if digit sequence is logical"""
        if not detections:
            return detections

        # Sort by position
        detections.sort(key=lambda x: x['center_x'])
        classes = [det['class'] for det in detections]

        # Remove unreasonable patterns
        # 1. Can't have 2 dots
        dot_count = classes.count('.')
        if dot_count > 1:
            # Keep dot with highest confidence
            dots = [det for det in detections if det['class'] == '.']
            best_dot = max(dots, key=lambda x: x['confidence'])
            detections = [det for det in detections if det['class'] != '.' or det == best_dot]

        # 2. Dot can't be at start or end
        if detections and detections[0]['class'] == '.':
            detections = detections[1:]
        if detections and detections[-1]['class'] == '.':
            detections = detections[:-1]

        return detections

    def process_detections(self, results, crop_coords=None):
        """Process YOLO detection results"""
        if not self.model:
            return None
            
        for result in results:
            if len(result.boxes) == 0:
                return None

            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Adjust coordinates if we used cropping
                if crop_coords:
                    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                    x1 += crop_x1
                    y1 += crop_y1
                    x2 += crop_x1
                    y2 += crop_y1
                
                center_x = (x1 + x2) / 2
                cls_id = int(box.cls)
                conf = float(box.conf)
                cls_name = self.model.names[cls_id]

                # Adjusted thresholds
                if cls_name == '.':
                    min_conf = 0.2
                elif cls_name in ['1']:
                    min_conf = 0.15
                elif cls_name in ['7']:
                    min_conf = 0.2
                else:
                    min_conf = 0.3

                if conf > min_conf:
                    detections.append({
                        'class': cls_name,
                        'confidence': conf,
                        'center_x': center_x,
                        'bbox': (x1, y1, x2, y2),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })

            if not detections:
                return None

            # Apply filters
            detections = self.filter_detections_by_size(detections)
            detections = self.smart_dot_filtering(detections)
            detections = self.validate_digit_sequence(detections)

            if not detections:
                return None

            # Sort by position
            detections.sort(key=lambda x: x['center_x'])

            # Create number string
            number_str = ''.join([det['class'] for det in detections])

            # Post-processing
            number_str = number_str.strip('.')

            # Remove consecutive dots
            while '..' in number_str:
                number_str = number_str.replace('..', '.')

            try:
                if '.' in number_str and number_str.count('.') == 1:
                    parts = number_str.split('.')
                    if len(parts) == 2 and parts[0] and parts[1]:
                        return float(number_str), detections
                elif number_str and '.' not in number_str:
                    return int(number_str), detections
            except:
                pass

            return number_str, detections

    def get_stable_result(self, current_result, camera_id):
        """Get stable result using voting"""
        if current_result is None:
            return None

        # Initialize buffer for camera if not exists
        if camera_id not in self.camera_readings:
            self.camera_readings[camera_id] = {
                'buffer': deque(maxlen=7),
                'reading': 0,
                'confidence': 0,
                'last_update': time.time()
            }

        buffer = self.camera_readings[camera_id]['buffer']
        buffer.append(current_result[0])

        if len(buffer) < 4:
            return current_result

        # Get recent results
        recent_results = list(buffer)[-5:]

        # Find most common result
        from collections import Counter
        counter = Counter(recent_results)
        most_common, count = counter.most_common(1)[0]

        # Need at least 60% agreement
        if count >= len(recent_results) * 0.6:
            return most_common, current_result[1]
        else:
            return current_result

    def draw_detections(self, frame, detections, number):
        """Draw detection results with bounding boxes"""
        if frame is None or not detections:
            return frame
            
        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # Color coding by type
            if det['class'] == '.':
                color = (0, 255, 255)  # Yellow for dot
            elif det['class'] == '1':
                color = (255, 0, 255)  # Magenta for digit 1
            elif det['class'] in ['7']:
                color = (255, 165, 0)  # Orange for digit 7
            else:
                color = (0, 255, 0)    # Green for other digits

            # Thicker lines for digit 1 and dots
            thickness = 3 if det['class'] in ['1', '.'] else 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # Text with background
            text = f"{det['class']} ({det['confidence']:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (int(x1), int(y1 - 20)), (int(x1) + text_size[0], int(y1)), color, -1)
            cv2.putText(frame, text, (int(x1), int(y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Display result with background
        result_text = f"Digital: {number}"
        text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.rectangle(frame, (5, 20), (text_size[0] + 15, 55), (0, 0, 0), -1)
        cv2.putText(frame, result_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        return frame

    def detect_gauge_reading(self, frame, camera_id):
        """Detect gauge reading from frame"""
        if not self.model or frame is None:
            return

        try:
            # Initialize lock for camera if not exists
            if camera_id not in self.camera_locks:
                self.camera_locks[camera_id] = threading.Lock()

            with self.camera_locks[camera_id]:
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                if processed_frame is None:
                    return

                # Crop center for better detection
                cropped_frame, crop_coords = self.crop_center(processed_frame)
                if cropped_frame is None:
                    return

                # Run detection
                results = self.model.predict(cropped_frame, conf=0.15, iou=0.4, verbose=False)
                result = self.process_detections(results, crop_coords)

                if result:
                    stable_result = self.get_stable_result(result, camera_id)
                    if stable_result:
                        number, detections = stable_result
                        
                        # Store result
                        if camera_id not in self.camera_readings:
                            self.camera_readings[camera_id] = {}
                            
                        self.camera_readings[camera_id].update({
                            'reading': float(number) if isinstance(number, (int, float)) else 0,
                            'confidence': int(np.mean([det['confidence'] for det in detections]) * 100),
                            'unit': '',  # Digital gauges typically don't show units
                            'last_update': time.time()
                        })
                        
                        # Store detections for visualization
                        self.last_detections[camera_id] = {
                            'detections': detections,
                            'number': number,
                            'frame_shape': frame.shape
                        }

        except Exception as e:
            logger.error(f"Error detecting gauge reading for camera {camera_id}: {e}")

    def get_camera_reading(self, camera_id):
        """Get reading for specific camera"""
        if camera_id in self.camera_readings:
            return self.camera_readings[camera_id]
        return {'reading': 0, 'confidence': 0, 'unit': '', 'last_update': 0}

    def get_all_readings(self):
        """Get all camera readings"""
        return {cam_id: self.get_camera_reading(cam_id) for cam_id in self.camera_readings.keys()}

    def get_detection_overlay(self, frame, camera_id):
        """Get frame with detection overlay"""
        if camera_id in self.last_detections and frame is not None:
            detection_data = self.last_detections[camera_id]
            return self.draw_detections(frame.copy(), detection_data['detections'], detection_data['number'])
        return frame

    def clear_camera_data(self, camera_id):
        """Clear data for specific camera"""
        if camera_id in self.camera_readings:
            del self.camera_readings[camera_id]
        if camera_id in self.camera_locks:
            del self.camera_locks[camera_id]
        if camera_id in self.last_detections:
            del self.last_detections[camera_id]
