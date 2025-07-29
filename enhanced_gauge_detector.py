from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np
import threading
import time
import logging
import os
import tempfile
from PIL import Image



# Import the analog gauge pipeline
try:
    import sys
    import os
    # Add the gauge_reader_web directory to Python path
    gauge_reader_path = os.path.join(os.path.dirname(__file__), 'gauge_reader_web')
    if gauge_reader_path not in sys.path:
        sys.path.append(gauge_reader_path)
    
    from pipeline import process_image
    ANALOG_PIPELINE_AVAILABLE = True
    print("Analog gauge pipeline imported successfully")
except ImportError as e:
    ANALOG_PIPELINE_AVAILABLE = False
    print(f"Analog gauge pipeline not available: {e}")
    logging.warning("Analog gauge pipeline not available. Only digital gauges will be supported.")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EnhancedGaugeDetector:
    def __init__(self, digital_model_path='best.pt', analog_models_config={
    'detection_model': 'gauge_reader_web/models/gauge_detection_model.pt',
    'key_point_model': 'gauge_reader_web/models/keypoint_model.pt',  # Your actual filename
    'segmentation_model': 'gauge_reader_web/models/needle_segmentation_model.pt'  # Your actual filename
}):
        """
        Initialize the enhanced gauge detector for both digital and analog gauges
        
        Args:
            digital_model_path: Path to YOLO model for digital gauge detection
            analog_models_config: Dict with paths to analog gauge models:
                {
                    'detection_model': 'path/to/detection_model.pt',
                    'key_point_model': 'path/to/key_point_model.pt', 
                    'segmentation_model': 'path/to/segmentation_model.pt'
                }
        """
        try:
            # Initialize digital gauge detector
            self.digital_model = YOLO(digital_model_path)
            self.result_buffer = deque(maxlen=7)
            self.camera_readings = {}
            self.camera_locks = {}
            self.last_detections = {}
            
            # Initialize analog gauge models
            self.analog_models_config = analog_models_config or {
                'detection_model': 'gauge_reader_web/models/gauge_detection_model.pt',
                'key_point_model': 'gauge_reader_web/models/key_point_model.pt',
                'segmentation_model': 'gauge_reader_web/models/segmentation_model.pt'
            }
            
            # Check if analog models exist
            self.analog_available = ANALOG_PIPELINE_AVAILABLE and all(
                os.path.exists(path) for path in self.analog_models_config.values()
            )
            
            if not self.analog_available:
                logger.warning("Analog gauge models not found. Only digital gauges will be supported.")
            
            print("EnhancedGaugeDetector initialized successfully")
            print(f"Digital gauges: Available")
            print(f"Analog gauges: {'Available' if self.analog_available else 'Not Available'}")
            
        except Exception as e:
            print(f"Error initializing EnhancedGaugeDetector: {e}")
            self.digital_model = None
            self.analog_available = False

    def detect_gauge_reading(self, frame, camera_id):
        """Detect gauge reading from frame - handles both digital and analog"""
        with open('debug_log.txt', 'a') as f:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] detect_gauge_reading called for camera {camera_id}\n")

        if frame is None:
            return

        try:
            # Initialize lock for camera if not exists
            if camera_id not in self.camera_locks:
                self.camera_locks[camera_id] = threading.Lock()

            with self.camera_locks[camera_id]:
                # Get camera type from camera_id (you'll need to pass this info)
                camera_type = self.get_camera_type(camera_id)
                
                if camera_type == 'digital':
                    self._detect_digital_gauge(frame, camera_id)
                elif camera_type == 'analog' and self.analog_available:
                    self._detect_analog_gauge(frame, camera_id)
                else:
                    logger.warning(f"Unsupported camera type '{camera_type}' for camera {camera_id}")

        except Exception as e:
            logger.error(f"Error detecting gauge reading for {camera_id}: {e}")

    def _detect_digital_gauge(self, frame, camera_id):
        """Handle digital gauge detection (existing logic)"""
        if not self.digital_model:
            return
            
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        if processed_frame is None:
            return

        # Crop center for better detection
        cropped_frame, crop_coords = self.crop_center(processed_frame)
        if cropped_frame is None:
            return

        # Run detection
        results = self.digital_model.predict(cropped_frame, conf=0.15, iou=0.4, verbose=False)
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
                    'last_update': time.time(),
                    'type': 'digital'
                })
                
                # Store detections for visualization
                self.last_detections[camera_id] = {
                    'detections': detections,
                    'number': number,
                    'frame_shape': frame.shape,
                    'type': 'digital'
                }

    def _detect_analog_gauge(self, frame, camera_id):
        """Handle analog gauge detection using the pipeline"""
        try:
            # Create temporary file for the frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, frame)
            
            # Create temporary directory for results
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Process the image using the analog pipeline
                    result = process_image(
                        temp_path,
                        self.analog_models_config['detection_model'],
                        self.analog_models_config['key_point_model'], 
                        self.analog_models_config['segmentation_model'],
                        temp_dir,
                        debug=False,
                        eval_mode=False,
                        image_is_raw=False
                    )
                    
                    if result and 'value' in result:
                        reading = result['value']
                        unit = result.get('unit', '')
                        
                        # Store result
                        if camera_id not in self.camera_readings:
                            self.camera_readings[camera_id] = {}
                            
                        self.camera_readings[camera_id].update({
                            'reading': float(reading) if reading is not None else 0,
                            'confidence': 90,  # Analog pipeline confidence
                            'last_update': time.time(),
                            'type': 'analog',
                            'unit': unit
                        })
                        
                        # Store basic detection info for visualization
                        self.last_detections[camera_id] = {
                            'reading': reading,
                            'unit': unit,
                            'frame_shape': frame.shape,
                            'type': 'analog'
                        }
                        
                        logger.info(f"Analog gauge reading for camera {camera_id}: {reading} {unit}")
                    
                except Exception as e:
                    logger.error(f"Analog pipeline processing failed for camera {camera_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in analog gauge detection for camera {camera_id}: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    def get_camera_type(self, camera_id):
        """
        Get camera type from camera_id
        """
        if camera_id in self.camera_readings and 'camera_type' in self.camera_readings[camera_id]:
            return self.camera_readings[camera_id]['camera_type']
        return 'digital'  # Default to digital

    def set_camera_type(self, camera_id, camera_type):
        """Set camera type for a specific camera"""
        if camera_id not in self.camera_readings:
            self.camera_readings[camera_id] = {}
        self.camera_readings[camera_id]['camera_type'] = camera_type

    def get_detection_overlay(self, frame, camera_id):
        """Get frame with detection overlay for both digital and analog"""
        with open('debug_log.txt', 'a') as f:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] get_detection_overlay called for camera {camera_id}\n")

        if frame is None:
            return frame

        if camera_id in self.last_detections:
            detection_data = self.last_detections[camera_id]
            detection_type = detection_data.get('type', 'digital')
            
            if detection_type == 'digital':
                return self.draw_detections(frame.copy(), 
                                          detection_data.get('detections', []), 
                                          detection_data.get('number', 'N/A'))
            elif detection_type == 'analog':
                return self.draw_analog_result(frame.copy(), detection_data)
        
        return frame

    def draw_analog_result(self, frame, detection_data):
        """Draw analog gauge result on frame"""
        if frame is None:
            return frame
            
        reading = detection_data.get('reading', 'N/A')
        unit = detection_data.get('unit', '')
        
        # Display result with background
        result_text = f"Reading: {reading} {unit}"
        text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(frame, (5, 20), (text_size[0] + 15, 55), (0, 0, 0), -1)
        cv2.putText(frame, result_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Add analog gauge indicator
        cv2.putText(frame, "ANALOG GAUGE", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return frame

    # Include all the existing methods from your original GaugeDetector class
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

    # ... (include all other existing methods from the original GaugeDetector)
    
    def process_detections(self, results, crop_coords=None):
        """Process YOLO detection results for digital gauges"""
        if not self.digital_model:
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
                cls_name = self.digital_model.names[cls_id]

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

            # Apply filters (include your existing filter methods)
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

    # Include other existing methods...
    def filter_detections_by_size(self, detections):
        """Filter detections based on reasonable size"""
        if not detections:
            return detections

        number_detections = [det for det in detections if det['class'] != '.']
        if not number_detections:
            return detections

        avg_width = np.mean([det['width'] for det in number_detections])
        avg_height = np.mean([det['height'] for det in number_detections])

        filtered_detections = []
        for det in detections:
            if det['class'] == '.':
                if (det['width'] > avg_width * 0.1 and det['width'] < avg_width * 0.8 and
                        det['height'] > avg_height * 0.1 and det['height'] < avg_height * 0.6):
                    filtered_detections.append(det)
            else:
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

        dots.sort(key=lambda x: (-x['confidence'], x['center_x']))

        filtered_dots = []
        for dot in dots:
            too_close = False
            for existing_dot in filtered_dots:
                distance = abs(dot['center_x'] - existing_dot['center_x'])
                if distance < 40:
                    too_close = True
                    break

            if not too_close:
                filtered_dots.append(dot)
                break

        return numbers + filtered_dots

    def validate_digit_sequence(self, detections):
        """Validate if digit sequence is logical"""
        if not detections:
            return detections

        detections.sort(key=lambda x: x['center_x'])
        classes = [det['class'] for det in detections]

        dot_count = classes.count('.')
        if dot_count > 1:
            dots = [det for det in detections if det['class'] == '.']
            best_dot = max(dots, key=lambda x: x['confidence'])
            detections = [det for det in detections if det['class'] != '.' or det == best_dot]

        if detections and detections[0]['class'] == '.':
            detections = detections[1:]
        if detections and detections[-1]['class'] == '.':
            detections = detections[:-1]

        return detections

    def get_stable_result(self, current_result, camera_id):
        """Get stable result using voting"""
        if current_result is None:
            return None

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

        recent_results = list(buffer)[-5:]

        from collections import Counter
        counter = Counter(recent_results)
        most_common, count = counter.most_common(1)[0]

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

            if det['class'] == '.':
                color = (0, 255, 255)
            elif det['class'] == '1':
                color = (255, 0, 255)
            elif det['class'] in ['7']:
                color = (255, 165, 0)
            else:
                color = (0, 255, 0)

            thickness = 3 if det['class'] in ['1', '.'] else 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            text = f"{det['class']} ({det['confidence']:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (int(x1), int(y1 - 20)), (int(x1) + text_size[0], int(y1)), color, -1)
            cv2.putText(frame, text, (int(x1), int(y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        result_text = f"Number: {number}"
        text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.rectangle(frame, (5, 20), (text_size[0] + 15, 55), (0, 0, 0), -1)
        cv2.putText(frame, result_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        return frame

    def get_camera_reading(self, camera_id):
        """Get reading for specific camera"""
        if camera_id in self.camera_readings:
            return self.camera_readings[camera_id]
        return {'reading': 0, 'confidence': 0, 'last_update': 0}

    def get_all_readings(self):
        """Get all camera readings"""
        return {cam_id: self.get_camera_reading(cam_id) for cam_id in self.camera_readings.keys()}

    def clear_camera_data(self, camera_id):
        """Clear data for specific camera"""
        if camera_id in self.camera_readings:
            del self.camera_readings[camera_id]
        if camera_id in self.camera_locks:
            del self.camera_locks[camera_id]
        if camera_id in self.last_detections:
            del self.last_detections[camera_id]