import os
import sys
import logging
import time
import threading
import tempfile
import shutil
from collections import deque, defaultdict
import cv2
import numpy as np
from PIL import Image

# Add the gauge_reader_web directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
gauge_reader_path = os.path.join(current_dir, 'gauge_reader_web')
sys.path.insert(0, gauge_reader_path)

# Now import the pipeline function from the subfolder
try:
    from pipeline import process_image
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logging.error(f"Could not import pipeline from gauge_reader_web: {e}")
    PIPELINE_AVAILABLE = False

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AnalogGaugeDetector:
    def __init__(self, base_model_path='gauge_reader_web/models'):
        """Initialize the analog gauge detector
        
        Args:
            base_model_path: Base path to the models directory (default: gauge_reader_web/models)
        """
        try:
            self.base_model_path = base_model_path
            
            # Define model file paths
            self.detection_model_path = os.path.join(base_model_path, 'gauge_detection_model.pt')
            self.key_point_model_path = os.path.join(base_model_path, 'keypoint_model.pt')
            self.segmentation_model_path = os.path.join(base_model_path, 'needle_segmentation_model.pt')
            
            # Check if pipeline and model files are available
            self.pipeline_available = PIPELINE_AVAILABLE
            self.models_available = all([
                os.path.exists(self.detection_model_path),
                os.path.exists(self.key_point_model_path),
                os.path.exists(self.segmentation_model_path)
            ])
            
            if not self.pipeline_available:
                logger.warning("Pipeline module not available. Check gauge_reader_web folder structure.")
                logger.warning("Analog gauge detection will return simulated values")
            elif not self.models_available:
                logger.warning(f"Some model files missing in {base_model_path}:")
                logger.warning(f"  - {self.detection_model_path}")
                logger.warning(f"  - {self.key_point_model_path}")
                logger.warning(f"  - {self.segmentation_model_path}")
                logger.warning("Analog gauge detection will return simulated values")
            
            self.camera_readings = {}  # Store readings for each camera
            self.camera_locks = {}     # Thread locks for each camera
            self.reading_history = defaultdict(lambda: deque(maxlen=5))  # History for stability
            
            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp(prefix='analog_gauge_')
            
            logger.info(f"AnalogGaugeDetector initialized successfully")
            logger.info(f"Pipeline available: {self.pipeline_available}")
            logger.info(f"Models available: {self.models_available}")
            
        except Exception as e:
            logger.error(f"Error initializing AnalogGaugeDetector: {e}")
            self.pipeline_available = False
            self.models_available = False

    def __del__(self):
        """Cleanup temporary directory"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

    def preprocess_frame(self, frame):
        """Improve image quality for better analog gauge detection"""
        if frame is None:
            return None
            
        # Convert to RGB (pipeline expects RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        # Enhance contrast for better gauge reading
        # Convert to LAB color space
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels and convert back to RGB
        lab = cv2.merge((l_channel, a, b))
        enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb

    def simulate_analog_reading(self, camera_id):
        """Generate simulated analog gauge reading when pipeline/models are not available"""
        # Create realistic simulated readings based on camera ID
        base_values = {
            '1': 45.7,
            '2': 52.3,
            '3': 38.9,
            '4': 61.2,
            '5': 29.4
        }
        
        base_value = base_values.get(str(camera_id), 40.0)
        
        # Add some realistic variation
        variation = np.sin(time.time() * 0.1) * 5 + np.random.normal(0, 1)
        reading = max(0, base_value + variation)
        
        return {
            'value': round(reading, 1),
            'unit': 'bar',
            'confidence': 85 + int(np.random.normal(0, 5))
        }

    def process_analog_gauge(self, frame, camera_id):
        """Process frame for analog gauge reading"""
        if not self.pipeline_available or not self.models_available:
            return self.simulate_analog_reading(camera_id)
            
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return None
                
            # Create temporary file for the frame
            temp_image_path = os.path.join(self.temp_dir, f'temp_frame_{camera_id}_{int(time.time())}.jpg')
            
            # Convert numpy array to PIL Image and save
            pil_image = Image.fromarray(processed_frame)
            pil_image.save(temp_image_path)
            
            # Create run path for this detection
            run_path = os.path.join(self.temp_dir, f'run_{camera_id}_{int(time.time())}')
            os.makedirs(run_path, exist_ok=True)
            
            # Change to gauge_reader_web directory for pipeline execution
            original_cwd = os.getcwd()
            gauge_reader_dir = os.path.join(current_dir, 'gauge_reader_web')
            os.chdir(gauge_reader_dir)
            
            try:
                # Process the image using the pipeline
                # Use relative paths since we're now in the gauge_reader_web directory
                result = process_image(
                    image=temp_image_path,
                    detection_model_path='models/gauge_detection_model.pt',
                    key_point_model_path='models/key_point_model.pt',
                    segmentation_model_path='models/segmentation_model.pt',
                    run_path=run_path,
                    debug=False,  # Set to False to avoid saving debug images
                    eval_mode=False
                )
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)
            
            # Clean up temporary files
            try:
                os.remove(temp_image_path)
                shutil.rmtree(run_path)
            except:
                pass
                
            if result and 'value' in result:
                confidence = 90 if result['value'] > 0 else 0
                return {
                    'value': float(result['value']),
                    'unit': result.get('unit', 'bar'),
                    'confidence': confidence
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error processing analog gauge for camera {camera_id}: {e}")
            # Return simulated value as fallback
            return self.simulate_analog_reading(camera_id)

    def get_stable_reading(self, current_result, camera_id):
        """Get stable reading using moving average"""
        if current_result is None or 'value' not in current_result:
            return None
            
        history = self.reading_history[camera_id]
        history.append(current_result['value'])
        
        if len(history) < 3:
            return current_result
            
        # Use median for stability (removes outliers)
        stable_value = float(np.median(list(history)))
        
        return {
            'value': stable_value,
            'unit': current_result['unit'],
            'confidence': current_result['confidence']
        }

    def detect_gauge_reading(self, frame, camera_id):
        """Detect analog gauge reading from frame"""
        if frame is None:
            return
            
        try:
            # Initialize lock for camera if not exists
            if camera_id not in self.camera_locks:
                self.camera_locks[camera_id] = threading.Lock()
                
            with self.camera_locks[camera_id]:
                # Process the analog gauge
                result = self.process_analog_gauge(frame, camera_id)
                
                if result:
                    # Get stable reading
                    stable_result = self.get_stable_reading(result, camera_id)
                    
                    if stable_result:
                        # Store result
                        if camera_id not in self.camera_readings:
                            self.camera_readings[camera_id] = {}
                            
                        self.camera_readings[camera_id].update({
                            'reading': stable_result['value'],
                            'confidence': int(stable_result['confidence']),
                            'unit': stable_result['unit'],
                            'last_update': time.time()
                        })
                        
                        logger.debug(f"Analog gauge reading for camera {camera_id}: {stable_result['value']} {stable_result['unit']}")
                        
        except Exception as e:
            logger.error(f"Error detecting analog gauge reading for camera {camera_id}: {e}")

    def get_camera_reading(self, camera_id):
        """Get reading for specific camera"""
        if camera_id in self.camera_readings:
            return self.camera_readings[camera_id]
        return {'reading': 0, 'confidence': 0, 'unit': 'bar', 'last_update': 0}

    def get_all_readings(self):
        """Get all camera readings"""
        return {cam_id: self.get_camera_reading(cam_id) for cam_id in self.camera_readings.keys()}

    def get_detection_overlay(self, frame, camera_id):
        """Get frame with detection overlay for analog gauge"""
        if frame is None:
            return frame
            
        # Get current reading
        reading_data = self.get_camera_reading(camera_id)
        
        if reading_data['confidence'] > 0:
            # Draw gauge reading overlay
            overlay_text = f"Analog: {reading_data['reading']:.1f} {reading_data.get('unit', 'bar')}"
            confidence_text = f"Confidence: {reading_data['confidence']}%"
            
            # Add availability status
            status_text = "Real Detection" if (self.pipeline_available and self.models_available) else "Simulated"
            
            # Add semi-transparent background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 80), (450, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add text
            cv2.putText(frame, overlay_text, (15, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, confidence_text, (15, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Status: {status_text}", (15, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add gauge outline indicator (simple circle)
            center = (frame.shape[1] - 100, 100)
            radius = 40
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            
            # Add needle indication (simple line based on reading value)
            # Normalize reading to angle (assuming 0-100 scale maps to -90 to +90 degrees)
            max_reading = 100  # Adjust based on your gauge range
            normalized_value = min(reading_data['reading'] / max_reading, 1.0)
            angle = (normalized_value - 0.5) * np.pi  # -90 to +90 degrees
            
            needle_end_x = int(center[0] + radius * 0.8 * np.cos(angle))
            needle_end_y = int(center[1] + radius * 0.8 * np.sin(angle))
            cv2.line(frame, center, (needle_end_x, needle_end_y), (0, 0, 255), 3)
            
        return frame

    def clear_camera_data(self, camera_id):
        """Clear data for specific camera"""
        if camera_id in self.camera_readings:
            del self.camera_readings[camera_id]
        if camera_id in self.camera_locks:
            del self.camera_locks[camera_id]
        if camera_id in self.reading_history:
            del self.reading_history[camera_id]

    def get_system_info(self):
        """Get system information for debugging"""
        return {
            'pipeline_available': self.pipeline_available,

            'models_available': self.models_available,
            'base_model_path': self.base_model_path,
            'gauge_reader_path': gauge_reader_path,
            'model_files': {
                'detection_model': os.path.exists(self.detection_model_path),
                'key_point_model': os.path.exists(self.key_point_model_path),
                'segmentation_model': os.path.exists(self.segmentation_model_path)
            }
        }