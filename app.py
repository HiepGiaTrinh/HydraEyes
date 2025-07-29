from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import time
import json
import requests
import logging
import threading
from digital_detector import GaugeDetector  # Your existing digital detector
from analog_detector import AnalogGaugeDetector  # New analog detector

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize gauge detectors
digital_detector = None
analog_detector = None

# Initialize digital detector
try:
    digital_detector = GaugeDetector('best.pt')  # Your existing digital model
    logger.info("Digital gauge detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize digital gauge detector: {e}")
    digital_detector = None

# Initialize analog detector 
try:
    analog_detector = AnalogGaugeDetector('gauge_reader_web/models')  # Points to subfolder
    logger.info("Analog gauge detector initialized successfully")
    
    # Log system info for debugging
    system_info = analog_detector.get_system_info()
    logger.info(f"Analog detector system info: {system_info}")
    
except Exception as e:
    logger.error(f"Failed to initialize analog gauge detector: {e}")
    analog_detector = None

# Global variables for camera processing
camera_processors = {}
frame_count = 0

camera_configs = {
    # Camera configurations will be stored here
    # Format: camera_id: {'address': 'url', 'active': bool, 'type': 'digital/analog/video'}
}

def get_detector_for_camera(camera_id):
    """Get the appropriate detector based on camera type"""
    camera_config = camera_configs.get(camera_id, {})
    camera_type = camera_config.get('type', 'video')
    
    if camera_type == 'digital':
        return digital_detector
    elif camera_type == 'analog':
        return analog_detector
    elif camera_type == 'video':
        return digital_detector  # Default to digital for video feeds
    else:
        return digital_detector  # Fallback

def get_scada_data():
    """Get SCADA data with real gauge readings if available"""
    readings = {}
    active_cameras = 0
    total_pressure = 0
    pressure_count = 0
    
    # Get all configured cameras dynamically
    for cam_id, config in camera_configs.items():
        if config.get('active', False):
            # Choose detector based on camera type
            camera_type = config.get('type', 'digital')
            if camera_type == 'analog' and analog_detector:
                detector = analog_detector
            else:
                detector = digital_detector  # Use digital for 'digital' and 'video' types
            
            if detector:
                reading_data = detector.get_camera_reading(cam_id)
                if reading_data.get('confidence', 0) > 50:
                    readings[cam_id] = {
                        "reading": reading_data.get('reading', 0),
                        "confidence": reading_data.get('confidence', 0),
                        "fps": 30,
                        "processing": 85
                    }
                    active_cameras += 1
                    # For pressure readings, check if it's a digital or analog gauge
                    if camera_type in ['digital', 'analog']:
                        total_pressure += reading_data.get('reading', 0)
                        pressure_count += 1
                else:
                    readings[cam_id] = {"reading": 0, "confidence": 0, "fps": 0, "processing": 0}
            else:
                readings[cam_id] = {"reading": 0, "confidence": 0, "fps": 0, "processing": 0}
        else:
            readings[cam_id] = {"reading": 0, "confidence": 0, "fps": 0, "processing": 0}
    
    avg_pressure = (total_pressure / pressure_count) if pressure_count > 0 else 0
    
    return {
        **readings,
        "system_status": "OPERATIONAL" if active_cameras > 0 else "STANDBY",
        "active_cameras": active_cameras,
        "h2_detection": "NORMAL",
        "average_pressure": avg_pressure
    }

def process_frame_for_detection(frame, camera_id):
    """Process frame for gauge detection in a separate thread"""
    if frame is not None:
        try:
            # Run detection every few frames to avoid overload
            global frame_count
            frame_count += 1
            
            # Convert camera_id to string to ensure consistency
            cam_str = str(camera_id)
            
            # Get camera type to choose detector
            camera_config = camera_configs.get(cam_str, {})
            camera_type = camera_config.get('type', 'digital')
            
            if camera_type == 'analog' and analog_detector:
                # Process every 15th frame for analog (more CPU intensive)
                if frame_count % 15 == 0:
                    threading.Thread(
                        target=analog_detector.detect_gauge_reading, 
                        args=(frame, cam_str),
                        daemon=True
                    ).start()
                    logger.debug(f"Started analog detection thread for camera {cam_str}")
            elif digital_detector:
                # Process every 5th frame for digital (your original logic)
                if frame_count % 5 == 0:
                    threading.Thread(
                        target=digital_detector.detect_gauge_reading, 
                        args=(frame, cam_str),
                        daemon=True
                    ).start()
                    logger.debug(f"Started digital detection thread for camera {cam_str}")
            
        except Exception as e:
            logger.error(f"Error processing frame for {camera_id}: {e}")

def gen_frames_with_detection(camera_id):
    """Generate video frames with gauge detection for specific camera"""
    camera_config = camera_configs.get(camera_id, {})
    
    if not camera_config.get('active') or not camera_config.get('address'):
        # Return placeholder frame if camera not configured
        while True:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            camera_type = camera_config.get('type', 'video')
            cv2.putText(placeholder, f'{camera_id}: Not Configured ({camera_type.upper()})', (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, 'Configure camera address to start streaming', (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        return

    # Try to connect to camera
    address = camera_config['address']
    camera_type = camera_config.get('type', 'video')
    
    try:
        if address.startswith('http'):
            cap = cv2.VideoCapture(address)
        else:
            cap = cv2.VideoCapture(int(address))  # For device index like 0, 1
    except Exception as e:
        logger.error(f"Error opening camera {camera_id}: {e}")
        cap = cv2.VideoCapture(0)  # Fallback to default camera
    
    if not cap.isOpened():
        # If camera can't open, show error frame
        while True:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f'{camera_id}: Connection Failed ({camera_type.upper()})', (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, f'Cannot connect to: {address}', (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_counter = 0
    
    while True:
        try:
            success, frame = cap.read()
            if not success:
                logger.error(f"Failed to read frame from {camera_id}")
                # Show connection lost frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f'{camera_id}: Connection Lost ({camera_type.upper()})', (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
                
            frame_counter += 1
            
            process_frame_for_detection(frame.copy(), camera_id)

            # Get overlay with detections based on camera type
            camera_type = camera_config.get('type', 'digital')
            if camera_type == 'analog' and analog_detector:
                frame = analog_detector.get_detection_overlay(frame, camera_id)
            elif digital_detector:
                frame = digital_detector.get_detection_overlay(frame, camera_id)
            
            # Add camera info overlay
            cv2.putText(frame, f'{camera_id}: {address} ({camera_type.upper()})', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add current reading if available

            detector = get_detector_for_camera(camera_id)
            if detector:
                reading_data = detector.get_camera_reading(camera_id)
                if reading_data.get('confidence', 0) > 0:
                    unit = reading_data.get('unit', '')
                    reading_text = f"Reading: {reading_data.get('reading', 0):.2f} {unit} ({reading_data.get('confidence', 0)}%)"
                    cv2.putText(frame, reading_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Error processing frame for {camera_id}: {e}")
            break
            
    cap.release()

@app.route('/camera_feed/<camera_id>')
def camera_feed(camera_id):
    """Video feed endpoint for specific camera"""
    return Response(gen_frames_with_detection(camera_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_camera_address', methods=['POST'])
def set_camera_address():
    """Set camera address and type"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        address = data.get('address', '').strip()
        camera_type = data.get('type', 'video')  # digital, analog, or video
        
        # Create camera config if it doesn't exist
        if camera_id not in camera_configs:
            camera_configs[camera_id] = {'address': '', 'active': False, 'type': 'video'}
        
        camera_configs[camera_id]['address'] = address
        camera_configs[camera_id]['active'] = bool(address)
        camera_configs[camera_id]['type'] = camera_type
        
        # Clear previous detection data for this camera from all detectors
        if digital_detector:
            digital_detector.clear_camera_data(camera_id)
        if analog_detector:
            analog_detector.clear_camera_data(camera_id)
        
        logger.info(f"Camera {camera_id} configured: {address} ({camera_type})")
        return jsonify({'success': True, 'message': f'Camera {camera_id} configured as {camera_type}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_camera', methods=['POST'])
def delete_camera():
    """Delete camera configuration"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        
        if camera_id in camera_configs:
            camera_configs[camera_id]['address'] = ''
            camera_configs[camera_id]['active'] = False
            
            # Clear detection data from all detectors
            if digital_detector:
                digital_detector.clear_camera_data(camera_id)
            if analog_detector:
                analog_detector.clear_camera_data(camera_id)
            
            logger.info(f"Camera {camera_id} deleted")
            return jsonify({'success': True, 'message': f'Camera {camera_id} deleted'})
        else:
            return jsonify({'success': False, 'message': 'Invalid camera ID'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/camera_config/<camera_id>')
def get_camera_config(camera_id):
    """Get camera configuration"""
    if camera_id in camera_configs:
        return jsonify(camera_configs[camera_id])
    else:
        return jsonify({'address': '', 'active': False, 'type': 'video'})

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', data=get_scada_data())

@app.route('/api/readings')
def get_readings():
    """API endpoint to get current readings"""
    return jsonify(get_scada_data())

@app.route('/api/gauge_status')
def gauge_status():
    """Get detailed gauge detection status"""
    digital_available = digital_detector is not None
    analog_available = analog_detector is not None
    
    all_readings = {}
    
    # Collect readings from both detectors
    if digital_detector:
        digital_readings = digital_detector.get_all_readings()
        for cam_id, reading in digital_readings.items():
            if camera_configs.get(cam_id, {}).get('type') == 'digital':
                all_readings[cam_id] = {**reading, 'detector_type': 'digital'}
    
    if analog_detector:
        analog_readings = analog_detector.get_all_readings()
        for cam_id, reading in analog_readings.items():
            if camera_configs.get(cam_id, {}).get('type') == 'analog':
                all_readings[cam_id] = {**reading, 'detector_type': 'analog'}
    
    return jsonify({
        'digital_detector_available': digital_available,
        'analog_detector_available': analog_available,
        'readings': all_readings,
        'timestamp': time.time()
    })

@app.route('/api/camera_reading/<camera_id>')
def get_camera_reading(camera_id):
    """Get real-time reading for specific camera"""
    # Choose detector based on camera type
    camera_config = camera_configs.get(camera_id, {})
    camera_type = camera_config.get('type', 'digital')
    
    if camera_type == 'analog' and analog_detector:
        detector = analog_detector
    else:
        detector = digital_detector  # Use digital for 'digital' and 'video' types
    
    if detector:
        reading_data = detector.get_camera_reading(camera_id)
        return jsonify({
            'camera_id': camera_id,
            'reading': reading_data.get('reading', 0),
            'confidence': reading_data.get('confidence', 0),
            'last_update': reading_data.get('last_update', 0),
            'timestamp': time.time(),
            'status': 'active' if reading_data.get('confidence', 0) > 30 else 'inactive'
        })
    else:
        return jsonify({
            'camera_id': camera_id,
            'reading': 0,
            'confidence': 0,
            'last_update': 0,
            'timestamp': time.time(),
            'status': 'detector_unavailable'
        })

@app.route('/api/all_camera_readings')
def get_all_camera_readings():
    """Get real-time readings for all cameras"""
    result = {}
    
    for cam_id, config in camera_configs.items():
        # Choose detector based on camera type
        camera_type = config.get('type', 'digital')
        if camera_type == 'analog' and analog_detector:
            detector = analog_detector
        else:
            detector = digital_detector
        
        if detector:
            reading_data = detector.get_camera_reading(cam_id)
            result[cam_id] = {
                'camera_id': cam_id,
                'reading': reading_data.get('reading', 0),
                'confidence': reading_data.get('confidence', 0),
                'last_update': reading_data.get('last_update', 0),
                'status': 'active' if reading_data.get('confidence', 0) > 30 else 'inactive'
            }
        else:
            result[cam_id] = {
                'camera_id': cam_id,
                'reading': 0,
                'confidence': 0,
                'last_update': 0,
                'status': 'detector_unavailable'
            }
    
    return jsonify({
        'readings': result,
        'timestamp': time.time(),
        'detector_available': True
    })


@app.route('/api/detector_status')
def detector_status():
    """Get status of both detectors"""
    analog_info = {}
    if analog_detector:
        analog_info = analog_detector.get_system_info()
    
    return jsonify({
        'digital_detector': {
            'available': digital_detector is not None,
            'model_file': 'best.pt',
            'type': 'Digital Gauge Detection (YOLO)'
        },
        'analog_detector': {
            'available': analog_detector is not None,
            'pipeline_available': analog_info.get('pipeline_available', False),
            'models_available': analog_info.get('models_available', False),
            'type': 'Analog Gauge Detection (Pipeline)',
            'model_path': 'gauge_reader_web/models/',
            'model_files': analog_info.get('model_files', {}),
            'system_info': analog_info
        }
    })

@app.route('/api/system_info')
def system_info():
    """Get detailed system information for debugging"""
    info = {
        'detectors': {
            'digital': digital_detector is not None,
            'analog': analog_detector is not None
        },
        'cameras': camera_configs,
        'current_time': time.time()
    }
    
    if analog_detector:
        info['analog_system'] = analog_detector.get_system_info()
    
    return jsonify(info)

if __name__ == "__main__":
    print("Starting H2 Factory Camera Monitoring System...")
    print("Digital detector status:", "Available" if digital_detector else "Not available")
    print("Analog detector status:", "Available" if analog_detector else "Not available")
    
    if analog_detector:
        system_info = analog_detector.get_system_info()
        print(f"Analog detector pipeline status: {'Available' if system_info['pipeline_available'] else 'Not available'}")
        print(f"Analog detector models status: {'Available' if system_info['models_available'] else 'Missing - using simulated values'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)