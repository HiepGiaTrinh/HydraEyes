from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import time
import json
import requests
import logging
import threading
from gauge_detector import GaugeDetector

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize gauge detector
gauge_detector = None
try:
    gauge_detector = GaugeDetector('best.pt')  # Make sure best.pt is in the same directory
    logger.info("Gauge detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize gauge detector: {e}")
    gauge_detector = None

# Global variables for camera processing
camera_processors = {}
frame_count = 0

camera_configs = {
}

def get_scada_data():
    """Get SCADA data with real gauge readings if available"""
    readings = {}
    active_cameras = 0
    total_pressure = 0
    pressure_count = 0
    
    # Get all configured cameras dynamically
    for cam_id, config in camera_configs.items():
        if gauge_detector and config.get('active', False):
            reading_data = gauge_detector.get_camera_reading(cam_id)
            if reading_data['confidence'] > 50:
                readings[cam_id] = {
                    "reading": reading_data['reading'],
                    "confidence": reading_data['confidence'],
                    "fps": 30,
                    "processing": 85
                }
                active_cameras += 1
                # For pressure readings, check if it's a digital or analog gauge
                if config.get('type') in ['digital', 'analog']:
                    total_pressure += reading_data['reading']
                    pressure_count += 1
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
    if gauge_detector and frame is not None:
        try:
            # Run detection every few frames to avoid overload
            global frame_count
            frame_count += 1
            
            # Convert camera_id to string to ensure consistency
            cam_str = str(camera_id)
            
            # Process every 5th frame for better performance
            # You can adjust this interval based on your needs
            if frame_count % 5 == 0:
                threading.Thread(
                    target=gauge_detector.detect_gauge_reading, 
                    args=(frame, cam_str),
                    daemon=True
                ).start()
                logger.debug(f"Started detection thread for camera {cam_str}")
            
        except Exception as e:
            logger.error(f"Error processing frame for {camera_id}: {e}")


def gen_frames_with_detection(camera_id):
    """Generate video frames with gauge detection for specific camera"""
    camera_config = camera_configs.get(camera_id, {})
    
    if not camera_config.get('active') or not camera_config.get('address'):
        # Return placeholder frame if camera not configured
        while True:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f'{camera_id}: Not Configured', (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, 'Configure camera address to start streaming', (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        return  # Add return statement here

    # Try to connect to camera
    address = camera_config['address']
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
            cv2.putText(error_frame, f'{camera_id}: Connection Failed', (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, f'Cannot connect to: {address}', (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        return  # Add return statement here
    
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
                cv2.putText(error_frame, f'{camera_id}: Connection Lost', (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
                
            frame_counter += 1
            
            # Process for gauge detection every few frames
            if frame_counter % 1 == 0 and gauge_detector:
                process_frame_for_detection(frame.copy(), camera_id)
                # logger.info("Condition is working normal")
                # Get overlay with detections
                frame = gauge_detector.get_detection_overlay(frame, camera_id)
            
            # Add camera info overlay
            cv2.putText(frame, f'{camera_id}: {address}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add current reading if available
            if gauge_detector:
                reading_data = gauge_detector.get_camera_reading(camera_id)
                if reading_data['confidence'] > 0:
                    reading_text = f"Reading: {reading_data['reading']:.2f} ({reading_data['confidence']}%)"
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
        camera_type = data.get('type', 'video')  # Default to video
        
        # Create camera config if it doesn't exist
        if camera_id not in camera_configs:
            camera_configs[camera_id] = {'address': '', 'active': False, 'type': 'video'}
        
        camera_configs[camera_id]['address'] = address
        camera_configs[camera_id]['active'] = bool(address)
        camera_configs[camera_id]['type'] = camera_type
        
        # Clear previous detection data for this camera
        if gauge_detector:
            gauge_detector.clear_camera_data(camera_id)
        
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
            
            # Clear detection data
            if gauge_detector:
                gauge_detector.clear_camera_data(camera_id)
            
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
        return jsonify({'address': '', 'active': False})

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', data=get_scada_data())

@app.route('/external_video_feed')
def external_video_feed():
    """Video feed endpoint"""
    return Response(gen_frames_with_detection(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/readings')
def get_readings():
    """API endpoint to get current readings"""
    return jsonify(get_scada_data())

@app.route('/api/gauge_status')
def gauge_status():
    """Get detailed gauge detection status"""
    if gauge_detector:
        readings = gauge_detector.get_all_readings()
        return jsonify({
            'detector_available': True,
            'readings': readings,
            'timestamp': time.time()
        })
    else:
        return jsonify({
            'detector_available': False,
            'error': 'Gauge detector not initialized',
            'timestamp': time.time()
        })
@app.route('/api/camera_reading/<camera_id>')
def get_camera_reading(camera_id):
    """Get real-time reading for specific camera"""
    if gauge_detector:
        reading_data = gauge_detector.get_camera_reading(camera_id)
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
    if gauge_detector:
        all_readings = gauge_detector.get_all_readings()
        result = {}
        for cam_id, reading_data in all_readings.items():
            result[cam_id] = {
                'camera_id': cam_id,
                'reading': reading_data.get('reading', 0),
                'confidence': reading_data.get('confidence', 0),
                'last_update': reading_data.get('last_update', 0),
                'status': 'active' if reading_data.get('confidence', 0) > 30 else 'inactive'
            }
        
        return jsonify({
            'readings': result,
            'timestamp': time.time(),
            'detector_available': True
        })
    else:
        return jsonify({
            'readings': {},
            'timestamp': time.time(),
            'detector_available': False
        })
    
if __name__ == "__main__":
    print("Starting H2 Factory Camera Monitoring System...")
    print("Gauge detector status:", "Available" if gauge_detector else "Not available")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)