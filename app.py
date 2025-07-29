from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import time
import json
import requests
import logging
import threading

# Import your enhanced detector
try:
    from enhanced_gauge_detector import EnhancedGaugeDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Could not import enhanced detector: {e}")
    DETECTOR_AVAILABLE = False
    EnhancedGaugeDetector = None

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize enhanced gauge detector
gauge_detector = None
if DETECTOR_AVAILABLE:
    try:
        gauge_detector = EnhancedGaugeDetector('best.pt')  # Just pass your digital model path
        logger.info("Enhanced gauge detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize enhanced gauge detector: {e}")
        gauge_detector = None
else:
    logger.error("Enhanced gauge detector not available")

# Global variables for camera processing
camera_processors = {}
frame_count = 0
camera_configs = {}

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
            if reading_data.get('confidence', 0) > 50:
                readings[cam_id] = {
                    "reading": reading_data.get('reading', 0),
                    "confidence": reading_data.get('confidence', 0),
                    "fps": 30,
                    "processing": 85,
                    "type": config.get('type', 'video'),
                    "unit": reading_data.get('unit', '')
                }
                active_cameras += 1
                # For pressure readings, check if it's a digital or analog gauge
                if config.get('type') in ['digital', 'analog']:
                    total_pressure += reading_data.get('reading', 0)
                    pressure_count += 1
            else:
                readings[cam_id] = {
                    "reading": 0, 
                    "confidence": 0, 
                    "fps": 0, 
                    "processing": 0,
                    "type": config.get('type', 'video'),
                    "unit": ""
                }
        else:
            readings[cam_id] = {
                "reading": 0, 
                "confidence": 0, 
                "fps": 0, 
                "processing": 0,
                "type": config.get('type', 'video'),
                "unit": ""
            }
    
    avg_pressure = (total_pressure / pressure_count) if pressure_count > 0 else 0
    
    return {
        **readings,
        "system_status": "OPERATIONAL" if active_cameras > 0 else "STANDBY",
        "active_cameras": active_cameras,
        "h2_detection": "NORMAL",
        "average_pressure": avg_pressure
    }

def process_frame_for_detection(frame, camera_id):
    """Process frame for gauge detection - SIMPLIFIED ONE-LINE CALL!"""
    if gauge_detector and frame is not None:
        try:
            global frame_count
            frame_count += 1
            
            cam_str = str(camera_id)
            
            # Process every 5th frame for better performance
            if frame_count % 5 == 0:
                # THIS IS THE SIMPLE ONE-LINE CALL!
                # For analog gauges: This runs the full pipeline (detection→keypoints→ellipse→OCR→segmentation→reading)
                # For digital gauges: This runs YOLO detection with number recognition
                threading.Thread(
                    target=gauge_detector.detect_gauge_reading,  # This handles everything internally
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
        return

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
        return
    
    # Set camera properties
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
            
            # SIMPLE DETECTION CALL - all the complex pipeline stuff happens inside detect_gauge_reading()
            if frame_counter % 1 == 0 and gauge_detector:
                process_frame_for_detection(frame.copy(), camera_id)
                # Get overlay with detections (shows bounding boxes for digital, result overlay for analog)
                frame = gauge_detector.get_detection_overlay(frame, camera_id)
            
            # Add camera info overlay
            camera_type = camera_config.get('type', 'video').upper()
            cv2.putText(frame, f'{camera_id}: {address} ({camera_type})', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add current reading if available
            if gauge_detector:
                reading_data = gauge_detector.get_camera_reading(camera_id)
                if reading_data.get('confidence', 0) > 0:
                    unit = reading_data.get('unit', '')
                    reading_value = reading_data.get('reading', 0)
                    confidence = reading_data.get('confidence', 0)
                    reading_text = f"Reading: {reading_value:.2f}{unit} ({confidence}%)"
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
        
        # Validate camera type
        valid_types = ['digital', 'analog', 'video']
        if camera_type not in valid_types:
            return jsonify({'success': False, 'message': f'Invalid camera type. Must be one of: {valid_types}'})
        
        # Create camera config if it doesn't exist
        if camera_id not in camera_configs:
            camera_configs[camera_id] = {'address': '', 'active': False, 'type': 'video'}
        
        camera_configs[camera_id]['address'] = address
        camera_configs[camera_id]['active'] = bool(address)
        camera_configs[camera_id]['type'] = camera_type
        
        # Clear previous detection data for this camera
        if gauge_detector:
            gauge_detector.clear_camera_data(camera_id)
            # Set the camera type in the detector
            gauge_detector.set_camera_type(camera_id, camera_type)
        
        logger.info(f"Camera {camera_id} configured: {address} ({camera_type})")
        return jsonify({'success': True, 'message': f'Camera {camera_id} configured as {camera_type}'})
    except Exception as e:
        logger.error(f"Error setting camera address: {e}")
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
        logger.error(f"Error deleting camera: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/camera_config/<camera_id>')
def get_camera_config(camera_id):
    """Get camera configuration"""
    try:
        if camera_id in camera_configs:
            return jsonify(camera_configs[camera_id])
        else:
            return jsonify({'address': '', 'active': False, 'type': 'video'})
    except Exception as e:
        logger.error(f"Error getting camera config: {e}")
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return render_template('index.html', data=get_scada_data())
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return f"Error loading page: {e}", 500

@app.route('/external_video_feed')
def external_video_feed():
    """Video feed endpoint for external access"""
    # Default to camera 1 if no specific camera requested
    return Response(gen_frames_with_detection('1'), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/readings')
def get_readings():
    """API endpoint to get current readings"""
    try:
        return jsonify(get_scada_data())
    except Exception as e:
        logger.error(f"Error getting readings: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/gauge_status')
def gauge_status():
    """Get detailed gauge detection status"""
    try:
        if gauge_detector:
            readings = gauge_detector.get_all_readings()
            return jsonify({
                'detector_available': True,
                'readings': readings,
                'timestamp': time.time(),
                'analog_support': getattr(gauge_detector, 'analog_available', False),
                'digital_support': gauge_detector.digital_model is not None
            })
        else:
            return jsonify({
                'detector_available': False,
                'error': 'Gauge detector not initialized',
                'timestamp': time.time(),
                'analog_support': False,
                'digital_support': False
            })
    except Exception as e:
        logger.error(f"Error getting gauge status: {e}")
        return jsonify({
            'detector_available': False,
            'error': str(e),
            'timestamp': time.time(),
            'analog_support': False,
            'digital_support': False
        })

@app.route('/api/detector_capabilities')
def detector_capabilities():
    """Get detector capabilities"""
    try:
        if gauge_detector:
            analog_available = getattr(gauge_detector, 'analog_available', False)
            digital_available = gauge_detector.digital_model is not None
            
            supported_types = ['video']  # Always support video
            if digital_available:
                supported_types.append('digital')
            if analog_available:
                supported_types.append('analog')
                
            return jsonify({
                'digital_gauges': digital_available,
                'analog_gauges': analog_available,
                'supported_types': supported_types,
                'detector_status': 'Available'
            })
        else:
            return jsonify({
                'digital_gauges': False,
                'analog_gauges': False,
                'supported_types': ['video'],
                'detector_status': 'Not Available'
            })
    except Exception as e:
        logger.error(f"Error getting detector capabilities: {e}")
        return jsonify({
            'digital_gauges': False,
            'analog_gauges': False,
            'supported_types': [],
            'detector_status': f'Error: {str(e)}'
        })

@app.route('/api/system_info')
def system_info():
    """Get system information"""
    try:
        info = {
            'timestamp': time.time(),
            'total_cameras': len(camera_configs),
            'active_cameras': len([c for c in camera_configs.values() if c.get('active', False)]),
            'detector_available': gauge_detector is not None,
            'version': '2.0.0'
        }
        
        if gauge_detector:
            info.update({
                'digital_support': gauge_detector.digital_model is not None,
                'analog_support': getattr(gauge_detector, 'analog_available', False)
            })
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    print("🚀 Starting H2 Factory Camera Monitoring System...")
    print("=" * 50)
    print("📊 Gauge detector status:")
    if gauge_detector:
        digital_status = "✅ Available" if gauge_detector.digital_model else "❌ Not available"
        analog_status = "✅ Available" if getattr(gauge_detector, 'analog_available', False) else "❌ Not available"
        print(f"  - Digital gauges: {digital_status}")
        print(f"  - Analog gauges: {analog_status}")
        print("  - System ready for camera configuration")
    else:
        print("  - ❌ Detector not initialized")
        print("  - System running in video-only mode")
    
    print("=" * 50)
    print("🌐 Starting Flask server on http://0.0.0.0:5000")
    print("📱 Access the web interface to add cameras")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)