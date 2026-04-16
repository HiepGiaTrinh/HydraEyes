#!/usr/bin/env python3
"""
Webcam Pipeline - Universal Analog Gauge Reader
Outputs: Angle (degrees) + Percentage (0-100%) - Works with any analog gauge
Features: Temporal filtering, ellipse fitting, needle detection
"""
import json
import cv2
import numpy as np
import time
import os
import sys
import argparse
import logging
from datetime import datetime
import torch
from gauge_reader_web.angle_reading_fit.angle_converter import AngleConverter
from gauge_reader_web.angle_reading_fit.line_fit import line_fit, line_fit_ransac
from gauge_reader_web.geometry.ellipse import get_point_from_angle, get_theta_middle

# Tắt logging của ultralytics
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Thêm path để import các module của pipeline
sys.path.append('gauge_reader_web')

import numpy as np  # Make sure numpy is available for class methods
class NumberLabel:
    """Class to store OCR number with ellipse angle"""
    def __init__(self, number, position, theta):
        self.number = number
        self.position = position  # (x, y) position
        self.theta = theta  # angle on ellipse


class GaugeDataLogger:
    """Helper class to log data for chart viewer"""

    def __init__(self, output_file="gauge_readings.json"):
        self.output_file = output_file

    def log_reading(self, reading, angle_degrees, timestamp=None):
        """Log a single reading to file for chart viewer"""
        if timestamp is None:
            timestamp = time.time()

        data = {
            'reading': float(reading),
            'angle_degrees': float(angle_degrees),
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'status': 'active'
        }

        try:
            # Append mode instead of overwrite
            with open(self.output_file, 'a') as f:  # ← 'a' = append
                json.dump(data, f)
                f.write('\n')  # New line for each entry
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            return False

class FullGaugeProcessor:
    """
    Full pipeline processor với gauge detection, keypoint detection và needle segmentation
    """

    def __init__(self, detection_model_path, key_point_model_path, segmentation_model_path):
        self.detection_model_path = detection_model_path
        self.key_point_model_path = key_point_model_path
        self.segmentation_model_path = segmentation_model_path

        self.frame_skip = 2  # Skip frames cho gauge detection
        self.keypoint_skip = 8  # Skip nhiều hơn cho keypoint (heavy)
        # Optimize for real-time performance
        self.detection_skip = 2  # More frequent gauge detection
        self.needle_skip = 2  # Separate needle skip
        self.ellipse_skip = 6  # Less frequent ellipse updates
        self.frame_counter = 0
        self._cached_ellipse = None
        self._tensor_device_cache = {}

        # Cache results
        self.last_gauge_box = None
        self.last_keypoints = None
        self.last_needle_line = None

        # Temporal filtering for stability
        self.reading_history = []
        self.angle_history = []
        self.ellipse_history = []
        self.history_size = 1  # Keep last N readings

        # Stability tracking
        self.stable_gauge_count = 0
        self.stable_threshold = 3

        # Feature toggles
        self.needle_enabled = True

        # NEW: One-time calibration flags
        self.one_time_calibration_mode = False  # ← FORCE DISABLE
        self.ocr_enabled = False  # ← FORCE DISABLE OCR
        self.calibration_completed = False  # Chưa calibrate
        self.calibration_attempts = 0  # Số lần thử
        self.max_calibration_attempts = 5  # Tối đa 5 lần thử
        self.calibration_success_threshold = 3  # Cần ít nhất 3 markers để accept

        # Cache permanent data
        self.permanent_scale_mapping = None
        self.permanent_ellipse_params = None
        self.permanent_unit = None

        self.detected_unit = None

        # NEW: Manual calibration variables
        self.manual_calibration_mode = True  # ← Mặc định BẬT
        # MANUAL CALIBRATION CONFIG
        self.required_calibration_points = 4  # Default, có thể thay đổi
        self.calibration_disabled_calculation = True  # Block tính toán cho đến khi xong
        self.manual_points = []
        self.current_calibration_step = 0
        self.waiting_for_click = False
        self.calibration_instruction = "Click on first number on gauge"
        self.manual_calibrated = False
        self.reading_change_threshold = 0.03
        self.last_logged_reading = None
        # Missing attributes for interpolation
        self.current_scale_mapping = None
        self.last_interpolation_result = None
        self.last_interpolation_angle = None
        self.interpolation_cache_threshold = 2.0  # degrees

        # Data logging for chart
        self.data_logger = GaugeDataLogger("gauge_readings.json")
        self.last_log_time = 0
        self.log_interval = 0.5  # Log every 1 second

        # Chart settings
        self.chart_enabled = True
        self.chart_buffer = []
        self.chart_time_buffer = []
        self.chart_max_points = 100  # Keep last 100 readings
        self.chart_width = 300
        self.chart_height = 150
        self.chart_position = (10, 200)  # Top-left position in frame



        # Load keypoint model
        self.key_point_inferencer = None
        try:
            from gauge_reader_web.key_point_detection.key_point_inference import KeyPointInference
            self.key_point_inferencer = KeyPointInference(key_point_model_path)
            print("✅ Keypoint model loaded")
        except Exception as e:
            print(f"❌ Keypoint model failed: {e}")
            print("📍 Running without keypoint detection - needle tracking only")
            self.keypoint_skip = 99999  # Effectively disable keypoint detection

        print("✅ FullGaugeProcessor initialized")
        print("🎯 Manual Calibration Mode ACTIVE")
        print("   📍 Detect stable gauge first, then click on numbers")

        # NEW: Try to load previous calibration
        if self.manual_calibration_mode:
            print("🎯 Manual mode - OCR disabled")
        else:
            print("🔄 One-time calibration mode enabled")
            if self._load_calibration_from_file():
                print("📂 Using previous calibration - skipping OCR")
            else:
                print("📂 No previous calibration - will run OCR once")

    def save_current_frame(self, frame, frame_id):
        """Save current frame as PNG file for pipeline processing"""
        try:
            # Create filename
            temp_filename = f"temp_frame_{frame_id}.png"

            # Convert BGR to RGB for saving (camera gives BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save as PNG
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            pil_image.save(temp_filename)

            print(f"💾 Saved frame as {temp_filename}")
            return temp_filename

        except Exception as e:
            print(f"Frame save error: {e}")
            return None

    def _tensor_to_numpy(self, tensor):
        """Safely convert tensor to numpy, handling CUDA tensors"""
        if torch.is_tensor(tensor):
            if tensor.is_cuda:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.detach().numpy()
        return tensor

    def _fast_tensor_convert(self, tensor):
        """Fast tensor conversion with caching"""
        if not torch.is_tensor(tensor):
            return tensor

        # Cache tensor device info to avoid repeated checks
        if not hasattr(self, '_tensor_device_cache'):
            self._tensor_device_cache = {}

        tensor_id = id(tensor)
        if tensor_id not in self._tensor_device_cache:
            self._tensor_device_cache[tensor_id] = tensor.is_cuda

        if self._tensor_device_cache[tensor_id]:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()

    def _extract_box_coordinates(self, box):
        """Safely extract box coordinates from tensor or numpy"""
        if box is None:
            return None

        # Convert tensor to numpy if needed
        if torch.is_tensor(box):
            if box.is_cuda:
                box_np = box.detach().cpu().numpy()
            else:
                box_np = box.detach().numpy()
        else:
            box_np = np.array(box)

        # Ensure we have 4 coordinates
        if len(box_np.shape) > 1:
            box_np = box_np.flatten()

        if len(box_np) >= 4:
            return box_np[:4].astype(int).tolist()
        return None

    def process_frame(self, frame):
        """
        Xử lý frame với stable caching
        """
        self.frame_counter += 1

        results = {
            'processed_frame': frame.copy(),
            'gauge_detected': False,
            'gauge_box': None,
            'keypoints': None,
            'needle_line': None,
            'reading': None
        }

        # Step 1: Gauge Detection
        if self.frame_counter % self.detection_skip == 0:
            results = self._detect_gauge(frame, results)
        else:
            # Always reuse cached gauge for stability
            if self.last_gauge_box is not None:
                results['gauge_detected'] = True
                results['gauge_box'] = self.last_gauge_box

        # Early exit if no stable gauge
        if not results['gauge_detected'] or self.stable_gauge_count < self.stable_threshold:
            self._draw_results(results)
            return results

        # Step 2: Keypoint Detection
        if (self.frame_counter % self.keypoint_skip == 0 and
                self.key_point_inferencer is not None):
            results = self._detect_keypoints(frame, results)
        else:
            # Always reuse cached keypoints for stability
            if self.last_keypoints is not None:
                results['keypoints'] = self.last_keypoints

        # Step 3: Needle Detection
        if (self.frame_counter % self.needle_skip == 0 and
                self.needle_enabled):
            results = self._detect_needle(frame, results)
        else:
            # Always reuse cached needle for stability
            if self.last_needle_line is not None and self.needle_enabled:
                results['needle_line'] = self.last_needle_line

        # Step 4: Manual Calibration
        if (self.manual_calibration_mode and not self.manual_calibrated and
                not self.waiting_for_click and len(self.manual_points) == 0):
            self.waiting_for_click = True

        # Always draw all available results
        self._draw_results(results)
        return results

    def _detect_needle(self, frame, results):
        """Needle segmentation step"""
        try:
            gauge_box = results['gauge_box']

            # Crop và resize gauge region
            cropped_img = self._crop_image(frame, gauge_box)
            cropped_resized_img = cv2.resize(cropped_img, (448, 448), interpolation=cv2.INTER_CUBIC)

            # Needle segmentation
            from gauge_reader_web.segmentation.segmenation_inference import segment_gauge_needle
            from gauge_reader_web.segmentation.segmenation_inference import get_fitted_line, get_start_end_line

            needle_mask_x, needle_mask_y = segment_gauge_needle(
                cropped_resized_img, self.segmentation_model_path
            )

            if len(needle_mask_x) > 0 and len(needle_mask_y) > 0:
                # Fit line through needle pixels
                needle_line_coeffs, needle_error = get_fitted_line(needle_mask_x, needle_mask_y)
                needle_line_start_x, needle_line_end_x = get_start_end_line(needle_mask_x)
                needle_line_start_y, needle_line_end_y = get_start_end_line(needle_mask_y)

                needle_line = {
                    'coeffs': needle_line_coeffs,
                    'start': (needle_line_start_x, needle_line_start_y),
                    'end': (needle_line_end_x, needle_line_end_y),
                    'error': needle_error,
                    'mask_x': needle_mask_x,
                    'mask_y': needle_mask_y
                }

                results['needle_line'] = needle_line
                self.last_needle_line = needle_line
                # SỬA THÀNH (chỉ log mỗi 3 giây):
                if self.frame_counter % 150 == 0:
                    print(f"✅ Needle detected (error: {needle_error:.3f})")

                # Calculate reading if we have both keypoints and needle
                if results.get('keypoints') is not None:
                    results = self._calculate_reading(results)


        except Exception as e:
            print(f"Needle segmentation error: {e}")

        return results

    def _calculate_reading(self, results):
        """Calculate final gauge reading from keypoints and needle with temporal filtering"""
        try:
            # CRITICAL: Block calculation until manual calibration is complete
            if self.manual_calibration_mode and not self.manual_calibrated:
                # Chưa calibrate xong - không tính reading
                return results

            if self.calibration_disabled_calculation and not self.calibration_completed:
                # Calibration chưa hoàn tất - không tính reading
                return results

            keypoints = results['keypoints']
            needle_line = results['needle_line']

            if len(keypoints) >= 3 and needle_line is not None:
                # Step 1: Use LOCKED ellipse if available, otherwise stable ellipse
                if hasattr(self, 'permanent_ellipse_params') and self.permanent_ellipse_params is not None:
                    ellipse_params = self.permanent_ellipse_params
                else:
                    ellipse_params = self._get_stable_ellipse(keypoints)
                if ellipse_params is None:
                    return results

                # Step 2: Find needle-ellipse intersection
                needle_coeffs = needle_line['coeffs']
                needle_start_x, needle_end_x = needle_line['start'][0], needle_line['end'][0]

                intersection_point = self._find_needle_ellipse_intersection(
                    needle_coeffs, [needle_start_x, needle_end_x], ellipse_params
                )

                if intersection_point is None:
                    return results

                # Step 3: Calculate angle
                angle = self._calculate_needle_angle(intersection_point, ellipse_params)

                # Step 4: Convert angle to actual reading using calibration
                raw_reading = self._angle_to_actual_reading(angle, results.get('calibration_status'), ellipse_params)

                # Step 5: Apply temporal filtering
                filtered_reading = self._apply_temporal_filter(raw_reading, angle)

                # DÒNG MỚI (format like pipeline.py):
                if filtered_reading is not None:
                    # Check if we have unit from OCR
                    unit = getattr(self, 'detected_unit', None) or 'units'
                    results['reading'] = f"{filtered_reading:.2f}"  # Remove % sign
                    results['unit'] = unit
                    results['raw_reading'] = raw_reading
                    results['angle'] = angle
                    results['intersection_point'] = intersection_point
                    results['ellipse_params'] = ellipse_params

                    # Chỉ log khi reading thay đổi đáng kể
                    unit = getattr(self, 'detected_unit', None) or 'units'
                    should_log = False

                    if self.last_logged_reading is None:
                        # Lần đầu tiên
                        should_log = True
                    elif abs(filtered_reading - self.last_logged_reading) >= self.reading_change_threshold:
                        # Reading thay đổi đáng kể
                        should_log = True

                    if should_log:
                        print(
                            f"📊 READING: {filtered_reading:.2f} {unit} | Angle: {np.degrees(angle):.1f}° | Raw: {raw_reading:.2f}")
                        self.last_logged_reading = filtered_reading
                        # Log data for chart viewer
                        # Log data for chart viewer - ALWAYS LOG
                        current_time = time.time()
                        if current_time - self.last_log_time >= 0.5:  # Log every 0.5 seconds
                            angle_degrees = np.degrees(angle) if angle is not None else 0
                            success = self.data_logger.log_reading(filtered_reading, angle_degrees, current_time)
                            if success:
                                print(
                                    f"✍️ Written: {filtered_reading:.2f} units at {datetime.now().strftime('%H:%M:%S')}")
                            if success and self.frame_counter % 60 == 0:  # Print status every 2 seconds
                                print(f"📊 Data logged to gauge_readings.json")
                            self.last_log_time = current_time

        except Exception as e:
            print(f"Reading calculation error: {e}")

        return results

    def _angle_to_actual_reading(self, angle, calibration_status=None, ellipse_params=None):
        """Convert angle using PERMANENT cached calibration"""
        try:
            # ===== THAY ĐỔI LOGIC =====
            # OLD: Check current_scale_mapping
            # NEW: Always use permanent cache first

            if ((self.calibration_completed or self.manual_calibrated) and
                    self.permanent_scale_mapping is not None):

                # print(f"⚡ Using permanent calibration ({len(self.permanent_scale_mapping)} markers)")
                return self._interpolate_from_permanent_calibration(angle)

            # Fallback to old logic if not calibrated yet
            elif (self.current_scale_mapping is not None and
                  calibration_status in ['active', 'calibrated']):

                print(f"🔄 Using temporary calibration (calibration in progress)")
                print(f"🔄 Using temporary calibration (calibration in progress)")
                return self._interpolate_from_scale_mapping(angle, ellipse_params)

            # Ultimate fallback
            else:
                percentage = self._angle_to_generic_reading(angle)
                print(f"📊 Using generic reading: {percentage:.1f}% (no calibration)")
                return percentage

        except Exception as e:
            print(f"Reading conversion error: {e}")
            return self._angle_to_generic_reading(angle)

    def _interpolate_from_scale_mapping(self, angle, ellipse_params=None):
        """Interpolate reading from calibrated scale mapping using proper angle fitting"""
        try:
            # DÒNG MỚI (chỉ log khi cần):
            if (self.last_interpolation_result is not None and
                    self.last_interpolation_angle is not None):
                angle_diff = abs(np.degrees(angle - self.last_interpolation_angle))
                if angle_diff < self.interpolation_cache_threshold:
                    return self.last_interpolation_result

            if not self.current_scale_mapping or len(self.current_scale_mapping) < 2:
                return self._angle_to_generic_reading(angle)

            if ellipse_params is None:
                # Use last stored ellipse if available
                if hasattr(self, 'last_ellipse_params') and self.last_ellipse_params is not None:
                    ellipse_params = self.last_ellipse_params
                    print("🔧 Using stored ellipse params for interpolation")
                else:
                    print("❌ No ellipse params for proper interpolation, falling back to generic")
                    return self._angle_to_generic_reading(angle)

            # Convert OCR markers to NumberLabel objects with ellipse angles
            number_labels = []
            for marker in self.current_scale_mapping:
                # Project marker position to ellipse to get angle
                marker_pos = np.array(marker['position'])

                # Project to ellipse and get angle
                from gauge_reader_web.geometry.ellipse import get_polar_angle, project_point
                try:
                    # Project marker center to ellipse
                    proj_point = project_point(marker_pos, ellipse_params)
                    marker_theta = get_polar_angle(proj_point, ellipse_params)

                    number_label = NumberLabel(
                        number=marker['value'],
                        position=marker['position'],
                        theta=marker_theta
                    )
                    number_labels.append(number_label)

                except Exception as e:
                    print(f"❌ Failed to project marker {marker['value']}: {e}")
                    continue

            if len(number_labels) < 2:
                print(f"❌ Not enough projected markers: {len(number_labels)}")
                return self._angle_to_generic_reading(angle)

            # Calculate proper zero point from start/end keypoints like pipeline.py
            theta_zero = 0  # Default
            if hasattr(self, 'last_keypoints') and self.last_keypoints is not None:
                try:
                    # Extract start and end points from keypoints (first and last groups)
                    keypoints = self.last_keypoints
                    if len(keypoints) >= 2:
                        start_points = keypoints[0]  # Start notch keypoints
                        end_points = keypoints[-1]  # End notch keypoints

                        # Get representative points (mean of each group)
                        if len(start_points) > 0 and len(end_points) > 0:
                            start_point = np.mean(start_points, axis=0)
                            end_point = np.mean(end_points, axis=0)

                            # Calculate angles for start and end points
                            from gauge_reader_web.geometry.ellipse import get_polar_angle, get_theta_middle
                            theta_start = get_polar_angle(start_point, ellipse_params)
                            theta_end = get_polar_angle(end_point, ellipse_params)

                            # Calculate middle point as zero (wrap-around point)
                            theta_zero = get_theta_middle(theta_start, theta_end)
                            print(
                                f"🎯 Calculated theta_zero: {np.degrees(theta_zero):.1f}° (start: {np.degrees(theta_start):.1f}°, end: {np.degrees(theta_end):.1f}°)")
                        else:
                            theta_zero = np.pi  # Bottom fallback
                            print("⚠️ Using fallback theta_zero: 180°")
                    else:
                        theta_zero = np.pi  # Bottom fallback
                        print("⚠️ Not enough keypoint groups, using fallback theta_zero: 180°")
                except Exception as e:
                    print(f"❌ Zero point calculation error: {e}")
                    theta_zero = np.pi  # Bottom fallback
            else:
                theta_zero = np.pi  # Bottom fallback
                print("⚠️ No keypoints available, using fallback theta_zero: 180°")

            # Convert angles using AngleConverter
            angle_converter = AngleConverter(theta_zero)

            angle_number_list = []
            for number_label in number_labels:
                converted_angle = angle_converter.convert_angle(number_label.theta)
                angle_number_list.append((converted_angle, number_label.number))

            angle_number_arr = np.array(angle_number_list)

            # Use RANSAC to fit line and remove outliers
            try:
                reading_line_coeff, inlier_mask, outlier_mask = line_fit_ransac(
                    angle_number_arr[:, 0], angle_number_arr[:, 1]
                )
                print(f"✅ RANSAC fit: {np.sum(inlier_mask)}/{len(inlier_mask)} inliers")
            except Exception as e:
                print(f"⚠️ RANSAC failed, using simple fit: {e}")
                reading_line_coeff = line_fit(angle_number_arr[:, 0], angle_number_arr[:, 1])

            # Create reading line function
            reading_line = np.poly1d(reading_line_coeff)

            # Convert needle angle and get reading
            needle_angle_conv = angle_converter.convert_angle(angle)
            reading = reading_line(needle_angle_conv)
            # DÒNG MỚI (chỉ log 1 lần khi calibration thành công):
            if self.frame_counter % 900 == 0:  # Only log every 6 seconds
                print(f"🎯 Reading: {reading:.1f} ({len(self.current_scale_mapping)} points)")

            return reading


        except Exception as e:
            print(f"Interpolation error: {e}")
            return self._angle_to_generic_reading(angle)

    def _get_stable_ellipse(self, keypoints):
        """Get stable ellipse with consistent caching"""
        try:
            # Return cached ellipse if available and recent
            if (hasattr(self, '_cached_ellipse') and self._cached_ellipse is not None):
                # Update cache less frequently but keep displaying
                if self.frame_counter % self.ellipse_skip != 0:
                    return self._cached_ellipse

            # Fit new ellipse
            new_ellipse = self._fit_ellipse_from_keypoints(keypoints)
            if new_ellipse is not None:
                # Update cache
                self._cached_ellipse = new_ellipse
                self.ellipse_history.append(new_ellipse)
                if len(self.ellipse_history) > self.history_size:
                    self.ellipse_history.pop(0)

                # Use median for stability
                if len(self.ellipse_history) >= 2:
                    avg_ellipse = np.mean(self.ellipse_history, axis=0)
                    self.last_ellipse_params = avg_ellipse
                    self._cached_ellipse = avg_ellipse
                    return avg_ellipse
                else:
                    self.last_ellipse_params = new_ellipse
                    return new_ellipse

            # Return cached if fitting fails
            return getattr(self, '_cached_ellipse', None)

        except Exception as e:
            print(f"Stable ellipse error: {e}")
            return getattr(self, '_cached_ellipse', None)




    def _angle_to_generic_reading(self, angle):
        """Convert angle to generic percentage reading (0-100%)"""
        try:
            # Generic approach - auto-detect gauge range from needle position
            # Normalize angle to 0-2π range
            normalized_angle = angle % (2 * np.pi)

            # Convert to degrees
            angle_degrees = np.degrees(normalized_angle)

            # Auto-detect approach: map full 360° to 0-100%
            # This is most universal - works for any gauge orientation
            percentage = (angle_degrees / 360) * 100

            # Alternative: If you know your gauge starts from a specific angle,
            # you can uncomment and modify this section:
            # start_angle = 225  # adjust for your gauge
            # gauge_range = 270  # adjust for your gauge
            # adjusted_angle = (angle_degrees - start_angle) % 360
            # percentage = (adjusted_angle / gauge_range) * 100

            # Clamp to 0-100%
            percentage = max(0, min(100, percentage))

            return percentage

        except Exception as e:
            print(f"Generic reading conversion error: {e}")
            return 0

    def _apply_temporal_filter(self, raw_reading, angle):
        """Apply adaptive temporal filtering - less filtering when needle moves fast"""
        try:
            # Add to history
            self.reading_history.append(raw_reading)
            self.angle_history.append(angle)

            # Maintain history size
            if len(self.reading_history) > self.history_size:
                self.reading_history.pop(0)
            if len(self.angle_history) > self.history_size:
                self.angle_history.pop(0)

            # Need at least 2 readings
            if len(self.reading_history) < 2:
                return raw_reading

            # ADAPTIVE FILTERING: Detect needle movement speed
            if len(self.reading_history) >= 3:
                recent_change = abs(self.reading_history[-1] - self.reading_history[-3])

                # If needle moving fast, use less filtering (more responsive)
                if recent_change > 0.3:  # Lower threshold for fast detection
                    filter_window = 1  # No filtering for fast movement
                    weight = 1.0  # Use raw reading only
                elif recent_change > 0.1:  # Medium movement
                    filter_window = 2
                    weight = 0.9  # Still very responsive
                else:  # Slow movement
                    filter_window = min(3, len(self.reading_history))  # Max 3 samples
                    weight = 0.7  # Less filtering than before

                # Apply adaptive median filter
                filtered_reading = np.median(self.reading_history[-filter_window:])

                # Weighted combination with raw reading for responsiveness
                final_reading = weight * raw_reading + (1 - weight) * filtered_reading
            else:
                final_reading = raw_reading

            return final_reading

        except Exception as e:
            return raw_reading

    def _finalize_calibration(self):
        """Cache permanent calibration data và set completed flag"""
        try:
            # Cache scale mapping permanently
            if self.current_scale_mapping is not None:
                self.permanent_scale_mapping = self.current_scale_mapping.copy()
                print(f"💾 Cached {len(self.permanent_scale_mapping)} scale markers permanently")

            # Cache ellipse params permanently - FIX LỖI Ở ĐÂY
            if hasattr(self, 'last_ellipse_params') and self.last_ellipse_params is not None:
                # FIX: Handle both tuple and numpy array
                if isinstance(self.last_ellipse_params, tuple):
                    self.permanent_ellipse_params = np.array(self.last_ellipse_params)  # ← FIX
                else:
                    self.permanent_ellipse_params = self.last_ellipse_params.copy()
                print("💾 Cached ellipse parameters permanently")

            # Cache unit permanently
            if hasattr(self, 'detected_unit') and self.detected_unit is not None:
                self.permanent_unit = self.detected_unit
                print(f"💾 Cached unit '{self.permanent_unit}' permanently")

            # Mark as completed - QUAN TRỌNG!
            self.calibration_completed = True

            # Optional: Save to file for next startup
            self._save_calibration_to_file()

        except Exception as e:
            print(f"❌ Error finalizing calibration: {e}")
            # FIX: Vẫn mark completed nếu có scale mapping
            if self.current_scale_mapping is not None:
                self.calibration_completed = True
                print("⚠️ Partial calibration completed despite error")

    # Also add the helper functions at class level

    def rescale_ellipse_resize(ellipse_params, original_resolution, resized_resolution):
        """Helper function from pipeline.py"""
        x0, y0, ap, bp, phi = ellipse_params
        # move ellipse center
        x0_new, y0_new = move_point_resize((x0, y0), original_resolution, resized_resolution)
        # rescale axis
        scaling_factor = resized_resolution[0] / original_resolution[0]
        ap_new = scaling_factor * ap
        bp_new = scaling_factor * bp
        return x0_new, y0_new, ap_new, bp_new, phi
    def _fit_ellipse_from_keypoints(self, keypoints):
        """Fit ellipse through keypoint coordinates"""
        try:
            from gauge_reader_web.geometry.ellipse import fit_ellipse, cart_to_pol

            # Collect all keypoint coordinates
            all_points = []
            for kp_group in keypoints:
                if hasattr(kp_group, 'shape') and kp_group.shape[0] > 0:
                    for point in kp_group:
                        all_points.append([point[0], point[1]])

            if len(all_points) < 5:  # Need at least 5 points for ellipse
                print(f"❌ Not enough points for ellipse: {len(all_points)}")
                return None

            all_points = np.array(all_points)
            x_coords = all_points[:, 0]
            y_coords = all_points[:, 1]

            # Fit ellipse
            ellipse_coeffs = fit_ellipse(x_coords, y_coords)
            ellipse_params = cart_to_pol(ellipse_coeffs)

            print(f"✅ Ellipse fitted from {len(all_points)} points")
            return ellipse_params

        except Exception as e:
            print(f"Ellipse fitting error: {e}")
            return None

    def _find_needle_ellipse_intersection(self, needle_coeffs, needle_x_range, ellipse_params):
        """Find intersection point between needle line and ellipse"""
        try:
            from gauge_reader_web.geometry.ellipse import get_line_ellipse_point

            intersection_point = get_line_ellipse_point(
                needle_coeffs, needle_x_range, ellipse_params
            )

            return intersection_point

        except Exception as e:
            print(f"Intersection calculation error: {e}")
            return None

    def _calculate_needle_angle(self, intersection_point, ellipse_params):
        """Calculate angle of intersection point on ellipse"""
        try:
            from gauge_reader_web.geometry.ellipse import get_polar_angle

            angle = get_polar_angle(intersection_point, ellipse_params)
            return angle

        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0

    def _detect_gauge(self, frame, results):
        """Gauge detection step with CUDA tensor handling"""
        try:
            # Try to import detection module
            try:
                from gauge_reader_web.gauge_detection.detection_inference import detection_gauge_face
            except Exception as import_error:
                if "Numpy" in str(import_error) or "numpy" in str(import_error):
                    print("⚠️ NumPy compatibility issue - detection disabled")
                    return results
                else:
                    print(f"Import error: {import_error}")
                    return results

            try:
                # Get detection results
                detection_result = detection_gauge_face(frame, self.detection_model_path)

                # Handle different return formats
                if isinstance(detection_result, tuple):
                    box, all_boxes = detection_result
                else:
                    box = detection_result
                    all_boxes = None

                # Extract coordinates safely
                if box is not None:
                    box_coords = self._extract_box_coordinates(box)

                    if box_coords is not None and len(box_coords) == 4:
                        results['gauge_detected'] = True
                        results['gauge_box'] = box_coords

                        # Check stability
                        if self._is_gauge_stable(box_coords):
                            self.stable_gauge_count += 1
                        else:
                            self.stable_gauge_count = 0

                        self.last_gauge_box = box_coords

                    else:
                        self.stable_gauge_count = 0
                        if self.frame_counter % (self.frame_skip * 4) == 0:
                            print("❌ Invalid box coordinates")
                else:
                    self.stable_gauge_count = 0
                    if self.frame_counter % (self.frame_skip * 4) == 0:
                        print("❌ No gauge detected")

            except Exception as e:
                if "Numpy is not available" in str(e):
                    # NumPy compatibility issue - skip this frame
                    pass
                elif "No gauge detected" not in str(e) and "cuda" in str(e).lower():
                    print(f"CUDA conversion error: {e}")
                elif "No gauge detected" not in str(e):
                    print(f"Detection error: {e}")

        except Exception as e:
            print(f"Detection module error: {e}")

        return results
    def _detect_keypoints(self, frame, results):
        """Keypoint detection step"""
        try:
            gauge_box = results['gauge_box']

            # Crop và resize gauge region
            cropped_img = self._crop_image(frame, gauge_box)
            cropped_resized_img = cv2.resize(cropped_img, (448, 448), interpolation=cv2.INTER_CUBIC)

            # Keypoint detection
            from gauge_reader_web.key_point_detection.key_point_inference import detect_key_points

            heatmaps = self.key_point_inferencer.predict_heatmaps(cropped_resized_img)

            # Convert heatmaps to numpy if needed
            if torch.is_tensor(heatmaps):
                if heatmaps.is_cuda:
                    heatmaps = heatmaps.detach().cpu().numpy()
                else:
                    heatmaps = heatmaps.detach().numpy()

            key_point_list = detect_key_points(heatmaps)

            if len(key_point_list) >= 3:
                # Convert keypoints to numpy if needed
                processed_keypoints = []
                for kp in key_point_list:
                    if torch.is_tensor(kp):
                        if kp.is_cuda:
                            kp = kp.detach().cpu().numpy()
                        else:
                            kp = kp.detach().numpy()
                    processed_keypoints.append(kp)

                results['keypoints'] = processed_keypoints
                self.last_keypoints = processed_keypoints
                if self.frame_counter % 1800 == 0:  # Every 12 seconds only
                    print(f"✅ Keypoints: {[len(kp) for kp in processed_keypoints]}")
            else:
                print("❌ Insufficient keypoints")

        except Exception as e:
            print(f"Keypoint error: {e}")

        return results

    def _is_gauge_stable(self, current_box):
        """Check if gauge detection is stable"""
        if self.last_gauge_box is None:
            return False

        try:
            # Ensure both boxes are lists/arrays
            if isinstance(current_box, (list, tuple, np.ndarray)) and \
                    isinstance(self.last_gauge_box, (list, tuple, np.ndarray)):
                # Calculate center distance
                prev_center = [(self.last_gauge_box[0] + self.last_gauge_box[2]) / 2,
                               (self.last_gauge_box[1] + self.last_gauge_box[3]) / 2]
                curr_center = [(current_box[0] + current_box[2]) / 2,
                               (current_box[1] + current_box[3]) / 2]

                distance = np.sqrt((prev_center[0] - curr_center[0]) ** 2 +
                                   (prev_center[1] - curr_center[1]) ** 2)

                return distance < 20  # pixels
        except Exception as e:
            print(f"Stability check error: {e}")
            return False

        return False

    def _crop_image(self, img, box):
        """Crop image using pipeline.py logic"""
        try:
            # Use same logic as pipeline.py crop_image function
            img = np.copy(img)

            # Ensure box coordinates are integers
            if isinstance(box, (list, tuple)):
                box = [int(coord) for coord in box[:4]]

            # Extract coordinates
            x1, y1, x2, y2 = box

            # Crop image: img[y1:y2, x1:x2, :] for RGB
            cropped_img = img[y1:y2, x1:x2, :]

            height = int(y2 - y1)
            width = int(x2 - x1)

            # Make square with padding (same as pipeline.py)
            if height > width:
                delta = height - width
                left, right = delta // 2, delta - (delta // 2)
                top = bottom = 0
            else:
                delta = width - height
                top, bottom = delta // 2, delta - (delta // 2)
                left = right = 0

            pad_color = [0, 0, 0]
            new_img = cv2.copyMakeBorder(
                cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
            )

            return new_img

        except Exception as e:
            print(f"Crop error: {e}")
            return img

    def _draw_results(self, results):
        """Vẽ tất cả results lên frame (fixed flickering)"""
        frame = results['processed_frame']
        # Ensure drawing consistency
        self._ensure_drawing_consistency(results)

        # Draw gauge bounding box (always)
        if results['gauge_detected'] and results['gauge_box'] is not None:
            try:
                box = results['gauge_box']
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1, y1, x2, y2 = [int(coord) for coord in box[:4]]

                    # Color based on stability
                    if self.stable_gauge_count >= self.stable_threshold:
                        color = (0, 255, 0)  # Green - stable
                        status = "STABLE"
                    else:
                        color = (0, 255, 255)  # Yellow - detecting
                        status = "DETECTING"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'GAUGE {status}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print(f"Draw gauge box error: {e}")

        # Draw keypoints (always when available)
        if (results['keypoints'] is not None and
                results['gauge_box'] is not None):
            self._draw_keypoints(frame, results['keypoints'], results['gauge_box'])

        # Draw needle line (always when available)
        if (results.get('needle_line') is not None and
                results['gauge_box'] is not None):
            self._draw_needle_line(frame, results['needle_line'], results['gauge_box'])

        # Draw reading (always)
        if results.get('reading') is not None:
            self._draw_reading(frame, results['reading'])

        # Draw ellipse and intersection (less frequently but stable)
        if (results.get('ellipse_params') is not None and
                results['gauge_box'] is not None):
            self._draw_ellipse(frame, results['ellipse_params'], results['gauge_box'])

        if (results.get('intersection_point') is not None and
                results['gauge_box'] is not None):
            self._draw_intersection(frame, results['intersection_point'], results['gauge_box'])

    def _draw_keypoints(self, frame, key_point_list, gauge_box):
        """Vẽ keypoints lên frame"""
        try:
            if len(key_point_list) >= 3 and isinstance(gauge_box, (list, tuple)) and len(gauge_box) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in gauge_box[:4]]
                gauge_width = x2 - x1
                gauge_height = y2 - y1

                # Scale từ 448x448 về gauge size
                scale_x = gauge_width / 448
                scale_y = gauge_height / 448

                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
                labels = ['Start', 'Middle', 'End']

                for i, points in enumerate(key_point_list):
                    color = colors[i % 3]

                    # Ensure points is numpy array
                    if torch.is_tensor(points):
                        if points.is_cuda:
                            points = points.detach().cpu().numpy()
                        else:
                            points = points.detach().numpy()

                    if hasattr(points, 'shape') and points.shape[0] > 0:
                        for point in points:
                            try:
                                # Transform coordinates
                                x = int(x1 + point[0] * scale_x)
                                y = int(y1 + point[1] * scale_y)
                                cv2.circle(frame, (x, y), 4, color, -1)
                                cv2.putText(frame, labels[i], (x + 5, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            except Exception as e:
                                print(f"Draw keypoint error: {e}")
                                continue
        except Exception as e:
            print(f"Draw keypoints error: {e}")

    def _draw_needle_line(self, frame, needle_line, gauge_box):
        return
        """Vẽ needle line lên frame (fixed flickering)"""
        try:
            if isinstance(gauge_box, (list, tuple)) and len(gauge_box) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in gauge_box[:4]]
                gauge_width = x2 - x1
                gauge_height = y2 - y1

                # Scale từ 448x448 về gauge size
                scale_x = gauge_width / 448
                scale_y = gauge_height / 448

                # Draw needle pixels as dots (simplified for performance)
                if 'mask_x' in needle_line and 'mask_y' in needle_line:
                    mask_x = needle_line['mask_x']
                    mask_y = needle_line['mask_y']

                    # Draw only every 3rd pixel for performance
                    for i in range(0, len(mask_x), 3):
                        px, py = mask_x[i], mask_y[i]
                        x = int(x1 + px * scale_x)
                        y = int(y1 + py * scale_y)
                        cv2.circle(frame, (x, y), 1, (0, 165, 255), -1)  # Orange dots

                # Draw fitted line (always)
                if 'start' in needle_line and 'end' in needle_line:
                    start_x = int(x1 + needle_line['start'][0] * scale_x)
                    start_y = int(y1 + needle_line['start'][1] * scale_y)
                    end_x = int(x1 + needle_line['end'][0] * scale_x)
                    end_y = int(y1 + needle_line['end'][1] * scale_y)

                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 165, 255), 3)
                    cv2.putText(frame, 'NEEDLE', (start_x + 5, start_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        except Exception as e:
            print(f"Draw needle line error: {e}")

    def _draw_reading(self, frame, reading):
        return
        """Enhanced reading display with permanent calibration status"""
        try:
            # Main reading display
            if reading == "READY_TO_CALCULATE":
                text = "READY TO CALCULATE"
                color = (0, 255, 0)
            else:
                # Use permanent unit if available
                unit = self.permanent_unit or getattr(self, 'detected_unit', 'units') or 'units'
                text = f"Reading: {reading} {unit}"
                color = (0, 255, 0)

            cv2.putText(frame, text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ===== THAY ĐỔI STATUS DISPLAY =====
            # Calibration Status
            # Manual Calibration Status
            if self.manual_calibration_mode:
                if self.manual_calibrated:
                    cal_text = f"MANUAL: COMPLETED ({len(self.manual_points)} points)"
                    cal_color = (0, 255, 0)  # Green
                elif self.waiting_for_click:
                    cal_text = f"MANUAL: Click on numbers ({len(self.manual_points)}/{self.required_calibration_points})"
                    cal_color = (0, 255, 255)  # Yellow
                else:
                    cal_text = "MANUAL: Wait for stable gauge..."
                    cal_color = (255, 255, 0)  # Cyan
            else:
                cal_text = "Manual calibration disabled"
                cal_color = (128, 128, 128)  # Gray

            # Only update calibration status occasionally to avoid flickering
            if self.frame_counter % 30 == 0 or not hasattr(self, '_last_cal_text'):
                self._last_cal_text = cal_text
                self._last_cal_color = cal_color

            cv2.putText(frame, self._last_cal_text, (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._last_cal_color, 1)
            # Draw mini chart
            # Draw mini chart - MOVED TO END
            if self.chart_enabled:
                # Always try to draw chart, even with "READY_TO_CALCULATE"
                chart_reading = reading if reading != "READY_TO_CALCULATE" else None
                self._draw_mini_chart(frame, chart_reading)

        except Exception as e:
            print(f"Draw reading error: {e}")

    def _draw_ellipse(self, frame, ellipse_params, gauge_box):
        return
        """Vẽ fitted ellipse lên frame"""
        try:
            from gauge_reader_web.geometry.ellipse import get_ellipse_pts

            if isinstance(gauge_box, (list, tuple)) and len(gauge_box) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in gauge_box[:4]]
                gauge_width = x2 - x1
                gauge_height = y2 - y1

                # Scale từ 448x448 về gauge size
                scale_x = gauge_width / 448
                scale_y = gauge_height / 448

                # Get ellipse points in 448x448 space
                ellipse_x, ellipse_y = get_ellipse_pts(ellipse_params, 100)

                # Transform to gauge coordinates
                points = []
                for ex, ey in zip(ellipse_x, ellipse_y):
                    x = int(x1 + ex * scale_x)
                    y = int(y1 + ey * scale_y)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [points], True, (255, 255, 0), 2)  # Cyan ellipse

        except Exception as e:
            print(f"Draw ellipse error: {e}")

    def _draw_intersection(self, frame, intersection_point, gauge_box):
        return
        """Vẽ needle-ellipse intersection point"""
        try:
            if isinstance(gauge_box, (list, tuple)) and len(gauge_box) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in gauge_box[:4]]
                gauge_width = x2 - x1
                gauge_height = y2 - y1

                # Scale từ 448x448 về gauge size
                scale_x = gauge_width / 448
                scale_y = gauge_height / 448

                # Transform intersection point
                ix = int(x1 + intersection_point[0] * scale_x)
                iy = int(y1 + intersection_point[1] * scale_y)

                cv2.circle(frame, (ix, iy), 8, (255, 0, 255), -1)  # Magenta dot
                cv2.putText(frame, 'NEEDLE TIP', (ix + 10, iy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        except Exception as e:
            print(f"Draw intersection error: {e}")

    def _ensure_drawing_consistency(self, results):
        """Ensure all drawing elements are consistently available"""
        # If we have keypoints, ensure ellipse is available
        if (results.get('keypoints') is not None and
                results.get('ellipse_params') is None and
                hasattr(self, '_cached_ellipse')):
            results['ellipse_params'] = self._cached_ellipse

        # If we have needle and ellipse, ensure intersection is calculated
        if (results.get('needle_line') is not None and
                results.get('ellipse_params') is not None and
                results.get('intersection_point') is None):
            try:
                needle_coeffs = results['needle_line']['coeffs']
                needle_start_x = results['needle_line']['start'][0]
                needle_end_x = results['needle_line']['end'][0]
                intersection = self._find_needle_ellipse_intersection(
                    needle_coeffs, [needle_start_x, needle_end_x],
                    results['ellipse_params']
                )
                if intersection is not None:
                    results['intersection_point'] = intersection
            except Exception as e:
                pass  # Silent fail for drawing consistency


    def _interpolate_from_permanent_calibration(self, angle):
        """Interpolate using permanent cached calibration data"""
        try:
            # Use permanent ellipse params
            if self.permanent_ellipse_params is None:
                print("❌ No permanent ellipse params")
                return self._angle_to_generic_reading(angle)

            # Convert markers to NumberLabel objects
            number_labels = []
            for marker in self.permanent_scale_mapping:
                from gauge_reader_web.geometry.ellipse import get_polar_angle, project_point

                marker_pos = np.array(marker['position'])
                proj_point = project_point(marker_pos, self.permanent_ellipse_params)
                marker_theta = get_polar_angle(proj_point, self.permanent_ellipse_params)

                number_label = NumberLabel(
                    number=marker['value'],
                    position=marker['position'],
                    theta=marker_theta
                )
                number_labels.append(number_label)

            if len(number_labels) < 2:
                print(f"❌ Not enough projected markers: {len(number_labels)}")
                return self._angle_to_generic_reading(angle)

            # Use permanent zero point
            # Use LOCKED permanent zero point
            if hasattr(self, 'permanent_zero_point'):
                theta_zero = self.permanent_zero_point
                # print(f"🔒 Using LOCKED zero point: {np.degrees(theta_zero):.1f}°")
            else:
                theta_zero = self._get_permanent_zero_point()
                # print(f"⚠️ Fallback to dynamic zero point: {np.degrees(theta_zero):.1f}°")

            # Convert angles using AngleConverter
            angle_converter = AngleConverter(theta_zero)

            angle_number_list = []
            for number_label in number_labels:
                converted_angle = angle_converter.convert_angle(number_label.theta)
                angle_number_list.append((converted_angle, number_label.number))

            angle_number_arr = np.array(angle_number_list)

            # RANSAC fit with improved parameters for many points
            try:
                # Adjust RANSAC parameters based on number of points
                if len(angle_number_arr) >= 10:
                    # More iterations for better fitting with many points
                    reading_line_coeff, inlier_mask, outlier_mask = line_fit_ransac(
                        angle_number_arr[:, 0], angle_number_arr[:, 1],
                        max_trials=1000,  # More iterations
                        residual_threshold=0.1  # Tighter threshold
                    )
                    print(f"✅ Enhanced RANSAC fit: {np.sum(inlier_mask)}/{len(inlier_mask)} inliers")
                else:
                    reading_line_coeff, inlier_mask, outlier_mask = line_fit_ransac(
                        angle_number_arr[:, 0], angle_number_arr[:, 1]
                    )

            except Exception as e:
                # print(f"⚠️ RANSAC failed, using simple fit: {e}")
                reading_line_coeff = line_fit(angle_number_arr[:, 0], angle_number_arr[:, 1])

            # Calculate final reading
            reading_line = np.poly1d(reading_line_coeff)
            needle_angle_conv = angle_converter.convert_angle(angle)
            reading = reading_line(needle_angle_conv)

            return reading

        except Exception as e:
            print(f"Permanent interpolation error: {e}")
            return self._angle_to_generic_reading(angle)
    def _get_permanent_zero_point(self):
        """Calculate permanent zero point from keypoints"""
        try:
            if hasattr(self, 'last_keypoints') and self.last_keypoints is not None:
                keypoints = self.last_keypoints
                if len(keypoints) >= 2:
                    start_points = keypoints[0]  # Start notch keypoints
                    end_points = keypoints[-1]  # End notch keypoints

                    if len(start_points) > 0 and len(end_points) > 0:
                        start_point = np.mean(start_points, axis=0)
                        end_point = np.mean(end_points, axis=0)

                        from gauge_reader_web.geometry.ellipse import get_polar_angle, get_theta_middle
                        theta_start = get_polar_angle(start_point, self.permanent_ellipse_params)
                        theta_end = get_polar_angle(end_point, self.permanent_ellipse_params)

                        return get_theta_middle(theta_start, theta_end)

            # Fallback
            return np.pi  # Bottom middle

        except Exception as e:
            print(f"❌ Zero point calculation error: {e}")
            return np.pi

    def _draw_mini_chart(self, frame, current_reading=None):
        """Draw mini realtime chart on frame - FIXED VERSION"""
        if not self.chart_enabled:
            return

        try:
            import time

            # Add current reading to buffer
            if current_reading is not None and current_reading != "READY_TO_CALCULATE":
                try:
                    reading_value = float(current_reading)
                    current_time = time.time()
                    self.chart_buffer.append(reading_value)
                    self.chart_time_buffer.append(current_time)

                    # Keep only last N points
                    if len(self.chart_buffer) > self.chart_max_points:
                        self.chart_buffer.pop(0)
                        self.chart_time_buffer.pop(0)
                except (ValueError, TypeError):
                    return

            if len(self.chart_buffer) < 1:
                return

            # Chart positioning - FIXED
            x, y = 400, 50  # Move to right side, top
            w, h = 350, 180  # Bigger size

            # Ensure chart fits in frame
            frame_height, frame_width = frame.shape[:2]
            if x + w > frame_width or y + h > frame_height:
                x, y = 50, 50  # Fallback position
                w, h = min(300, frame_width - 100), min(150, frame_height - 100)

            print(f"Drawing chart at ({x},{y}) size {w}x{h} with {len(self.chart_buffer)} points")

            # Draw chart background - BRIGHT COLORS for visibility
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 255, 255), -1)  # White background
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Black chart area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green border

            # Title with bright color
            cv2.putText(frame, "REALTIME CHART", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if len(self.chart_buffer) < 2:
                cv2.putText(frame, "Collecting data...", (x + 10, y + h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return

            # Get data range
            readings = np.array(self.chart_buffer)
            y_min, y_max = float(np.min(readings)), float(np.max(readings))

            # Add padding to y range
            y_range = y_max - y_min
            if y_range < 0.1:  # Minimum range
                y_range = 0.1
                y_center = (y_max + y_min) / 2
                y_min = y_center - 0.05
                y_max = y_center + 0.05
            else:
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
            y_range = y_max - y_min

            # Draw grid lines
            for i in range(1, 4):
                grid_y = int(y + (i * h / 4))
                cv2.line(frame, (x, grid_y), (x + w, grid_y), (50, 50, 50), 1)

            # Draw chart line
            points = []
            for i, reading in enumerate(readings):
                chart_x = int(x + (i * w / max(1, len(readings) - 1)))
                chart_y = int(y + h - ((reading - y_min) / y_range * h))
                points.append([chart_x, chart_y])

            # Draw line segments with BRIGHT COLOR
            if len(points) >= 2:
                for i in range(1, len(points)):
                    cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), (0, 255, 255), 3)  # Yellow line

                # Draw current point
                if len(points) > 0:
                    cv2.circle(frame, tuple(points[-1]), 6, (0, 0, 255), -1)  # Red dot

            # Y-axis labels
            cv2.putText(frame, f"{y_max:.1f}", (x - 40, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"{y_min:.1f}", (x - 40, y + h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Current value - BIG and BRIGHT
            if len(readings) > 0:
                current_val = readings[-1]
                cv2.putText(frame, f"VALUE: {current_val:.2f}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            print(f"Chart drawn successfully: {len(points)} points, range {y_min:.2f}-{y_max:.2f}")

        except Exception as e:
            print(f"Chart drawing error: {e}")
            # Draw error indicator
            cv2.rectangle(frame, (400, 50), (500, 100), (0, 0, 255), 2)
            cv2.putText(frame, "CHART ERROR", (405, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    def _save_calibration_to_file(self):
        """Save permanent calibration to file for next startup"""
        try:
            import json
            calibration_data = {
                'scale_mapping': self.permanent_scale_mapping,
                'ellipse_params': self.permanent_ellipse_params.tolist() if self.permanent_ellipse_params is not None else None,
                'unit': self.permanent_unit,
                'timestamp': time.time()
            }

            with open('gauge_calibration.json', 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print("💾 Calibration saved to gauge_calibration.json")

        except Exception as e:
            print(f"Save calibration error: {e}")

    def _load_calibration_from_file(self):
        """Load calibration from previous session"""
        try:
            import json
            with open('gauge_calibration.json', 'r') as f:
                calibration_data = json.load(f)

            self.permanent_scale_mapping = calibration_data['scale_mapping']
            self.permanent_ellipse_params = np.array(calibration_data['ellipse_params']) if calibration_data[
                'ellipse_params'] else None
            self.permanent_unit = calibration_data['unit']
            self.calibration_completed = True

            print(f"📂 Loaded previous calibration: {len(self.permanent_scale_mapping)} markers")
            return True

        except FileNotFoundError:
            print("📂 No previous calibration found")
            return False
        except Exception as e:
            print(f"Load calibration error: {e}")
            return False

    def _handle_mouse_click(self, x, y, force_manual=False):
        """Handle mouse click for manual calibration"""
        if not self.manual_calibration_mode or not self.waiting_for_click:
            return

        # Convert screen coordinates to gauge coordinates
        if self.last_gauge_box is not None:
            gauge_box = self.last_gauge_box
            x1, y1, x2, y2 = gauge_box
            gauge_width = x2 - x1
            gauge_height = y2 - y1

            # Convert to gauge coordinates (0-448 space)
            gauge_x = ((x - x1) / gauge_width) * 448
            gauge_y = ((y - y1) / gauge_height) * 448

            click_type = "RIGHT" if force_manual else "LEFT"
            print(f"🎯 {click_type} Click at screen ({x}, {y}) -> gauge ({gauge_x:.1f}, {gauge_y:.1f})")

            # NEW: Check if click is near any detected keypoint (unless forced manual)
            if not force_manual and self.last_keypoints is not None and len(self.last_keypoints) >= 3:
                keypoint_clicked, keypoint_info = self._find_nearest_keypoint(gauge_x, gauge_y)
                if keypoint_clicked:
                    print(f"🎯 Clicked on {keypoint_info['type']} keypoint at {keypoint_info['position']}")
                    try:
                        value = self._get_value_gui(f"Enter value for {keypoint_info['type']} keypoint:")
                        self.manual_points.append((keypoint_info['position'], value))
                        self.current_calibration_step += 1

                        print(f"✅ Keypoint {self.current_calibration_step}: {keypoint_info['type']} = {value}")

                        # Continue or finish
                        if self.current_calibration_step >= self.required_calibration_points:
                            print("🎯 Enough points collected. Processing calibration...")
                            self.waiting_for_click = False
                            self._complete_manual_calibration()
                        else:
                            self.calibration_instruction = f"Click on next number ({self.current_calibration_step + 1}/10)"
                        return  # Exit early - đã xử lý keypoint click
                    except ValueError:
                        print("❌ Invalid number. Click again.")
                        return

            # Check if click is inside gauge
            if 0 <= gauge_x <= 448 and 0 <= gauge_y <= 448:
                # Ask user for value input
                try:
                    value = float(input(f"Enter value for point {self.current_calibration_step + 1}: "))
                    self.manual_points.append(((gauge_x, gauge_y), value))  # Store gauge coords
                    self.current_calibration_step += 1

                    print(f"✅ Point {self.current_calibration_step}: gauge({gauge_x:.1f}, {gauge_y:.1f}) = {value}")

                    # Continue or finish
                    if self.current_calibration_step >= self.required_calibration_points:  # ← Trigger earlier
                        print("🎯 Enough points collected. Processing calibration...")
                        self.waiting_for_click = False
                        self._complete_manual_calibration()
                    else:
                        self.calibration_instruction = f"Click on next number ({self.current_calibration_step + 1}/10)"

                except ValueError:
                    print("❌ Invalid number. Click again.")
                except KeyboardInterrupt:
                    print("❌ Calibration cancelled")
                    self.waiting_for_click = False
            else:
                print("❌ Click outside gauge area")
        else:
            print("❌ No gauge detected")

        print()  # New line for readability

    def _find_nearest_keypoint(self, click_x, click_y, threshold=30):
        """Find nearest keypoint to click position"""
        try:
            min_distance = float('inf')
            nearest_keypoint = None

            # Check all keypoint groups
            keypoint_types = ['start', 'middle', 'end']
            for i, keypoints in enumerate(self.last_keypoints):
                if hasattr(keypoints, 'shape') and keypoints.shape[0] > 0:
                    for j, point in enumerate(keypoints):
                        distance = np.sqrt((point[0] - click_x) ** 2 + (point[1] - click_y) ** 2)
                        if distance < threshold and distance < min_distance:
                            min_distance = distance
                            nearest_keypoint = {
                                'type': keypoint_types[i],
                                'position': (float(point[0]), float(point[1])),
                                'distance': distance,
                                'group_index': i,
                                'point_index': j
                            }

            if nearest_keypoint:
                return True, nearest_keypoint
            return False, None

        except Exception as e:
            print(f"❌ Keypoint search error: {e}")
            return False, None

    def _get_value_gui(self, prompt):
        """Get value input using GUI popup"""
        import tkinter as tk
        from tkinter import simpledialog

        try:
            # Create hidden root window
            root = tk.Tk()
            root.withdraw()  # Hide main window
            root.attributes('-topmost', True)  # Always on top

            # Get input
            value = simpledialog.askfloat("Manual Calibration", prompt, parent=root)
            root.destroy()

            if value is not None:
                return value
            else:
                raise ValueError("Input cancelled")

        except Exception as e:
            print(f"GUI input error: {e}")
            # Fallback to terminal input
            return float(input(f"{prompt}: "))

    def _complete_manual_calibration(self):
        """Complete manual calibration and convert to permanent data"""
        try:
            if len(self.manual_points) < 2:
                print("❌ Need at least 2 points for calibration")
                return False

            print(f"🎯 Processing {len(self.manual_points)} manual calibration points...")
            print("🔧 DEBUG: Manual points collected:")
            for i, (pos, val) in enumerate(self.manual_points):
                print(f"   Point {i + 1}: {pos} = {val}")

            print(f"🎯 Processing {len(self.manual_points)} manual calibration points...")

            # Convert manual points to scale mapping format
            scale_markers = []
            for i, (position, value) in enumerate(self.manual_points):
                scale_markers.append({
                    'value': float(value),
                    'position': list(position),  # (gauge_x, gauge_y) in 448x448 space
                    'confidence': 1.0  # Manual points have highest confidence
                })
                print(f"  Point {i + 1}: ({position[0]:.1f}, {position[1]:.1f}) = {value}")

            # Store as permanent calibration data
            self.permanent_scale_mapping = scale_markers

            # Get current ellipse params for permanent storage
            # CRITICAL FIX: Fit ellipse từ keypoints hiện tại nếu chưa có
            if not hasattr(self, 'last_ellipse_params') or self.last_ellipse_params is None:
                if hasattr(self, 'last_keypoints') and self.last_keypoints is not None:
                    print("🔧 Fitting ellipse for calibration completion...")
                    fitted_ellipse = self._fit_ellipse_from_keypoints(self.last_keypoints)
                    if fitted_ellipse is not None:
                        self.last_ellipse_params = fitted_ellipse
                        print("✅ Ellipse fitted successfully for calibration")
                    else:
                        print("❌ Failed to fit ellipse for calibration")
                        return False
                else:
                    print("❌ No keypoints available for ellipse fitting")
                    return False

            # Get current ellipse params for permanent storage
            if hasattr(self, 'last_ellipse_params') and self.last_ellipse_params is not None:
                if isinstance(self.last_ellipse_params, tuple):
                    self.permanent_ellipse_params = np.array(self.last_ellipse_params)
                else:
                    self.permanent_ellipse_params = self.last_ellipse_params.copy()
                print("💾 Stored ellipse parameters")
            else:
                print("⚠️ No ellipse parameters available - will use generic reading")

            # Set unit (user can specify or use default)
            self.permanent_unit = "units"  # Default, user can change later

            # CRITICAL: Lock ellipse and zero point permanently
            if hasattr(self, 'last_ellipse_params') and self.last_ellipse_params is not None:
                if isinstance(self.last_ellipse_params, tuple):
                    self.permanent_ellipse_params = np.array(self.last_ellipse_params)
                else:
                    self.permanent_ellipse_params = self.last_ellipse_params.copy()

                # Lock zero point permanently
                self.permanent_zero_point = self._get_permanent_zero_point()
                print(f"🔒 LOCKED permanent ellipse and zero point (theta={np.degrees(self.permanent_zero_point):.1f}°)")
            else:
                print("⚠️ No ellipse parameters available - will use generic reading")
                return False
            # Mark as completed
            self.manual_calibrated = True
            self.calibration_completed = True
            self.waiting_for_click = False
            # Reset reading tracking for new calibration
            self.last_logged_reading = None

            # Calculate range
            values = [m['value'] for m in scale_markers]
            min_val, max_val = min(values), max(values)

            print(f"✅ Manual calibration completed!")
            # Optimize for responsiveness with many calibration points
            if len(scale_markers) >= 15:
                self.history_size = 2  # Very responsive with many points
            elif len(scale_markers) >= 10:
                self.history_size = 3  # Still very responsive
            else:
                self.history_size = 4  # Default for fewer points
                print("🚀 Optimized for balanced accuracy and speed")
            print(f"   📍 Range: {min_val} - {max_val} {self.permanent_unit}")
            print(f"   📊 {len(scale_markers)} calibration points")

            # Save to file for next session
            self._save_calibration_to_file()

            return True

        except Exception as e:
            print(f"❌ Manual calibration completion error: {e}")
            return False

    def set_test_mode(self, enabled=True):
        """Enable/disable test mode for automated testing"""
        self.test_mode = enabled
        if enabled:
            # Optimize for testing
            self.frame_skip = 1
            self.keypoint_skip = 1
            self.history_size = 1
            print("🧪 Test mode enabled")

    def reset_for_testing(self):
        """Reset processor state for testing multiple images"""
        # Clear all state
        self.frame_counter = 0
        self.stable_gauge_count = 0
        self.last_gauge_box = None
        self.last_keypoints = None
        self.last_needle_line = None

        # Clear histories
        self.reading_history = []
        self.angle_history = []
        self.ellipse_history = []

        # Reset calibration but keep manual mode
        self.waiting_for_click = False
        self.manual_points = []
        self.current_calibration_step = 0

        # Clear permanent calibration
        self.permanent_scale_mapping = None
        self.permanent_ellipse_params = None
        self.permanent_unit = None
        self.calibration_completed = False

        print("🔄 Processor reset for testing")

    def move_point_resize(point, original_resolution, resized_resolution):
        """Helper function from pipeline.py"""
        new_point_x = point[0] * resized_resolution[0] / original_resolution[0]
        new_point_y = point[1] * resized_resolution[1] / original_resolution[1]
        return new_point_x, new_point_y

    def rescale_ellipse_resize(ellipse_params, original_resolution, resized_resolution):
        """Helper function from pipeline.py"""
        x0, y0, ap, bp, phi = ellipse_params
        # move ellipse center
        x0_new, y0_new = move_point_resize((x0, y0), original_resolution, resized_resolution)
        # rescale axis
        scaling_factor = resized_resolution[0] / original_resolution[0]
        ap_new = scaling_factor * ap
        bp_new = scaling_factor * bp
        return x0_new, y0_new, ap_new, bp_new, phi

def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1, help="Camera ID")
    parser.add_argument("--detection_model", type=str,
                        default="gauge_reader_web/models/gauge_detection_model.pt",
                        help="Path to detection model")
    parser.add_argument("--key_point_model", type=str,
                        default="gauge_reader_web/models/keypoint_model.pt",
                        help="Path to key point model")
    parser.add_argument("--segmentation_model", type=str,
                        default="gauge_reader_web/models/needle_segmentation_model.pt",
                        help="Path to segmentation model")
    parser.add_argument("--save_frames", action="store_true", help="Save frames")
    parser.add_argument("--frame_skip", type=int, default=5, help="Frame skip for gauge detection")
    parser.add_argument("--keypoint_skip", type=int, default=15, help="Frame skip for keypoint detection")
    args = parser.parse_args()

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available, using CPU")

    # Check models
    required_models = [args.detection_model, args.key_point_model]
    for model_path in required_models:
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

    print("✅ All required models found")
    # Check for existing calibration
    if os.path.exists('gauge_calibration.json'):
        print("📂 Found previous calibration file")
        try:
            use_existing = input("Use existing calibration? (y/n, default y): ").strip().lower()
            if use_existing == '' or use_existing == 'y':
                print("✅ Using existing calibration - skipping manual calibration")
                # Initialize processor with existing calibration
                processor = FullGaugeProcessor(args.detection_model, args.key_point_model, args.segmentation_model)
                processor.frame_skip = args.frame_skip
                processor.keypoint_skip = args.keypoint_skip
                processor.detection_skip = 3
                processor.needle_skip = 5
                processor.ellipse_skip = 10

                # Load existing calibration and disable manual mode
                processor._load_calibration_from_file()
                processor.manual_calibration_mode = False
                processor.manual_calibrated = True
                processor.waiting_for_click = False

                # Skip the manual calibration setup
                print("🚀 Ready to read gauge!")
                manual_calibration_needed = False
            else:
                print("🔄 Will create new calibration")
                manual_calibration_needed = True
        except:
            print("⚠️ Invalid input, using existing calibration")
            manual_calibration_needed = False
    else:
        print("📂 No previous calibration found")
        manual_calibration_needed = True

    if manual_calibration_needed:
        # Ask user for calibration points
        if manual_calibration_needed:
            # Ask user for calibration points
            print("🎯 Manual Calibration Setup:")
            try:
                num_points = int(input("Enter number of calibration points (2-50, default 4): "))
                if num_points < 2 or num_points > 50:
                    print("⚠️ Invalid range, using default 4 points")
                    num_points = 4
            except:
                print("⚠️ Invalid input, using default 4 points")
                num_points = 4

            print(f"✅ Will calibrate with {num_points} points")

    if not manual_calibration_needed:
        # Processor already initialized above with existing calibration
        pass
    else:
        # Initialize processor for new calibration
        processor = FullGaugeProcessor(args.detection_model, args.key_point_model, args.segmentation_model)
        processor.frame_skip = args.frame_skip
        processor.keypoint_skip = args.keypoint_skip
        # Set optimized timing
        processor.detection_skip = 3
        processor.needle_skip = 5
        processor.ellipse_skip = 10
        processor.required_calibration_points = num_points

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Optimize camera buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag

    print("📋 Controls:")
    print("   'q' - Quit")
    print("   's' - Save frame")
    print("   'r' - Reset (clears temporal filtering)")
    print("   '+' - Increase frame skip")
    print("   '-' - Decrease frame skip")
    print("   'k' - Toggle keypoint detection")
    print("   'n' - Toggle needle detection")
    print("🖱️  Mouse Controls (during calibration):")
    print("   LEFT CLICK  - Auto-detect keypoints (snap to detected points)")
    print("   RIGHT CLICK - Manual point (ignore keypoint detection)")

    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    fps = 0
    keypoint_enabled = True
    needle_enabled = True

    try:
        while True:
            ret, frame = cap.read()
            # TEST: Read from file instead of camera
            # frame = cv2.imread("img_1.png")
            # if frame is not None:
            #     ret = True
            # else:
            #     ret = False
            if not ret:
                print("Cannot read frame")
                break

            frame_count += 1
            fps_counter += 1

            # Process frame
            processor.needle_enabled = needle_enabled

            if keypoint_enabled:
                results = processor.process_frame(frame)
            else:
                # Gauge detection only
                results = {
                    'processed_frame': frame.copy(),
                    'gauge_detected': False,
                    'gauge_box': None
                }
                if frame_count % processor.frame_skip == 0:
                    results = processor._detect_gauge(frame, results)
                processor._draw_results(results)

            processed_frame = results['processed_frame']

            # Scale frame for better display and clicking
            display_scale = 1.5  # Make it 2.5x larger
            display_frame = cv2.resize(processed_frame,
                                       (int(processed_frame.shape[1] * display_scale),
                                        int(processed_frame.shape[0] * display_scale)),
                                       interpolation=cv2.INTER_LINEAR)

            # Calculate FPS
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = fps_counter / (current_time - start_time)
                fps_counter = 0
                start_time = current_time

            # Add info text
            info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Skip: {processor.frame_skip}/{processor.keypoint_skip}"
            cv2.putText(display_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Status text
            if results['gauge_detected']:
                if processor.stable_gauge_count >= processor.stable_threshold:
                    reading = results.get('reading')
                    if reading is not None and reading != "READY_TO_CALCULATE":
                        # Show reading with filtering info
                        history_size = len(processor.reading_history)
                        status_text = f"✅ {reading} ({history_size} samples)"
                        color = (0, 255, 0)
                    elif results.get('needle_line') is not None and results.get('keypoints') is not None:
                        status_text = "✅ CALCULATING..."
                        color = (0, 255, 255)
                    elif results.get('needle_line') is not None:
                        status_text = "✅ GAUGE + KEYPOINTS + NEEDLE"
                        color = (0, 255, 0)
                    elif results.get('keypoints') is not None:
                        status_text = "✅ GAUGE + KEYPOINTS"
                        color = (0, 255, 0)
                    else:
                        status_text = "✅ GAUGE STABLE"
                        color = (0, 255, 255)
                else:
                    status_text = "🔍 DETECTING..."
                    color = (0, 255, 255)
            else:
                status_text = "❌ NO GAUGE"
                color = (0, 0, 255)


            # Keypoint & Needle status
            kp_status = "ON" if keypoint_enabled else "OFF"
            needle_status = "ON" if needle_enabled else "OFF"
            cv2.putText(display_frame, f"Keypoints: {kp_status} | Needle: {needle_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Angle info if available
            if results.get('angle') is not None:
                angle_deg = np.degrees(results['angle'])
                cv2.putText(display_frame, f"Angle: {angle_deg:.1f}°", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Raw vs filtered reading info
            if results.get('raw_reading') is not None:
                raw_reading = results['raw_reading']
                cv2.putText(display_frame, f"Raw: {raw_reading:.1f}%", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Debug output thay cho GUI
            # Early exit for testing (remove later)

            # DÒNG MỚI (giảm frequency):


            # Show frame
            # cv2.imshow('Full Gauge Reader Pipeline', display_frame)

            # Handle keys
            # key = cv2.waitKey(1) & 0xFF
            # Show frame
            # Show frame
            cv2.imshow('Full Gauge Reader Pipeline', display_frame)

            # Make window resizable and set larger size (only once)
            if frame_count == 1:
                cv2.namedWindow('Full Gauge Reader Pipeline', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Full Gauge Reader Pipeline', 1280, 960)  # 2x larger
                print("📺 Window resized to 1280x960 for easier clicking")

            if frame_count == 1:  # First frame only
                def mouse_callback(event, x, y, flags, param):
                    if processor.waiting_for_click:
                        if event == cv2.EVENT_LBUTTONDOWN:
                            # Left click - auto-detect keypoints
                            actual_x = int(x / display_scale)
                            actual_y = int(y / display_scale)
                            processor._handle_mouse_click(actual_x, actual_y, force_manual=False)
                        elif event == cv2.EVENT_RBUTTONDOWN:
                            # Right click - force manual point (skip keypoint detection)
                            actual_x = int(x / display_scale)
                            actual_y = int(y / display_scale)
                            processor._handle_mouse_click(actual_x, actual_y, force_manual=True)

                cv2.setMouseCallback('Full Gauge Reader Pipeline', mouse_callback)
                print("🖱️ Mouse callback registered with 2.5x scaling compensation")


            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔄 Reset...")
                frame_count = 0
                processor.stable_gauge_count = 0
                processor.last_gauge_box = None
                processor.last_keypoints = None
                # Clear temporal filtering history
                processor.reading_history = []
                processor.angle_history = []
                processor.ellipse_history = []
            elif key == ord('+') or key == ord('='):
                processor.frame_skip = min(20, processor.frame_skip + 1)
                print(f"Frame skip: {processor.frame_skip}")
            elif key == ord('-'):
                processor.frame_skip = max(1, processor.frame_skip - 1)
                print(f"Frame skip: {processor.frame_skip}")
            elif key == ord('k'):
                keypoint_enabled = not keypoint_enabled
                print(f"Keypoint detection: {'ON' if keypoint_enabled else 'OFF'}")
            elif key == ord('n'):
                needle_enabled = not needle_enabled
                print(f"Needle detection: {'ON' if needle_enabled else 'OFF'}")
            elif key == ord('c'):
                processor.chart_enabled = not processor.chart_enabled
                print(f"Chart display: {'ON' if processor.chart_enabled else 'OFF'}")



    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Pipeline stopped")


if __name__ == "__main__":
    main()
