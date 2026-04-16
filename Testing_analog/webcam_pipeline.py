#!/usr/bin/env python3
"""
Webcam Pipeline - Universal Analog Gauge Reader
Outputs: Angle (degrees) + Percentage (0-100%) - Works with any analog gauge
Features: Temporal filtering, ellipse fitting, needle detection
"""

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



class FullGaugeProcessor:
    """
    Full pipeline processor với gauge detection, keypoint detection và needle segmentation
    """

    def __init__(self, detection_model_path, key_point_model_path, segmentation_model_path):
        self.detection_model_path = detection_model_path
        self.key_point_model_path = key_point_model_path
        self.segmentation_model_path = segmentation_model_path

        self.frame_skip = 5  # Skip frames cho gauge detection
        self.keypoint_skip = 15  # Skip nhiều hơn cho keypoint (heavy)
        self.frame_counter = 0

        # Cache results
        self.last_gauge_box = None
        self.last_keypoints = None
        self.last_needle_line = None

        # Temporal filtering for stability
        self.reading_history = []
        self.angle_history = []
        self.ellipse_history = []
        self.history_size = 10  # Keep last N readings

        # Stability tracking
        self.stable_gauge_count = 0
        self.stable_threshold = 3

        # Feature toggles
        self.needle_enabled = True

        # OCR Components for hybrid approach
        self.ocr_enabled = True
        self.calibration_interval = 30  # seconds
        self.last_calibration_time = None
        self.current_scale_mapping = None  # Dict: angle -> actual_value
        self.scale_range = None  # (min_val, max_val, unit)

        # THÊM DÒNG MỚI sau existing variables:
        self.last_interpolation_result = None
        self.last_interpolation_angle = None
        self.interpolation_cache_threshold = 1.0  # degrees

        self.detected_unit = None

        # Tạm thời disable Fast OCR (do environment issue)
        self.fast_ocr = None
        print("⚠️ Fast OCR temporarily disabled")

        # High-accuracy OCR (MMOCR for calibration)
        self.high_accuracy_ocr = None
        try:
            # Import test - không init ngay để tránh chậm startup
            import mmocr
            print("✅ MMOCR available for calibration")
        except Exception as e:
            print(f"⚠️ MMOCR not available: {e}")
            self.ocr_enabled = False

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

    def _preprocess_for_ocr(self, image):
        """
        Preprocess image for better OCR results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)

            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Sharpen
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            # Convert back to RGB for MMOCR
            if len(image.shape) == 3:
                processed_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

            return processed_rgb

        except Exception as e:
            print(f"OCR preprocessing error: {e}")
            return image

    def _tensor_to_numpy(self, tensor):
        """Safely convert tensor to numpy, handling CUDA tensors"""
        if torch.is_tensor(tensor):
            if tensor.is_cuda:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.detach().numpy()
        return tensor

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
        Xử lý frame với full pipeline
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

        # Step 1: Gauge Detection (mỗi N frames)
        if self.frame_counter % self.frame_skip == 0:
            results = self._detect_gauge(frame, results)
        else:
            # Reuse cached gauge
            if self.last_gauge_box is not None:
                results['gauge_detected'] = True
                results['gauge_box'] = self.last_gauge_box

        # Step 2: Keypoint Detection (khi gauge stable và mỗi M frames)
        if (results['gauge_detected'] and
                self.stable_gauge_count >= self.stable_threshold and
                self.frame_counter % self.keypoint_skip == 0 and
                self.key_point_inferencer is not None):

            results = self._detect_keypoints(frame, results)
        else:
            # Reuse cached keypoints
            if self.last_keypoints is not None:
                results['keypoints'] = self.last_keypoints

        # Step 3: OCR Integration (hybrid approach) - MOVED UP!
        if (results['gauge_detected'] and
                self.stable_gauge_count >= self.stable_threshold and
                self.ocr_enabled):

            # Check if calibration needed
            current_time = time.time()
            needs_calibration = (
                    self.last_calibration_time is None or
                    current_time - self.last_calibration_time > self.calibration_interval
            )

            if needs_calibration and self.frame_counter % (self.keypoint_skip * 2) == 0:
                # Periodic high-accuracy calibration
                print(f"🔄 Attempting OCR calibration (frame: {self.frame_counter})")
                calibration_success = self._calibrate_scale_with_ocr(frame, results)
                if calibration_success:
                    results['calibration_status'] = 'calibrated'
                    # Removed spam logging
                else:
                    results['calibration_status'] = 'failed'
            elif self.current_scale_mapping is not None:
                results['calibration_status'] = 'active'

        # Step 4: Needle Segmentation - MOVED DOWN!
        if (results['gauge_detected'] and
                self.stable_gauge_count >= self.stable_threshold and
                self.frame_counter % self.keypoint_skip == 0 and
                self.needle_enabled):
            results = self._detect_needle(frame, results)

        else:
            # Reuse cached needle
            if self.last_needle_line is not None and self.needle_enabled:
                results['needle_line'] = self.last_needle_line

        # Vẽ tất cả lên frame
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
                # print(f"✅ Needle detected (error: {needle_error:.3f})")

                # Calculate reading if we have both keypoints and needle
                if results.get('keypoints') is not None:
                    results = self._calculate_reading(results)
            # else:
            #     print("❌ No needle pixels found")

        except Exception as e:
            print(f"Needle segmentation error: {e}")

        return results

    def _calculate_reading(self, results):
        """Calculate final gauge reading from keypoints and needle with temporal filtering"""
        try:
            keypoints = results['keypoints']
            needle_line = results['needle_line']

            if len(keypoints) >= 3 and needle_line is not None:
                # Step 1: Use stable ellipse or fit new one
                ellipse_params = self._get_stable_ellipse(keypoints)
                if ellipse_params is None:
                    # print("❌ Failed to get stable ellipse")
                    return results

                # Step 2: Find needle-ellipse intersection
                needle_coeffs = needle_line['coeffs']
                needle_start_x, needle_end_x = needle_line['start'][0], needle_line['end'][0]

                intersection_point = self._find_needle_ellipse_intersection(
                    needle_coeffs, [needle_start_x, needle_end_x], ellipse_params
                )

                if intersection_point is None:
                    # print("❌ Needle-ellipse intersection failed")
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

                    if self.frame_counter % 30 == 0:  # Reduce logging frequency
                        print(f"✅ Reading: {filtered_reading:.2f} {unit} (angle: {np.degrees(angle):.1f}°)")

        except Exception as e:
            print(f"Reading calculation error: {e}")

        return results

    def _angle_to_actual_reading(self, angle, calibration_status=None, ellipse_params=None):
        """Convert angle to actual gauge reading using calibrated scale"""
        try:


            # Priority 1: Use calibrated scale mapping
            if (self.current_scale_mapping is not None and
                    calibration_status in ['active', 'calibrated']):

                # print(f"✅ Using calibrated scale mapping with {len(self.current_scale_mapping)} markers")
                return self._interpolate_from_scale_mapping(angle, ellipse_params)

            # Fallback: Generic percentage (current method)
            else:
                percentage = self._angle_to_generic_reading(angle)
                # print(f"📊 Using generic reading: {percentage:.1f}% (calibration: {calibration_status})")
                return percentage

        except Exception as e:
            print(f"Reading conversion error: {e}")
            return self._angle_to_generic_reading(angle)  # Safe fallback

    def _interpolate_from_scale_mapping(self, angle, ellipse_params=None):
        """Interpolate reading from calibrated scale mapping using proper angle fitting"""
        try:
            # DÒNG MỚI (chỉ log khi cần):
            if (self.last_interpolation_result is not None and
                    self.last_interpolation_angle is not None):
                angle_diff = abs(np.degrees(angle - self.last_interpolation_angle))
                if angle_diff < self.interpolation_cache_threshold:
                    return self.last_interpolation_result

            # if self.frame_counter % 60 == 0:  # Only log every 60 frames
                # print(f"🎯 Interpolation: angle={np.degrees(angle):.1f}°, {len(self.current_scale_mapping)} markers")
            if not self.current_scale_mapping or len(self.current_scale_mapping) < 2:
                return self._angle_to_generic_reading(angle)

            if ellipse_params is None:
                # Use last stored ellipse if available
                if hasattr(self, 'last_ellipse_params') and self.last_ellipse_params is not None:
                    ellipse_params = self.last_ellipse_params
                    # print("🔧 Using stored ellipse params for interpolation")
                else:
                    # print("❌ No ellipse params for proper interpolation, falling back to generic")
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
                    # print(f"❌ Failed to project marker {marker['value']}: {e}")
                    continue

            if len(number_labels) < 2:
                # print(f"❌ Not enough projected markers: {len(number_labels)}")
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
                            # print(
                            #     f"🎯 Calculated theta_zero: {np.degrees(theta_zero):.1f}° (start: {np.degrees(theta_start):.1f}°, end: {np.degrees(theta_end):.1f}°)")
                        else:
                            theta_zero = np.pi  # Bottom fallback
                            # print("⚠️ Using fallback theta_zero: 180°")
                    else:
                        theta_zero = np.pi  # Bottom fallback
                        # print("⚠️ Not enough keypoint groups, using fallback theta_zero: 180°")
                except Exception as e:
                    # print(f"❌ Zero point calculation error: {e}")
                    theta_zero = np.pi  # Bottom fallback
            else:
                theta_zero = np.pi  # Bottom fallback
                # print("⚠️ No keypoints available, using fallback theta_zero: 180°")

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
                # print(f"✅ RANSAC fit: {np.sum(inlier_mask)}/{len(inlier_mask)} inliers")
            except Exception as e:
                # print(f"⚠️ RANSAC failed, using simple fit: {e}")
                reading_line_coeff = line_fit(angle_number_arr[:, 0], angle_number_arr[:, 1])

            # Create reading line function
            reading_line = np.poly1d(reading_line_coeff)

            # Convert needle angle and get reading
            needle_angle_conv = angle_converter.convert_angle(angle)
            reading = reading_line(needle_angle_conv)
            # DÒNG MỚI (chỉ log 1 lần khi calibration thành công):
            # if self.frame_counter % 60 == 0:  # Reduce logging frequency
            #     print(f"🎯 Calibrated reading: {reading:.1f} from {len(number_labels)} markers")
            return reading


        except Exception as e:
            # print(f"Interpolation error: {e}")
            return self._angle_to_generic_reading(angle)

    def _get_stable_ellipse(self, keypoints):
        """Get stable ellipse parameters using history"""
        try:
            # Try to fit new ellipse
            new_ellipse = self._fit_ellipse_from_keypoints(keypoints)

            if new_ellipse is not None:
                # Add to history
                self.ellipse_history.append(new_ellipse)
                if len(self.ellipse_history) > self.history_size:
                    self.ellipse_history.pop(0)

                # Use median of recent ellipses for stability
                if len(self.ellipse_history) >= 3:
                    # Average ellipse parameters
                    avg_ellipse = np.mean(self.ellipse_history, axis=0)
                    # Store for reuse in calibration
                    self.last_ellipse_params = avg_ellipse
                    return avg_ellipse

                else:
                    self.last_ellipse_params = new_ellipse
                    return new_ellipse

            # Fallback to last good ellipse
            if len(self.ellipse_history) > 0:
                return self.ellipse_history[-1]

            return None

        except Exception as e:
            # print(f"Stable ellipse error: {e}")
            return None

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
            # print(f"Generic reading conversion error: {e}")
            return 0

    def _apply_temporal_filter(self, raw_reading, angle):
        """Apply temporal filtering to stabilize readings"""
        try:
            # Add to history
            self.reading_history.append(raw_reading)
            self.angle_history.append(angle)

            # Maintain history size
            if len(self.reading_history) > self.history_size:
                self.reading_history.pop(0)
            if len(self.angle_history) > self.history_size:
                self.angle_history.pop(0)

            # Need at least 3 readings for filtering
            if len(self.reading_history) < 3:
                return raw_reading

            # Use median filter to remove outliers
            filtered_reading = np.median(self.reading_history[-5:])  # Last 5 readings

            # Additional smoothing with moving average
            if len(self.reading_history) >= 5:
                smooth_reading = np.mean(self.reading_history[-3:])  # Last 3 readings
                # Weighted combination
                final_reading = 0.7 * filtered_reading + 0.3 * smooth_reading
            else:
                final_reading = filtered_reading

            return final_reading

        except Exception as e:
            # print(f"Temporal filter error: {e}")
            return raw_reading

    def _calibrate_scale_with_ocr(self, frame, results):
        """FIXED: Use WARP OCR with rotation like pipeline.py"""
        try:
            if not self.ocr_enabled or not results['gauge_detected']:
                return False

            # print("🔍 Starting WARP OCR calibration (pipeline.py style)...")

            # Import needed functions from pipeline.py
            from gauge_reader_web.ocr.ocr_inference import ocr, ocr_warp
            from gauge_reader_web.geometry.ellipse import get_polar_angle, get_theta_middle, get_point_from_angle

            gauge_box = results['gauge_box']
            cropped_img = self._crop_image(frame, gauge_box)

            # CRITICAL: Convert BGR to RGB like pipeline.py
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            # print(f"🔧 Converted BGR to RGB for OCR compatibility")

            # Get ellipse params and keypoints
            if not hasattr(self, 'last_keypoints') or self.last_keypoints is None:
                # print("❌ No keypoints available for OCR calibration")
                return False

            keypoints = self.last_keypoints
            ellipse_params = self._fit_ellipse_from_keypoints(keypoints)
            if ellipse_params is None:
                # print("❌ Could not fit ellipse for OCR")
                return False

            # Calculate zero point like pipeline.py
            RESOLUTION = (448, 448)  # Same as pipeline.py
            WRAP_AROUND_FIX = True  # Same as pipeline.py

            if WRAP_AROUND_FIX and len(keypoints) >= 2:
                start_points = keypoints[0]  # Start notch keypoints
                end_points = keypoints[-1]  # End notch keypoints

                if len(start_points) > 0 and len(end_points) > 0:
                    start_point = np.mean(start_points, axis=0).flatten()
                    end_point = np.mean(end_points, axis=0).flatten()

                    theta_start = get_polar_angle(start_point, ellipse_params)
                    theta_end = get_polar_angle(end_point, ellipse_params)
                    theta_zero = get_theta_middle(theta_start, theta_end)
                else:
                    # Fallback to bottom middle
                    bottom_middle = np.array((RESOLUTION[0] / 2, RESOLUTION[1]))
                    theta_zero = get_polar_angle(bottom_middle, ellipse_params)
            else:
                # Fallback to bottom middle
                bottom_middle = np.array((RESOLUTION[0] / 2, RESOLUTION[1]))
                theta_zero = get_polar_angle(bottom_middle, ellipse_params)

            zero_point = get_point_from_angle(theta_zero, ellipse_params)
            # print(f"🎯 Zero point calculated: {zero_point} (theta: {np.degrees(theta_zero):.1f}°)")

            # CRITICAL: Use original cropped image resolution like pipeline.py
            cropped_img_resolution = (cropped_img_rgb.shape[1], cropped_img_rgb.shape[0])
            # print(f"🔧 Original cropped resolution: {cropped_img_resolution}")

            # Helper functions from pipeline.py
            def move_point_resize(point, original_resolution, resized_resolution):
                new_point_x = point[0] * resized_resolution[0] / original_resolution[0]
                new_point_y = point[1] * resized_resolution[1] / original_resolution[1]
                return new_point_x, new_point_y

            def rescale_ellipse_resize(ellipse_params, original_resolution, resized_resolution):
                x0, y0, ap, bp, phi = ellipse_params
                # move ellipse center
                x0_new, y0_new = move_point_resize((x0, y0), original_resolution, resized_resolution)
                # rescale axis
                scaling_factor = resized_resolution[0] / original_resolution[0]
                ap_new = scaling_factor * ap
                bp_new = scaling_factor * bp
                return x0_new, y0_new, ap_new, bp_new, phi

            # Scale zero point and ellipse params to original cropped image resolution like pipeline.py
            res_zero_point = list(move_point_resize(zero_point, RESOLUTION, cropped_img_resolution))
            res_ellipse_params = rescale_ellipse_resize(ellipse_params, RESOLUTION, cropped_img_resolution)

            # print(f"🔧 Scaled zero point: {res_zero_point}")
            # print(f"🔧 Scaled ellipse params: {res_ellipse_params}")

            # Use WARP OCR with rotation like pipeline.py
            WARP_OCR = True
            ZERO_POINT_ROTATION = True
            OCR_ROTATION = ZERO_POINT_ROTATION

            if WARP_OCR and OCR_ROTATION:
                # print("🔍 Using WARP OCR with zero-point rotation...")

                # Create a dummy plotter for ocr_warp (it needs this parameter)
                class DummyPlotter:
                    def plot_any_image(self, *args, **kwargs):
                        pass

                    def plot_ocr_visualization(self, *args, **kwargs):
                        pass

                dummy_plotter = DummyPlotter()
                debug = False  # Set to False to avoid plotting

                try:
                    ocr_readings, ocr_visualization, degree = ocr_warp(
                        cropped_img_rgb,
                        res_zero_point,
                        res_ellipse_params,
                        dummy_plotter,
                        debug,
                        False,  # RANDOM_ROTATIONS = False
                        True  # ZERO_POINT_ROTATION = True
                    )
                    # print(f"✅ WARP OCR completed with rotation: {degree}°")
                    # print(f"📊 WARP OCR detected: {len(ocr_readings)} readings")

                except Exception as e:
                    # print(f"❌ WARP OCR failed: {e}")
                    # Fallback to basic OCR
                    # print("🔄 Falling back to basic OCR...")
                    ocr_readings = ocr(cropped_img_rgb, False)
                    # print(f"📊 Basic OCR fallback: {len(ocr_readings)} readings")

            else:
                # Fallback to basic OCR
                # print("🔍 Using basic OCR...")
                ocr_readings = ocr(cropped_img_rgb, False)
                # print(f"📊 Basic OCR: {len(ocr_readings)} readings")

            # Detect unit like pipeline.py
            unit_readings = []
            for reading in ocr_readings:
                if reading.is_unit():
                    unit_readings.append(reading)

            if len(unit_readings) == 1:
                self.detected_unit = unit_readings[0].reading
                # print(f"📏 Unit detected: {self.detected_unit}")
            elif len(unit_readings) == 0:
                self.detected_unit = None
                # print(f"📏 No unit detected")
            else:
                self.detected_unit = None
                # print(f"📏 Multiple units detected: {len(unit_readings)}")

            # Scale OCR coordinates back to 448x448 space like pipeline.py
            for reading in ocr_readings:
                polygon = reading.polygon
                polygon[:, 0] = polygon[:, 0] * RESOLUTION[1] / cropped_img_rgb.shape[1]
                polygon[:, 1] = polygon[:, 1] * RESOLUTION[0] / cropped_img_rgb.shape[0]
                reading.set_polygon(polygon)

            # Filter number labels like pipeline.py
            OCR_THRESHOLD = 0.7
            number_labels = []
            for reading in ocr_readings:
                if reading.is_number() and reading.confidence > OCR_THRESHOLD:
                    # Add heuristics to filter out serial numbers
                    if not (
                            abs(reading.number) > 10000
                            or (abs(reading.number) > 100 and reading.number % 10 != 0)
                    ):
                        number_labels.append(reading)

            print(f"📊 Filtered number labels: {len(number_labels)}")
            for reading in number_labels:
                print(f"  - {reading.number} (conf: {reading.confidence:.3f})")

            if len(number_labels) >= 2:
                # Convert to scale markers format for webcam pipeline
                scale_markers = []
                for reading in number_labels:
                    scale_markers.append({
                        'value': reading.number,
                        'position': reading.center,
                        'confidence': reading.confidence
                    })

                self.current_scale_mapping = scale_markers
                self.last_calibration_time = time.time()

                # Store ellipse params for interpolation
                self.last_ellipse_params = ellipse_params

                values = [m['value'] for m in scale_markers]
                self.scale_range = (min(values), max(values), "units")

                print(
                    f"✅ WARP OCR calibration success: {len(scale_markers)} markers, range: {min(values)}-{max(values)}")
                return True
            else:
                print(f"❌ Insufficient number labels: {len(number_labels)}")
                return False

        except Exception as e:
            print(f"❌ WARP OCR calibration error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Also add the helper functions at class level
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

            # print(f"✅ Ellipse fitted from {len(all_points)} points")
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
                        # if self.frame_counter % (self.frame_skip * 4) == 0:  # Print less frequently
                            # print(f"✅ Gauge detected (stable: {self.stable_gauge_count})")

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
                # print(f"✅ Keypoints detected: {[len(kp) for kp in processed_keypoints]}")
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
        """Vẽ tất cả results lên frame"""
        frame = results['processed_frame']

        # Draw gauge bounding box
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

        # Draw keypoints
        if (results['keypoints'] is not None and
                results['gauge_box'] is not None):
            self._draw_keypoints(frame, results['keypoints'], results['gauge_box'])

        # Draw needle line
        if (results.get('needle_line') is not None and
                results['gauge_box'] is not None):
            self._draw_needle_line(frame, results['needle_line'], results['gauge_box'])

        # Draw reading
        if results.get('reading') is not None:
            self._draw_reading(frame, results['reading'])

        # Draw ellipse and intersection if available
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
        """Vẽ needle line lên frame"""
        try:
            if isinstance(gauge_box, (list, tuple)) and len(gauge_box) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in gauge_box[:4]]
                gauge_width = x2 - x1
                gauge_height = y2 - y1

                # Scale từ 448x448 về gauge size
                scale_x = gauge_width / 448
                scale_y = gauge_height / 448

                # Draw needle pixels as dots
                if 'mask_x' in needle_line and 'mask_y' in needle_line:
                    mask_x = needle_line['mask_x']
                    mask_y = needle_line['mask_y']

                    for px, py in zip(mask_x, mask_y):
                        x = int(x1 + px * scale_x)
                        y = int(y1 + py * scale_y)
                        cv2.circle(frame, (x, y), 1, (0, 165, 255), -1)  # Orange dots

                # Draw fitted line
                if 'start' in needle_line and 'end' in needle_line:
                    start_x = int(x1 + needle_line['start'][0] * scale_x)
                    start_y = int(y1 + needle_line['start'][1] * scale_y)
                    end_x = int(x1 + needle_line['end'][0] * scale_x)
                    end_y = int(y1 + needle_line['end'][1] * scale_y)

                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 165, 255), 2)
                    cv2.putText(frame, 'NEEDLE', (start_x + 5, start_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        except Exception as e:
            print(f"Draw needle line error: {e}")

    def _draw_reading(self, frame, reading):
        """Enhanced reading display with OCR calibration info"""
        try:
            # Main reading display
            if reading == "READY_TO_CALCULATE":
                text = "READY TO CALCULATE"
                color = (0, 255, 0)
            else:
                # Get unit from class attribute instead of results parameter
                unit = getattr(self, 'detected_unit', 'units')
                if unit is None:
                    unit = 'units'
                text = f"Reading: {reading} {unit}"
                color = (0, 255, 0)

            cv2.putText(frame, text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # OCR Calibration Status
            if hasattr(self, 'current_scale_mapping') and self.current_scale_mapping is not None:
                cal_text = f"OCR: {len(self.current_scale_mapping)} markers"
                cal_color = (0, 255, 0)  # Green - calibrated

                # Show scale range if available
                if hasattr(self, 'scale_range') and self.scale_range:
                    min_val, max_val, _ = self.scale_range
                    range_text = f"Range: {min_val:.0f}-{max_val:.0f}"
                    cv2.putText(frame, range_text, (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cal_color, 1)
            else:
                cal_text = "OCR: Not calibrated"
                cal_color = (0, 255, 255)  # Yellow - not calibrated

            cv2.putText(frame, cal_text, (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1)

            # Last calibration time
            if hasattr(self, 'last_calibration_time') and self.last_calibration_time:
                time_since = time.time() - self.last_calibration_time
                time_text = f"Last cal: {time_since:.0f}s ago"
                cv2.putText(frame, time_text, (10, 195),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        except Exception as e:
            print(f"Draw reading error: {e}")

    # And add this line in __init__ method:
    # self.detected_unit = None  # Store detected unit from OCR
    def _draw_ellipse(self, frame, ellipse_params, gauge_box):
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

    # Initialize processor
    processor = FullGaugeProcessor(args.detection_model, args.key_point_model, args.segmentation_model)
    processor.frame_skip = args.frame_skip
    processor.keypoint_skip = args.keypoint_skip

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # print("🚀 Universal Analog Gauge Reader Pipeline Ready!")
    # print("📊 Output Format:")
    # print("   - Angle: Raw needle angle in degrees (0-360°)")
    # print("   - Percentage: Normalized reading (0-100%)")
    # print("   - User maps percentage to actual scale values")
    # print("📋 Controls:")
    # print("   'q' - Quit")
    # print("   's' - Save frame")
    # print("   'r' - Reset (clears temporal filtering)")
    # print("   '+' - Increase frame skip")
    # print("   '-' - Decrease frame skip")
    # print("   'k' - Toggle keypoint detection")
    # print("   'n' - Toggle needle detection")

    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    fps = 0
    keypoint_enabled = True
    needle_enabled = True

    try:
        while True:
            # ret, frame = cap.read()
            # TEST: Read from file instead of camera
            frame = cv2.imread("img_1.png")
            if frame is not None:
                ret = True
            else:
                ret = False
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

            # Calculate FPS
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = fps_counter / (current_time - start_time)
                fps_counter = 0
                start_time = current_time

            # Add info text
            info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Skip: {processor.frame_skip}/{processor.keypoint_skip}"
            cv2.putText(processed_frame, info_text, (10, 30),
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

            cv2.putText(processed_frame, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Keypoint & Needle status
            kp_status = "ON" if keypoint_enabled else "OFF"
            needle_status = "ON" if needle_enabled else "OFF"
            cv2.putText(processed_frame, f"Keypoints: {kp_status} | Needle: {needle_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Angle info if available
            if results.get('angle') is not None:
                angle_deg = np.degrees(results['angle'])
                cv2.putText(processed_frame, f"Angle: {angle_deg:.1f}°", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Raw vs filtered reading info
            if results.get('raw_reading') is not None:
                raw_reading = results['raw_reading']
                cv2.putText(processed_frame, f"Raw: {raw_reading:.1f}%", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Debug output thay cho GUI
            # Early exit for testing (remove later)

            # DÒNG MỚI (giảm frequency):
            if frame_count % 90 == 0:  # Print every 90 frames instead of 30
                print(f"📊 Frame {frame_count}: {status_text}")
                if results.get('reading'):
                    print(f"   Reading: {results['reading']}")
                if hasattr(processor, 'current_scale_mapping') and processor.current_scale_mapping:
                    print(f"   📍 OCR: {len(processor.current_scale_mapping)} markers calibrated")

            # Show frame
            # cv2.imshow('Full Gauge Reader Pipeline', processed_frame)

            # Handle keys
            # key = cv2.waitKey(1) & 0xFF
            # Show frame
            cv2.imshow('Full Gauge Reader Pipeline', processed_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.save_frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gauge_full_{timestamp}.png"
                cv2.imwrite(filename, processed_frame)
                print(f"💾 Saved: {filename}")
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
                # Manual calibration trigger
                if processor.ocr_enabled and len(processor.reading_history) > 0:
                    print("🔄 Manual OCR calibration triggered...")
                    # Get latest results from processor
                    dummy_results = {
                        'gauge_detected': processor.last_gauge_box is not None,
                        'gauge_box': processor.last_gauge_box
                    }
                    success = processor._calibrate_scale_with_ocr(frame, dummy_results)
                    if success:
                        print("✅ Manual calibration successful")
                    else:
                        print("❌ Manual calibration failed")
                else:
                    print("⚠️ OCR not enabled or no gauge detected")

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