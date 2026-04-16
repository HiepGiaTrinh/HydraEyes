#!/usr/bin/env python3
"""
System Test for still2.py - Camera-based Analog Gauge Reader
Tests the FullGaugeProcessor class with images from gauges folder
Simulates camera input and measures accuracy according to paper metrics
Target: ≤2% relative reading error
"""

import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the FullGaugeProcessor from tools/still_processor.py
from tools.still_processor import FullGaugeProcessor

class Still2SystemTester:
    """
    Test system for still2.py FullGaugeProcessor
    Simulates camera-based gauge reading with static images
    """

    def __init__(self, gauges_dir="data/gauges",
                 detection_model="gauge_reader_web/models/gauge_detection_model.pt",
                 keypoint_model="gauge_reader_web/models/keypoint_model.pt",
                 segmentation_model="gauge_reader_web/models/needle_segmentation_model.pt"):

        self.gauges_dir = gauges_dir
        self.detection_model = detection_model
        self.keypoint_model = keypoint_model
        self.segmentation_model = segmentation_model

        # Test categorization (based on paper)
        self.test_categories = {
            'Front': [],        # Direct front view
            'Angled': [],       # Angled view
            'Rotated': [],      # Rotated gauges
            'Mixed': [],        # Angled + Rotated
            'All': []           # All images
        }

        # Results storage
        self.results = []
        self.failed_cases = []

        # Performance metrics
        self.paper_target_error = 2.0  # ≤2% relative error (paper target)

    def extract_true_value(self, filename):
        """Extract ground truth value from filename like simple_test_gauges.py"""
        try:
            name = os.path.splitext(filename)[0]
            if name.startswith('gauge_'):
                value_part = name[6:]  # Remove "gauge_"

                # gauge_-5 -> -5.0
                if value_part.startswith('-'):
                    return float(value_part)

                # gauge_0-5 -> 0.5
                if '-' in value_part:
                    parts = value_part.split('-')
                    if len(parts) == 2:
                        return float(f"{parts[0]}.{parts[1]}")

                # gauge_5 -> 5.0
                return float(value_part)
        except:
            pass
        return None

    def categorize_image(self, filename):
        """Categorize image based on name or content (simplified)"""
        name = filename.lower()

        # Simple categorization based on filename
        if 'angled' in name or 'angle' in name:
            if 'rot' in name:
                return 'Mixed'
            return 'Angled'
        elif 'rot' in name:
            return 'Rotated'
        else:
            return 'Front'  # Default to front view

    def setup_processor_for_testing(self):
        """Setup processor with optimal settings for testing"""
        processor = FullGaugeProcessor(
            self.detection_model,
            self.keypoint_model,
            self.segmentation_model
        )

        # Optimize for testing
        processor.frame_skip = 1  # Process every frame for accuracy
        processor.keypoint_skip = 1  # Detect keypoints every frame
        processor.history_size = 1  # Minimal filtering for testing

        # Enable manual calibration mode
        processor.manual_calibration_mode = True
        processor.required_calibration_points = 4  # Standard calibration

        return processor

    def interactive_calibration(self, image_path, processor):
        """Interactive calibration using still2.py UI"""
        try:
            print(f"📸 Loading calibration image: {os.path.basename(image_path)}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not load calibration image")

            # Setup processor for calibration
            processor.reset_for_testing()

            # Process frames to stabilize
            stable_count = 0
            for attempt in range(20):
                results = processor.process_frame(image)
                if results.get('gauge_detected'):
                    stable_count += 1
                    if stable_count >= 3:
                        break
                else:
                    stable_count = 0

            if stable_count < 3:
                raise Exception("Could not detect gauge for calibration")

            print("✅ Gauge detected and stable")
            print("\n🎯 INTERACTIVE CALIBRATION MODE")
            print("📋 Instructions:")
            print("   1. Click on scale markers on the gauge (4 points)")
            print("   2. Enter the value for each point when prompted")
            print("   3. This calibration will be used for all test images")
            print("\n🖱️ Controls:")
            print("   LEFT CLICK - Place calibration point")
            print("   'q' - Quit calibration")

            # Show calibration window
            self.show_calibration_window(image, processor)

            # Wait for calibration completion
            if processor.manual_calibrated:
                print("✅ Interactive calibration completed successfully!")
                return True
            else:
                print("❌ Calibration was not completed")
                return False

        except Exception as e:
            print(f"❌ Interactive calibration failed: {e}")
            return False

    def show_calibration_window(self, image, processor):
        """Show calibration window with mouse interaction"""
        display_scale = 2.0

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and processor.waiting_for_click:
                # Convert to original coordinates
                actual_x = int(x / display_scale)
                actual_y = int(y / display_scale)
                processor._handle_mouse_click(actual_x, actual_y, force_manual=False)

        # Initial processing to enable clicking
        processor.waiting_for_click = True

        while True:
            # Process frame
            results = processor.process_frame(image)
            display_frame = results['processed_frame'].copy()

            # Add calibration status
            status_lines = []
            if processor.manual_calibrated:
                status_lines.append("✅ CALIBRATION COMPLETE - Press 'q' to continue")
                color = (0, 255, 0)
            elif processor.waiting_for_click:
                status_lines.append(f"🎯 Click point {processor.current_calibration_step + 1}/{processor.required_calibration_points}")
                status_lines.append("Enter value when prompted in console")
                color = (0, 255, 255)
            else:
                status_lines.append("⏳ Wait for gauge to stabilize...")
                color = (255, 255, 0)

            for i, line in enumerate(status_lines):
                cv2.putText(display_frame, line, (10, 40 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Scale for better viewing
            display_height, display_width = display_frame.shape[:2]
            new_width = int(display_width * display_scale)
            new_height = int(display_height * display_scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))

            # Show window
            cv2.imshow('Gauge Calibration', display_frame)
            cv2.setMouseCallback('Gauge Calibration', mouse_callback)

            # Handle keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                if processor.manual_calibrated:
                    cv2.destroyAllWindows()
                    break
                else:
                    # Ask for confirmation
                    print("\n⚠️ Calibration not complete. Quit anyway? (y/n): ", end='')
                    response = input().strip().lower()
                    if response == 'y':
                        cv2.destroyAllWindows()
                        break

            # Auto exit when calibration complete
            if processor.manual_calibrated and processor.current_calibration_step >= processor.required_calibration_points:
                print("🎉 Calibration auto-completed!")
                time.sleep(2)  # Show completion for 2 seconds
                cv2.destroyAllWindows()
                break

    def test_single_image_with_calibration(self, img_path, processor):
        """Test single image using existing calibration"""
        filename = os.path.basename(img_path)
        true_value = self.extract_true_value(filename)

        if true_value is None:
            print(f"⚠️ Skip {filename} - no ground truth value")
            return None

        print(f"📊 Testing: {filename} (expected: {true_value})")

        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise Exception("Could not load image")

            # Reset frame counter but keep calibration
            processor.frame_counter = 0
            processor.stable_gauge_count = 0
            processor.last_gauge_box = None
            processor.last_keypoints = None
            processor.last_needle_line = None
            processor.reading_history = []
            processor.angle_history = []
            processor.ellipse_history = []

            # Read gauge multiple times
            readings = []
            processing_times = []

            for attempt in range(10):
                start_time = time.time()
                results = processor.process_frame(image)
                end_time = time.time()

                processing_times.append((end_time - start_time) * 1000)

                if results.get('reading') and results['reading'] != "READY_TO_CALCULATE":
                    try:
                        reading = float(results['reading'])
                        readings.append(reading)
                        if attempt < 3:  # Only show first few
                            print(f"   Reading {attempt+1}: {reading:.2f}")
                    except ValueError:
                        continue

                time.sleep(0.02)  # Small delay

            if not readings:
                raise Exception("No valid readings obtained")

            # Calculate results
            predicted = np.median(readings)
            avg_time = np.mean(processing_times)
            reading_std = np.std(readings) if len(readings) > 1 else 0

            abs_error = abs(predicted - true_value)
            rel_error = (abs_error / abs(true_value)) * 100 if true_value != 0 else abs_error * 100

            # Status
            if rel_error <= 2.0:
                status = "EXCELLENT"
                color = "🟢"
            elif rel_error <= 5.0:
                status = "GOOD"
                color = "🟡"
            elif rel_error <= 10.0:
                status = "OK"
                color = "🟠"
            else:
                status = "POOR"
                color = "🔴"

            print(f"✅ Predicted: {predicted:.3f} (from {len(readings)} readings)")
            print(f"   Error: {abs_error:.3f} ({rel_error:.2f}%)")
            print(f"   Time: {avg_time:.1f}ms")
            print(f"   Status: {color} {status}")

            return {
                'filename': filename,
                'true_value': true_value,
                'predicted': predicted,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'processing_time': avg_time,
                'readings_count': len(readings),
                'reading_std': reading_std,
                'status': status,
                'category': self.categorize_image(filename)
            }

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            return {
                'filename': filename,
                'true_value': true_value,
                'error': str(e),
                'failed': True
            }

    def test_single_image(self, img_path, processor):
        """Test single image through still2 pipeline"""
        filename = os.path.basename(img_path)
        true_value = self.extract_true_value(filename)

        if true_value is None:
            print(f"⚠️ Skip {filename} - no ground truth value")
            return None

        print(f"\n📊 Testing: {filename} (expected: {true_value})")

        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise Exception("Could not load image")

            # Simulate calibration if needed
            if not processor.manual_calibrated:
                gauge_range = (0, 15)  # Default PSI range
                calibration_success = self.simulate_manual_calibration(
                    processor, image, true_value, gauge_range
                )
                if not calibration_success:
                    raise Exception("Calibration failed")

            # Process image multiple times for stability
            readings = []
            processing_times = []

            for attempt in range(5):  # 5 attempts for averaging
                start_time = time.time()
                results = processor.process_frame(image)
                end_time = time.time()

                processing_times.append((end_time - start_time) * 1000)

                if results.get('reading') and results['reading'] != "READY_TO_CALCULATE":
                    try:
                        reading = float(results['reading'])
                        readings.append(reading)
                    except ValueError:
                        continue

            if not readings:
                raise Exception("No valid readings obtained")

            # Take median reading for stability
            predicted = np.median(readings)
            avg_time = np.mean(processing_times)

            # Calculate errors
            abs_error = abs(predicted - true_value)
            rel_error = (abs_error / abs(true_value)) * 100 if true_value != 0 else abs_error * 100

            print(f"✅ Predicted: {predicted:.3f} (from {len(readings)} readings)")
            print(f"   Error: {abs_error:.3f} ({rel_error:.2f}%)")
            print(f"   Time: {avg_time:.1f}ms")

            # Categorize result quality
            if rel_error <= 2.0:
                status = "EXCELLENT"
                color = "🟢"
            elif rel_error <= 5.0:
                status = "GOOD"
                color = "🟡"
            elif rel_error <= 10.0:
                status = "OK"
                color = "🟠"
            else:
                status = "POOR"
                color = "🔴"

            print(f"   Status: {color} {status}")

            return {
                'filename': filename,
                'true_value': true_value,
                'predicted': predicted,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'processing_time': avg_time,
                'readings_count': len(readings),
                'status': status,
                'category': self.categorize_image(filename)
            }

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            return {
                'filename': filename,
                'true_value': true_value,
                'error': str(e),
                'failed': True
            }

    def run_full_test(self):
        """Run complete test suite with shared calibration"""
        print("🚀 Still2.py System Test Suite")
        print("=" * 60)

        # Check requirements
        for path, name in [(self.gauges_dir, "gauges folder"),
                           (self.detection_model, "detection model"),
                           (self.keypoint_model, "keypoint model"),
                           (self.segmentation_model, "segmentation model")]:
            if not os.path.exists(path):
                print(f"❌ Missing {name}: {path}")
                return

        # Find images
        image_files = (glob.glob(os.path.join(self.gauges_dir, "*.png")) +
                       glob.glob(os.path.join(self.gauges_dir, "*.jpg")))

        if not image_files:
            print(f"❌ No images found in {self.gauges_dir}")
            return

        print(f"🔍 Found {len(image_files)} gauge images")
        print(f"🎯 Target: ≤{self.paper_target_error}% relative error (paper standard)")
        print("🔧 Testing mode: One-time calibration for all images")

        # STEP 1: One-time calibration using first image
        print(f"\n🎯 STEP 1: Interactive Calibration")
        print("Using first image for calibration...")

        first_image = image_files[0]
        processor = self.setup_processor_for_testing()

        calibration_success = self.interactive_calibration(first_image, processor)
        if not calibration_success:
            print("❌ Calibration failed, cannot continue")
            return

        print("✅ Calibration completed, now testing all images...")

        # STEP 2: Test all images with same calibration
        successful_tests = []
        failed_tests = []

        for i, img_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}]", "="*50)

            result = self.test_single_image_with_calibration(img_path, processor)

            if result:
                if result.get('failed'):
                    failed_tests.append(result)
                else:
                    successful_tests.append(result)

        # Analyze results
        self.analyze_results(successful_tests, failed_tests)
        self.plot_results(successful_tests)

        return successful_tests, failed_tests

    def analyze_results(self, successful_tests, failed_tests):
        """Analyze and display results according to paper metrics"""
        if not successful_tests:
            print("\n❌ No successful tests!")
            return

        print(f"\n" + "="*80)
        print("📊 STILL2.PY SYSTEM TEST RESULTS")
        print("="*80)

        # Basic metrics
        total_tests = len(successful_tests) + len(failed_tests)
        success_rate = len(successful_tests) / total_tests * 100

        rel_errors = [t['rel_error'] for t in successful_tests]
        abs_errors = [t['abs_error'] for t in successful_tests]
        times = [t['processing_time'] for t in successful_tests]

        mean_rel_error = np.mean(rel_errors)
        mean_abs_error = np.mean(abs_errors)
        mean_time = np.mean(times)

        print(f"📈 OVERALL PERFORMANCE:")
        print(f"  Success Rate:           {success_rate:.1f}% ({len(successful_tests)}/{total_tests})")
        print(f"  Mean Relative Error:    {mean_rel_error:.2f}%")
        print(f"  Mean Absolute Error:    {mean_abs_error:.3f}")
        print(f"  Average Processing:     {mean_time:.1f}ms")
        print("-" * 80)

        # Accuracy breakdown (paper categories)
        excellent = sum(1 for e in rel_errors if e <= 2.0)  # Paper target
        good = sum(1 for e in rel_errors if e <= 5.0)
        ok = sum(1 for e in rel_errors if e <= 10.0)

        print(f"🎯 ACCURACY ANALYSIS (Paper Standards):")
        print(f"  Excellent (≤2%):        {excellent/len(successful_tests)*100:.1f}% ({excellent}/{len(successful_tests)})")
        print(f"  Good (≤5%):             {good/len(successful_tests)*100:.1f}% ({good}/{len(successful_tests)})")
        print(f"  Acceptable (≤10%):      {ok/len(successful_tests)*100:.1f}% ({ok}/{len(successful_tests)})")
        print("-" * 80)

        # Paper comparison
        paper_target = self.paper_target_error
        if mean_rel_error <= paper_target:
            print(f"🎯 ✅ MEETS PAPER TARGET: {mean_rel_error:.2f}% ≤ {paper_target}%")
        else:
            print(f"🎯 ❌ BELOW PAPER TARGET: {mean_rel_error:.2f}% > {paper_target}%")

        # Category breakdown
        categories = {}
        for test in successful_tests:
            cat = test.get('category', 'Unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test['rel_error'])

        if len(categories) > 1:
            print(f"\n📊 PERFORMANCE BY CATEGORY:")
            for cat, errors in categories.items():
                mean_cat_error = np.mean(errors)
                print(f"  {cat:12s}: {mean_cat_error:.2f}% ({len(errors)} images)")

        # Best/Worst cases
        best = min(successful_tests, key=lambda x: x['rel_error'])
        worst = max(successful_tests, key=lambda x: x['rel_error'])

        print(f"\n🏆 BEST:  {best['filename']} - {best['rel_error']:.2f}% error")
        print(f"💔 WORST: {worst['filename']} - {worst['rel_error']:.2f}% error")

        # Failed cases
        if failed_tests:
            print(f"\n💥 FAILED CASES ({len(failed_tests)}):")
            for failure in failed_tests[:5]:  # Show first 5
                print(f"  - {failure['filename']}: {failure.get('error', 'Unknown error')}")
            if len(failed_tests) > 5:
                print(f"  ... and {len(failed_tests) - 5} more failures")

        print("="*80)

        # System assessment
        if success_rate >= 90 and mean_rel_error <= 2.0:
            print("🎯 ✅ SYSTEM ASSESSMENT: EXCELLENT - Ready for deployment")
        elif success_rate >= 80 and mean_rel_error <= 5.0:
            print("🎯 🟡 SYSTEM ASSESSMENT: GOOD - Minor improvements needed")
        elif success_rate >= 70 and mean_rel_error <= 10.0:
            print("🎯 ⚠️  SYSTEM ASSESSMENT: ACCEPTABLE - Significant improvements needed")
        else:
            print("🎯 ❌ SYSTEM ASSESSMENT: POOR - Major issues require fixing")

    def plot_results(self, successful_tests):
        """Plot comprehensive results analysis"""
        if not successful_tests:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Still2.py System Test Results - Camera-based Gauge Reading',
                     fontsize=16, fontweight='bold')

        true_vals = [t['true_value'] for t in successful_tests]
        pred_vals = [t['predicted'] for t in successful_tests]
        rel_errors = [t['rel_error'] for t in successful_tests]
        abs_errors = [t['abs_error'] for t in successful_tests]
        times = [t['processing_time'] for t in successful_tests]

        # 1. True vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(true_vals, pred_vals, alpha=0.7, s=50)
        min_val, max_val = min(true_vals + pred_vals), max(true_vals + pred_vals)
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('True Value')
        ax1.set_ylabel('Predicted Value')
        ax1.set_title('True vs Predicted Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Relative Error Distribution
        ax2 = axes[0, 1]
        ax2.hist(rel_errors, bins=min(15, len(rel_errors)), alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Paper Target (2%)')
        ax2.axvline(5.0, color='orange', linestyle='--', linewidth=2, label='Good (5%)')
        ax2.axvline(np.mean(rel_errors), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(rel_errors):.1f}%')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Count')
        ax2.set_title('Relative Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Processing Time
        ax3 = axes[0, 2]
        ax3.hist(times, bins=min(10, len(times)), alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(times), color='red', linestyle='--',
                    label=f'Mean: {np.mean(times):.0f}ms')
        ax3.set_xlabel('Processing Time (ms)')
        ax3.set_ylabel('Count')
        ax3.set_title('Processing Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Error vs True Value
        ax4 = axes[1, 0]
        ax4.scatter(true_vals, abs_errors, alpha=0.7, s=50, color='red')
        ax4.set_xlabel('True Value')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Error vs True Value')
        ax4.grid(True, alpha=0.3)

        # 5. Accuracy Pie Chart
        ax5 = axes[1, 1]
        excellent = sum(1 for e in rel_errors if e <= 2.0)
        good = sum(1 for e in rel_errors if 2.0 < e <= 5.0)
        ok = sum(1 for e in rel_errors if 5.0 < e <= 10.0)
        poor = sum(1 for e in rel_errors if e > 10.0)

        labels = ['Excellent\n(≤2%)', 'Good\n(2-5%)', 'OK\n(5-10%)', 'Poor\n(>10%)']
        sizes = [excellent, good, ok, poor]
        colors = ['green', 'yellow', 'orange', 'red']

        # Only include non-zero categories
        non_zero_labels = [labels[i] for i, size in enumerate(sizes) if size > 0]
        non_zero_sizes = [size for size in sizes if size > 0]
        non_zero_colors = [colors[i] for i, size in enumerate(sizes) if size > 0]

        ax5.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%')
        ax5.set_title('Accuracy Distribution')

        # 6. System Performance Summary
        ax6 = axes[1, 2]
        ax6.axis('off')

        summary_text = f"""SYSTEM PERFORMANCE SUMMARY

Total Tests: {len(successful_tests)}
Success Rate: {len(successful_tests)/(len(successful_tests))*100:.1f}%

Mean Relative Error: {np.mean(rel_errors):.2f}%
Paper Target (≤2%): {"✅ PASS" if np.mean(rel_errors) <= 2.0 else "❌ FAIL"}

Processing Speed: {np.mean(times):.1f}ms avg
Excellent Accuracy: {excellent/len(successful_tests)*100:.1f}%

System Status: {"READY" if np.mean(rel_errors) <= 2.0 else "NEEDS WORK"}
"""

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        plt.savefig('still2_system_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("📊 Detailed charts saved as: still2_system_test_results.png")


def main():
    """Run the Still2 system test with interactive calibration"""
    print("🚀 STILL2.PY CAMERA-BASED GAUGE READER SYSTEM TEST")
    print("="*70)
    print("📋 Based on paper: 'Under pressure: learning-based analog gauge reading'")
    print("🎯 Target Performance: ≤2% relative reading error")
    print("🔧 Interactive calibration mode - calibrate once, test all images")
    print("="*70)

    # Configuration
    gauges_dir = "gauges"
    detection_model = "gauge_reader_web/models/gauge_detection_model.pt"
    keypoint_model = "gauge_reader_web/models/keypoint_model.pt"
    segmentation_model = "gauge_reader_web/models/needle_segmentation_model.pt"

    # Initialize tester
    tester = Still2SystemTester(
        gauges_dir=gauges_dir,
        detection_model=detection_model,
        keypoint_model=keypoint_model,
        segmentation_model=segmentation_model
    )

    # Run tests
    successful_tests, failed_tests = tester.run_full_test()

    if successful_tests:
        print(f"\n✅ System test completed!")
        print(f"📊 Results: {len(successful_tests)} successful, {len(failed_tests)} failed")

        # Quick summary
        rel_errors = [t['rel_error'] for t in successful_tests]
        mean_error = np.mean(rel_errors)
        excellent_count = sum(1 for e in rel_errors if e <= 2.0)

        print(f"🎯 Mean Error: {mean_error:.2f}%")
        print(f"🏆 Paper Standard (≤2%): {excellent_count}/{len(successful_tests)} ({excellent_count/len(successful_tests)*100:.1f}%)")

        if mean_error <= 2.0:
            print("🎉 EXCELLENT: System meets paper target!")
        elif mean_error <= 5.0:
            print("🟡 GOOD: System performs well, minor improvements possible")
        else:
            print("⚠️ NEEDS IMPROVEMENT: System requires optimization")
    else:
        print("❌ No successful tests - system needs debugging")


if __name__ == "__main__":
    main()