import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Import các components cần thiết từ pipeline
from gauge_reader_web.gauge_detection.detection_inference import detection_gauge_face
from gauge_reader_web.key_point_detection.key_point_inference import KeyPointInference, detect_key_points
from gauge_reader_web.geometry.ellipse import (
    fit_ellipse, cart_to_pol, get_point_from_angle, get_polar_angle,
    get_theta_middle, get_line_ellipse_point
)
from gauge_reader_web.segmentation.segmenation_inference import segment_gauge_needle, get_fitted_line, get_start_end_line, cut_off_line
from gauge_reader_web.angle_reading_fit.angle_converter import AngleConverter


class CalibratedGaugeTester:
    def __init__(self, gauges_dir, detection_model, keypoint_model, segmentation_model,
                 gauge_min=0.0, gauge_max=15.0, unit="psi"):
        self.gauges_dir = gauges_dir
        self.detection_model = detection_model
        self.keypoint_model = keypoint_model
        self.segmentation_model = segmentation_model
        self.gauge_min = gauge_min
        self.gauge_max = gauge_max
        self.gauge_range = gauge_max - gauge_min
        self.unit = unit

        self.predictions = []
        self.ground_truths = []
        self.inference_times = []
        self.success_readings = []
        self.failed_images = []

        self.RESOLUTION = (448, 448)

    def extract_true_value(self, filename):
        """Trích xuất giá trị từ tên file gauge"""
        try:
            name = os.path.splitext(filename)[0]
            if name.startswith('gauge_'):
                value_part = name[6:]  # Bỏ "gauge_"

                if value_part.startswith('-'):
                    return float(value_part)

                if '-' in value_part:
                    parts = value_part.split('-')
                    if len(parts) == 2:
                        return float(f"{parts[0]}.{parts[1]}")

                return float(value_part)
        except:
            pass
        return None

    def crop_image(self, img, box):
        """Crop image theo bounding box"""
        img = np.copy(img)
        cropped_img = img[box[1]:box[3], box[0]:box[2], :]

        height = int(box[3] - box[1])
        width = int(box[2] - box[0])

        # Padding để tạo hình vuông
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

    def process_single_gauge(self, img_path):
        """Xử lý 1 ảnh gauge với calibration mode"""
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            image = np.asarray(image)

            # 1. Gauge Detection
            box, _ = detection_gauge_face(image, self.detection_model)
            cropped_img = self.crop_image(image, box)
            cropped_resized_img = cv2.resize(
                cropped_img, dsize=self.RESOLUTION, interpolation=cv2.INTER_CUBIC
            )

            # 2. Key Point Detection
            key_point_inferencer = KeyPointInference(self.keypoint_model)
            heatmaps = key_point_inferencer.predict_heatmaps(cropped_resized_img)
            key_point_list = detect_key_points(heatmaps)

            key_points = key_point_list[1]  # Middle notches
            start_point = key_point_list[0]  # Start point
            end_point = key_point_list[2]  # End point

            if len(key_points) < 3:
                raise Exception("Not enough keypoints detected")

            # 3. Ellipse Fitting
            coeffs = fit_ellipse(key_points[:, 0], key_points[:, 1])
            ellipse_params = cart_to_pol(coeffs)

            # 4. Calculate zero point (bottom middle hoặc giữa start-end)
            if start_point.shape == (1, 2) and end_point.shape == (1, 2):
                theta_start = get_polar_angle(start_point.flatten(), ellipse_params)
                theta_end = get_polar_angle(end_point.flatten(), ellipse_params)
                theta_zero = get_theta_middle(theta_start, theta_end)
            else:
                bottom_middle = np.array((self.RESOLUTION[0] / 2, self.RESOLUTION[1]))
                theta_zero = get_polar_angle(bottom_middle, ellipse_params)

            # 5. Needle Segmentation
            needle_mask_x, needle_mask_y = segment_gauge_needle(
                cropped_resized_img, self.segmentation_model
            )
            needle_line_coeffs, _ = get_fitted_line(needle_mask_x, needle_mask_y)
            needle_line_start_x, needle_line_end_x = get_start_end_line(needle_mask_x)
            needle_line_start_y, needle_line_end_y = get_start_end_line(needle_mask_y)
            needle_line_start_x, needle_line_end_x = cut_off_line(
                [needle_line_start_x, needle_line_end_x],
                needle_line_start_y, needle_line_end_y, needle_line_coeffs
            )

            # 6. Find needle intersection với ellipse
            point_needle_ellipse = get_line_ellipse_point(
                needle_line_coeffs, (needle_line_start_x, needle_line_end_x), ellipse_params
            )

            if point_needle_ellipse.shape[0] == 0:
                raise Exception("Needle line and ellipse do not intersect")

            # 7. CALIBRATED READING CALCULATION
            needle_angle = get_polar_angle(point_needle_ellipse, ellipse_params)
            angle_converter = AngleConverter(theta_zero)
            needle_angle_conv = angle_converter.convert_angle(needle_angle)

            # Giả định đồng hồ có full range từ start đến end point
            if start_point.shape == (1, 2) and end_point.shape == (1, 2):
                theta_start_conv = angle_converter.convert_angle(theta_start)
                theta_end_conv = angle_converter.convert_angle(theta_end)

                # Linear mapping: angle -> reading
                angle_range = theta_end_conv - theta_start_conv
                if angle_range != 0:
                    # Tính tỷ lệ vị trí kim trên scale
                    position_ratio = (needle_angle_conv - theta_start_conv) / angle_range
                    # Map vào gauge range
                    reading = self.gauge_min + position_ratio * self.gauge_range
                else:
                    reading = self.gauge_min
            else:
                # Fallback: sử dụng toàn bộ ellipse (0-2π)
                normalized_angle = (needle_angle_conv % (2 * np.pi)) / (2 * np.pi)
                reading = self.gauge_min + normalized_angle * self.gauge_range

            return reading

        except Exception as e:
            raise Exception(f"Processing failed: {str(e)}")

    def run_test(self):
        """Chạy test với calibration mode"""
        image_files = glob.glob(os.path.join(self.gauges_dir, "*.png")) + \
                      glob.glob(os.path.join(self.gauges_dir, "*.jpg"))

        if not image_files:
            print(f"❌ Không tìm thấy ảnh nào trong {self.gauges_dir}")
            return

        print(f"🔍 Tìm thấy {len(image_files)} ảnh gauge")
        print(f"🎯 Calibration: {self.gauge_min}-{self.gauge_max} {self.unit}")
        print("🚀 Bắt đầu test (CALIBRATED MODE - No OCR)...")

        # Tạo tmp folder
        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        results = []
        failed = []

        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            true_value = self.extract_true_value(filename)

            if true_value is None:
                print(f"⚠️  Skip {filename} - không đọc được giá trị")
                continue

            print(f"\n[{i + 1}/{len(image_files)}] Testing {filename}")
            print(f"Expected: {true_value} {self.unit}")

            try:
                start_time = time.time()
                predicted = self.process_single_gauge(img_path)
                end_time = time.time()

                inference_time = (end_time - start_time) * 1000

                # Tính lỗi
                abs_error = abs(predicted - true_value)
                rel_error = (abs_error / abs(true_value)) * 100 if true_value != 0 else abs_error * 100

                print(f"Predicted: {predicted:.3f} {self.unit}")
                print(f"Error: {abs_error:.3f} ({rel_error:.1f}%)")
                print(f"Time: {inference_time:.0f}ms")

                results.append({
                    'filename': filename,
                    'true': true_value,
                    'pred': predicted,
                    'abs_error': abs_error,
                    'rel_error': rel_error,
                    'time': inference_time
                })

                # Status
                if rel_error <= 2.0:
                    print("✅ EXCELLENT (≤2% error)")
                elif rel_error <= 5.0:
                    print("🟡 GOOD (≤5% error)")
                elif rel_error <= 10.0:
                    print("⚠️  OK (≤10% error)")
                else:
                    print("❌ BAD (>10% error)")

            except Exception as e:
                print(f"💥 FAILED: {str(e)}")
                failed.append({'filename': filename, 'true': true_value, 'error': str(e)})

        # Tổng kết
        if not results:
            print("\n❌ Không có kết quả nào thành công!")
            return

        self.show_results(results, failed)
        self.plot_results(results)

    def show_results(self, results, failed):
        """Hiển thị kết quả tổng kết"""
        print(f"\n" + "=" * 70)
        print("📊 KẾT QUẢ CALIBRATED GAUGE TEST")
        print("=" * 70)

        # Tính metrics
        rel_errors = [r['rel_error'] for r in results]
        abs_errors = [r['abs_error'] for r in results]
        times = [r['time'] for r in results]

        mean_rel_error = np.mean(rel_errors)
        mean_abs_error = np.mean(abs_errors)
        mean_time = np.mean(times)

        # Accuracy theo ngưỡng
        excellent_count = sum(1 for e in rel_errors if e <= 2.0)
        good_count = sum(1 for e in rel_errors if e <= 5.0)
        ok_count = sum(1 for e in rel_errors if e <= 10.0)

        success_rate = len(results) / (len(results) + len(failed)) * 100

        print(f"📈 PERFORMANCE METRICS:")
        print(f"  Success Rate:           {success_rate:.1f}% ({len(results)}/{len(results) + len(failed)})")
        print(f"  Mean Relative Error:    {mean_rel_error:.2f}%")
        print(f"  Mean Absolute Error:    {mean_abs_error:.3f} {self.unit}")
        print(f"  Average Time:           {mean_time:.0f}ms")
        print("-" * 70)
        print(f"🎯 ACCURACY BREAKDOWN:")
        print(
            f"  Excellent (≤2% error):  {excellent_count / len(results) * 100:.1f}% ({excellent_count}/{len(results)})")
        print(f"  Good (≤5% error):       {good_count / len(results) * 100:.1f}% ({good_count}/{len(results)})")
        print(f"  OK (≤10% error):        {ok_count / len(results) * 100:.1f}% ({ok_count}/{len(results)})")
        print("-" * 70)

        # So sánh với paper target
        if mean_rel_error <= 2.0:
            print("🎯 ✅ ĐẠT TARGET PAPER: ≤2% relative error")
        elif mean_rel_error <= 5.0:
            print("🎯 🟡 GẦN TARGET PAPER: ≤5% relative error")
        else:
            print("🎯 ❌ CHƯA ĐẠT TARGET PAPER: >5% relative error")

        # Best/Worst cases
        best_result = min(results, key=lambda x: x['rel_error'])
        worst_result = max(results, key=lambda x: x['rel_error'])

        print(f"\n🏆 BEST: {best_result['filename']} - {best_result['rel_error']:.1f}% error")
        print(f"💔 WORST: {worst_result['filename']} - {worst_result['rel_error']:.1f}% error")

        # Failed cases
        if failed:
            print(f"\n💥 FAILED CASES ({len(failed)}):")
            for f in failed[:5]:  # Show first 5
                print(f"  - {f['filename']}: {f['error']}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        print("=" * 70)

    def plot_results(self, results):
        """Vẽ biểu đồ kết quả"""
        if not results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Calibrated Gauge Test Results ({self.gauge_min}-{self.gauge_max} {self.unit})',
                     fontsize=16, fontweight='bold')

        true_vals = [r['true'] for r in results]
        pred_vals = [r['pred'] for r in results]
        rel_errors = [r['rel_error'] for r in results]
        times = [r['time'] for r in results]

        # 1. True vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(true_vals, pred_vals, alpha=0.7, s=50)
        min_val, max_val = min(true_vals + pred_vals), max(true_vals + pred_vals)
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel(f'True Value ({self.unit})')
        ax1.set_ylabel(f'Predicted Value ({self.unit})')
        ax1.set_title('True vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Relative Error Distribution
        ax2 = axes[0, 1]
        ax2.hist(rel_errors, bins=min(15, len(rel_errors)), alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(2.0, color='red', linestyle='--', linewidth=2, label='2% target')
        ax2.axvline(5.0, color='orange', linestyle='--', linewidth=2, label='5% target')
        ax2.axvline(np.mean(rel_errors), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(rel_errors):.1f}%')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Error vs True Value
        ax3 = axes[1, 0]
        abs_errors = [r['abs_error'] for r in results]
        ax3.scatter(true_vals, abs_errors, alpha=0.7, s=50, color='red')
        ax3.set_xlabel(f'True Value ({self.unit})')
        ax3.set_ylabel(f'Absolute Error ({self.unit})')
        ax3.set_title('Error vs True Value')
        ax3.grid(True, alpha=0.3)

        # 4. Inference Time
        ax4 = axes[1, 1]
        ax4.hist(times, bins=min(10, len(times)), alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(np.mean(times), color='red', linestyle='--',
                    label=f'Mean: {np.mean(times):.0f}ms')
        ax4.set_xlabel('Inference Time (ms)')
        ax4.set_ylabel('Count')
        ax4.set_title('Processing Time Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'calibrated_gauge_test_{self.gauge_min}_{self.gauge_max}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Charts saved as: calibrated_gauge_test_{self.gauge_min}_{self.gauge_max}.png")


def main():
    # Cấu hình
    gauges_dir = "data/gauges"
    detection_model = "gauge_reader_web/models/gauge_detection_model.pt"
    keypoint_model = "gauge_reader_web/models/keypoint_model.pt"
    segmentation_model = "gauge_reader_web/models/needle_segmentation_model.pt"

    # Thông số calibration
    gauge_min = 0.0  # Min value
    gauge_max = 15.0  # Max value
    unit = "psi"  # Unit

    # Kiểm tra files
    for path, name in [(gauges_dir, "gauges folder"),
                       (detection_model, "detection model"),
                       (keypoint_model, "keypoint model"),
                       (segmentation_model, "segmentation model")]:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy {name}: {path}")
            return

    print("🎯 CALIBRATED ANALOG GAUGE TESTER")
    print(f"📏 Range: {gauge_min}-{gauge_max} {unit}")
    print("🔧 Mode: No OCR - Pure geometric calibration")
    print("=" * 60)

    # Tạo tester và chạy
    tester = CalibratedGaugeTester(
        gauges_dir=gauges_dir,
        detection_model=detection_model,
        keypoint_model=keypoint_model,
        segmentation_model=segmentation_model,
        gauge_min=gauge_min,
        gauge_max=gauge_max,
        unit=unit
    )

    tester.run_test()


if __name__ == "__main__":
    main()