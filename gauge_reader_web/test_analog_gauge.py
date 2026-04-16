import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import argparse

# Import pipeline function từ pipeline.py
from pipeline import process_image


class AnalogGaugeTester:
    def __init__(self, gauges_dir, detection_model, keypoint_model, segmentation_model):
        self.gauges_dir = gauges_dir
        self.detection_model = detection_model
        self.keypoint_model = keypoint_model
        self.segmentation_model = segmentation_model

        self.predictions = []
        self.ground_truths = []
        self.inference_times = []
        self.success_readings = []
        self.failed_images = []

        # Disable logging for cleaner output
        logging.getLogger().setLevel(logging.ERROR)

    def extract_true_value_from_filename(self, filename):
        """Trích xuất giá trị thật từ tên file"""
        try:
            # Loại bỏ extension
            name = os.path.splitext(filename)[0]

            # Trích xuất giá trị từ pattern gauge_X hoặc gauge_X-Y
            if name.startswith('gauge_'):
                value_part = name[6:]  # Bỏ "gauge_"

                # Xử lý trường hợp âm: gauge_-5 -> -5
                if value_part.startswith('-'):
                    return float(value_part)

                # Xử lý trường hợp có dấu gạch ngang: gauge_0-5 -> 0.5
                if '-' in value_part:
                    parts = value_part.split('-')
                    if len(parts) == 2:
                        integer_part = int(parts[0])
                        decimal_part = int(parts[1])
                        return float(f"{integer_part}.{decimal_part}")

                # Trường hợp bình thường: gauge_5 -> 5.0
                return float(value_part)

        except ValueError as e:
            print(f"Warning: Cannot extract value from {filename}: {e}")

        return None

    def run_test(self):
        """Chạy test trên toàn bộ gauges"""
        image_files = glob.glob(os.path.join(self.gauges_dir, "*.png")) + \
                      glob.glob(os.path.join(self.gauges_dir, "*.jpg"))

        if not image_files:
            print(f"❌ Không tìm thấy ảnh nào trong {self.gauges_dir}")
            return

        print(f"🔍 Tìm thấy {len(image_files)} ảnh gauge")

        success_count = 0

        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            true_value = self.extract_true_value_from_filename(filename)

            if true_value is None:
                print(f"⚠️  Skip {filename} - không thể trích xuất giá trị")
                continue

            print(f"\n--- Test {i + 1}/{len(image_files)}: {filename} ---")
            print(f"Expected value: {true_value}")

            # Chạy pipeline
            try:
                start_time = time.time()
                result = process_image(
                    image=img_path,
                    detection_model_path=self.detection_model,
                    key_point_model_path=self.keypoint_model,
                    segmentation_model_path=self.segmentation_model,
                    run_path="/tmp/gauge_test",  # Temp path
                    debug=False,
                    eval_mode=False,
                    image_is_raw=False
                )
                end_time = time.time()

                inference_time = (end_time - start_time) * 1000
                predicted_value = result['value']
                unit = result.get('unit', 'unknown')

                print(f"Predicted value: {predicted_value:.3f} {unit}")
                print(f"Inference time: {inference_time:.1f}ms")

                # Tính error
                error = abs(predicted_value - true_value)
                relative_error = (error / abs(true_value)) * 100 if true_value != 0 else error * 100

                print(f"Absolute error: {error:.3f}")
                print(f"Relative error: {relative_error:.2f}%")

                # Lưu kết quả
                self.predictions.append(predicted_value)
                self.ground_truths.append(true_value)
                self.inference_times.append(inference_time)
                self.success_readings.append({
                    'filename': filename,
                    'true_value': true_value,
                    'predicted_value': predicted_value,
                    'error': error,
                    'relative_error': relative_error,
                    'unit': unit
                })

                success_count += 1

            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                self.failed_images.append({
                    'filename': filename,
                    'true_value': true_value,
                    'error': str(e)
                })

        print(f"\n✅ Successfully processed {success_count}/{len(image_files)} images")
        return success_count > 0

    def calculate_metrics(self, relative_error_threshold=2.0):
        """Tính các metrics theo paper (relative error threshold = 2%)"""
        if len(self.predictions) == 0:
            print("❌ No successful predictions to evaluate!")
            return None

        pred = np.array(self.predictions)
        gt = np.array(self.ground_truths)
        times = np.array(self.inference_times)

        # Absolute metrics
        mae = mean_absolute_error(gt, pred)
        mse = mean_squared_error(gt, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(gt, pred)

        # Relative error metrics (như trong paper)
        relative_errors = []
        for i in range(len(pred)):
            if gt[i] != 0:
                rel_err = abs(pred[i] - gt[i]) / abs(gt[i]) * 100
            else:
                rel_err = abs(pred[i] - gt[i]) * 100
            relative_errors.append(rel_err)

        relative_errors = np.array(relative_errors)
        mean_relative_error = np.mean(relative_errors)

        # Accuracy based on relative error threshold
        accurate_predictions = relative_errors <= relative_error_threshold
        accuracy = np.mean(accurate_predictions) * 100

        # Performance
        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0

        # Success rate
        total_images = len(self.predictions) + len(self.failed_images)
        success_rate = len(self.predictions) / total_images * 100 if total_images > 0 else 0

        # Outliers (>10% relative error)
        outliers = np.sum(relative_errors > 10)

        print("\n" + "=" * 70)
        print("📊 ANALOG GAUGE READING TEST RESULTS")
        print("=" * 70)
        print("ACCURACY METRICS:")
        print(f"  Mean Absolute Error (MAE):    {mae:.4f}")
        print(f"  Root Mean Square Error:       {rmse:.4f}")
        print(f"  R² Score:                     {r2:.4f}")
        print(f"  Mean Relative Error:          {mean_relative_error:.2f}%")
        print(f"  Accuracy (≤{relative_error_threshold}% error):      {accuracy:.1f}%")
        print("-" * 70)
        print("PIPELINE PERFORMANCE:")
        print(f"  Success Rate:                 {success_rate:.1f}%")
        print(f"  Successful readings:          {len(self.predictions)}")
        print(f"  Failed readings:              {len(self.failed_images)}")
        print(f"  Average inference time:       {avg_time:.1f}ms")
        print(f"  FPS:                          {fps:.1f}")
        print("-" * 70)
        print("ERROR ANALYSIS:")
        print(f"  Outliers (>10% error):        {outliers} ({outliers / len(pred) * 100:.1f}%)")
        print(f"  Best prediction error:        {np.min(relative_errors):.2f}%")
        print(f"  Worst prediction error:       {np.max(relative_errors):.2f}%")
        print("=" * 70)

        # Paper comparison
        print("📝 COMPARISON WITH PAPER RESULTS:")
        if mean_relative_error <= 2.0:
            print(f"  ✅ Mean relative error {mean_relative_error:.2f}% ≤ 2% (Paper target)")
        else:
            print(f"  ❌ Mean relative error {mean_relative_error:.2f}% > 2% (Paper target)")

        print("=" * 70)

        return {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'mean_relative_error': mean_relative_error,
            'accuracy': accuracy,
            'success_rate': success_rate,
            'avg_time': avg_time, 'fps': fps,
            'outliers': outliers
        }

    def plot_results(self):
        """Vẽ biểu đồ kết quả"""
        if len(self.predictions) == 0:
            print("❌ No data to plot!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analog Gauge Reading Test Results', fontsize=16, fontweight='bold')

        pred = np.array(self.predictions)
        gt = np.array(self.ground_truths)

        # 1. Prediction vs Ground Truth
        ax1 = axes[0, 0]
        ax1.scatter(gt, pred, alpha=0.6, s=50, color='blue')
        min_val, max_val = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Ground Truth')
        ax1.set_ylabel('Predictions')
        ax1.set_title('Prediction vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Relative Error Distribution
        ax2 = axes[0, 1]
        relative_errors = []
        for i in range(len(pred)):
            if gt[i] != 0:
                rel_err = abs(pred[i] - gt[i]) / abs(gt[i]) * 100
            else:
                rel_err = abs(pred[i] - gt[i]) * 100
            relative_errors.append(rel_err)

        ax2.hist(relative_errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(2.0, color='red', linestyle='--', linewidth=2, label='2% (Paper target)')
        ax2.axvline(np.mean(relative_errors), color='green', linestyle='--',
                    label=f'Mean: {np.mean(relative_errors):.1f}%')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Relative Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Absolute Error vs True Value
        ax3 = axes[1, 0]
        abs_errors = np.abs(pred - gt)
        ax3.scatter(gt, abs_errors, alpha=0.6, s=50, color='red')
        ax3.set_xlabel('Ground Truth Value')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Absolute Error vs True Value')
        ax3.grid(True, alpha=0.3)

        # 4. Inference Time Distribution
        ax4 = axes[1, 1]
        ax4.hist(self.inference_times, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(np.mean(self.inference_times), color='red', linestyle='--',
                    label=f'Mean: {np.mean(self.inference_times):.1f}ms')
        ax4.set_xlabel('Inference Time (ms)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Inference Time Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analog_gauge_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("📊 Charts saved as 'analog_gauge_test_results.png'")

    def show_detailed_results(self, top_n=5):
        """Hiển thị kết quả chi tiết"""
        if not self.success_readings:
            print("❌ No successful readings to show!")
            return

        print(f"\n🎯 TOP {top_n} BEST PREDICTIONS:")
        sorted_success = sorted(self.success_readings, key=lambda x: x['relative_error'])

        for i, result in enumerate(sorted_success[:top_n]):
            print(f"{i + 1}. {result['filename']}")
            print(f"   True: {result['true_value']:.3f}, Pred: {result['predicted_value']:.3f}")
            print(f"   Error: {result['relative_error']:.2f}% | Unit: {result['unit']}")

        print(f"\n❌ TOP {top_n} WORST PREDICTIONS:")
        for i, result in enumerate(sorted_success[-top_n:]):
            print(f"{i + 1}. {result['filename']}")
            print(f"   True: {result['true_value']:.3f}, Pred: {result['predicted_value']:.3f}")
            print(f"   Error: {result['relative_error']:.2f}% | Unit: {result['unit']}")

        if self.failed_images:
            print(f"\n💥 FAILED IMAGES ({len(self.failed_images)}):")
            for failure in self.failed_images:
                print(f"- {failure['filename']} (expected: {failure['true_value']:.3f})")
                print(f"  Error: {failure['error']}")


def main():
    parser = argparse.ArgumentParser(description='Test Analog Gauge Reading Pipeline')
    parser.add_argument('--gauges_dir', type=str, default='gauges',
                        help='Directory containing gauge images')
    parser.add_argument('--detection_model', type=str,
                        default='models/gauge_detection_model.pt',
                        help='Path to gauge detection model')
    parser.add_argument('--keypoint_model', type=str,
                        default='models/keypoint_model.pt',
                        help='Path to keypoint detection model')
    parser.add_argument('--segmentation_model', type=str,
                        default='models/needle_segmentation_model.pt',
                        help='Path to needle segmentation model')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Relative error threshold for accuracy calculation (%)')

    args = parser.parse_args()

    # Kiểm tra files tồn tại
    paths_to_check = [
        (args.gauges_dir, "gauges directory"),
        (args.detection_model, "detection model"),
        (args.keypoint_model, "keypoint model"),
        (args.segmentation_model, "segmentation model")
    ]

    for path, name in paths_to_check:
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            return

    print("🚀 ANALOG GAUGE READING TESTER")
    print(f"📁 Gauges directory: {args.gauges_dir}")
    print(f"🎯 Error threshold: {args.threshold}%")
    print("=" * 50)

    # Tạo tester và chạy
    tester = AnalogGaugeTester(
        gauges_dir=args.gauges_dir,
        detection_model=args.detection_model,
        keypoint_model=args.keypoint_model,
        segmentation_model=args.segmentation_model
    )

    # Chạy test
    success = tester.run_test()

    if success:
        # Tính metrics
        metrics = tester.calculate_metrics(relative_error_threshold=args.threshold)

        # Hiển thị kết quả chi tiết
        tester.show_detailed_results()

        # Vẽ biểu đồ
        tester.plot_results()

    else:
        print("❌ Test failed - no successful predictions!")


if __name__ == "__main__":
    main()