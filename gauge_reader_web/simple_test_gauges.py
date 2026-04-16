import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

# Import pipeline function
from pipeline import process_image


def extract_true_value(filename):
    """Trích xuất giá trị từ tên file gauge"""
    try:
        name = os.path.splitext(filename)[0]
        if name.startswith('gauge_'):
            value_part = name[6:]  # Bỏ "gauge_"

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


def test_gauges():
    """Test đơn giản cho folder gauges"""

    # Đường dẫn cố định
    gauges_dir = "gauges"
    detection_model = "models/gauge_detection_model.pt"
    keypoint_model = "models/keypoint_model.pt"
    segmentation_model = "models/needle_segmentation_model.pt"

    # Kiểm tra files
    for path, name in [(gauges_dir, "gauges folder"),
                       (detection_model, "detection model"),
                       (keypoint_model, "keypoint model"),
                       (segmentation_model, "segmentation model")]:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy {name}: {path}")
            return

    # Tạo tmp folder nếu chưa có
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        print(f"📁 Created tmp folder: {tmp_dir}")

    # Tìm ảnh
    image_files = glob.glob(os.path.join(gauges_dir, "*.png")) + \
                  glob.glob(os.path.join(gauges_dir, "*.jpg"))

    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong {gauges_dir}")
        return

    print(f"🔍 Tìm thấy {len(image_files)} ảnh gauge")
    print("🚀 Bắt đầu test...")

    results = []
    failed = []

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        true_value = extract_true_value(filename)

        if true_value is None:
            print(f"⚠️  Skip {filename} - không đọc được giá trị")
            continue

        print(f"\n[{i + 1}/{len(image_files)}] Testing {filename}")
        print(f"Expected: {true_value}")

        try:
            # Tạo subfolder riêng cho mỗi test
            test_dir = os.path.join(tmp_dir, f"test_{i}")

            start_time = time.time()
            result = process_image(
                image=img_path,
                detection_model_path=detection_model,
                key_point_model_path=keypoint_model,
                segmentation_model_path=segmentation_model,
                run_path=test_dir,
                debug=False,
                eval_mode=False,
                image_is_raw=False
            )
            end_time = time.time()

            predicted = result['value']
            unit = result.get('unit', '')
            inference_time = (end_time - start_time) * 1000

            # Tính lỗi
            abs_error = abs(predicted - true_value)
            rel_error = (abs_error / abs(true_value)) * 100 if true_value != 0 else abs_error * 100

            print(f"Predicted: {predicted:.3f} {unit}")
            print(f"Error: {abs_error:.3f} ({rel_error:.1f}%)")
            print(f"Time: {inference_time:.0f}ms")

            results.append({
                'filename': filename,
                'true': true_value,
                'pred': predicted,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'time': inference_time,
                'unit': unit
            })

            # Status
            if rel_error <= 2.0:
                print("✅ GOOD (≤2% error)")
            elif rel_error <= 5.0:
                print("⚠️  OK (≤5% error)")
            else:
                print("❌ BAD (>5% error)")

        except Exception as e:
            print(f"💥 FAILED: {str(e)}")
            failed.append({'filename': filename, 'true': true_value, 'error': str(e)})

    # Tổng kết
    if not results:
        print("\n❌ Không có kết quả nào thành công!")
        return

    print(f"\n" + "=" * 60)
    print("📊 KẾT QUẢ TỔNG KẾT")
    print("=" * 60)

    # Tính metrics
    rel_errors = [r['rel_error'] for r in results]
    abs_errors = [r['abs_error'] for r in results]
    times = [r['time'] for r in results]

    mean_rel_error = np.mean(rel_errors)
    mean_abs_error = np.mean(abs_errors)
    mean_time = np.mean(times)

    # Accuracy theo ngưỡng
    good_count = sum(1 for e in rel_errors if e <= 2.0)
    ok_count = sum(1 for e in rel_errors if e <= 5.0)

    success_rate = len(results) / (len(results) + len(failed)) * 100
    accuracy_2pct = good_count / len(results) * 100
    accuracy_5pct = ok_count / len(results) * 100

    print(f"Success Rate:           {success_rate:.1f}% ({len(results)}/{len(results) + len(failed)})")
    print(f"Mean Relative Error:    {mean_rel_error:.2f}%")
    print(f"Mean Absolute Error:    {mean_abs_error:.3f}")
    print(f"Accuracy (≤2% error):   {accuracy_2pct:.1f}% ({good_count}/{len(results)})")
    print(f"Accuracy (≤5% error):   {accuracy_5pct:.1f}% ({ok_count}/{len(results)})")
    print(f"Average Time:           {mean_time:.0f}ms")
    print("-" * 60)

    # So sánh với paper
    if mean_rel_error <= 2.0:
        print("🎯 ĐẠT MỤC TIÊU PAPER: ≤2% relative error")
    else:
        print("📋 CHƯA ĐẠT MỤC TIÊU PAPER: >2% relative error")

    # Top worst cases
    worst_results = sorted(results, key=lambda x: x['rel_error'], reverse=True)[:3]
    print(f"\nTop 3 trường hợp tệ nhất:")
    for i, r in enumerate(worst_results):
        print(f"{i + 1}. {r['filename']}: {r['rel_error']:.1f}% error")

    # Failed cases
    if failed:
        print(f"\nCác trường hợp thất bại:")
        for f in failed:
            print(f"- {f['filename']}: {f['error']}")

    print("=" * 60)

    # Simple plot
    if len(results) > 1:
        plt.figure(figsize=(12, 4))

        # Relative errors
        plt.subplot(1, 2, 1)
        plt.hist(rel_errors, bins=min(10, len(rel_errors)), alpha=0.7, color='orange')
        plt.axvline(2.0, color='red', linestyle='--', label='2% target')
        plt.axvline(mean_rel_error, color='blue', linestyle='--', label=f'Mean: {mean_rel_error:.1f}%')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Count')
        plt.title('Relative Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # True vs Predicted
        plt.subplot(1, 2, 2)
        true_vals = [r['true'] for r in results]
        pred_vals = [r['pred'] for r in results]
        plt.scatter(true_vals, pred_vals, alpha=0.7)
        min_val, max_val = min(true_vals + pred_vals), max(true_vals + pred_vals)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title('True vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gauge_test_simple.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Biểu đồ lưu tại: gauge_test_simple.png")

    print(f"\n📁 Temp files in: {tmp_dir}/ (you can delete this folder)")


if __name__ == "__main__":
    test_gauges()