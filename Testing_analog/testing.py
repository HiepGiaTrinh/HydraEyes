#!/usr/bin/env python3
"""
Gauge Reading Benchmark Evaluator
Reads test results from Excel and generates comprehensive benchmark analysis
Based on "Under pressure: learning-based analog gauge reading in the wild" paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from datetime import datetime


class GaugeBenchmarkEvaluator:
    """Comprehensive benchmark evaluator for gauge reading systems"""

    def __init__(self, excel_file):
        """Initialize evaluator with Excel data file"""
        self.excel_file = excel_file
        self.data = None
        self.results = {}

        # Set style for professional plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Fix font rendering issues
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10

    def load_data(self):
        """Load and validate data from Excel file"""
        try:
            # Try different sheet names
            sheet_names = ['results', 'data', 'test_results', 'Sheet1', 0]

            for sheet in sheet_names:
                try:
                    self.data = pd.read_excel(self.excel_file, sheet_name=sheet)
                    print(f"✅ Loaded data from sheet: {sheet}")
                    break
                except:
                    continue

            if self.data is None:
                # If no named sheets work, try the first sheet
                self.data = pd.read_excel(self.excel_file)
                print("✅ Loaded data from default sheet")

            print(f"📊 Data shape: {self.data.shape}")
            print(f"📋 Columns: {list(self.data.columns)}")

            # Auto-detect column names (flexible naming)
            self.detect_columns()

            return True

        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return False

    def detect_columns(self):
        """Auto-detect column names with flexible mapping"""
        cols = [col.lower().strip() for col in self.data.columns]

        # Mapping patterns
        true_patterns = ['true', 'actual', 'ground_truth', 'target', 'reference']
        pred_patterns = ['pred', 'predicted', 'output', 'result', 'reading']
        time_patterns = ['time', 'duration', 'processing', 'inference', 'latency']

        self.col_map = {}

        # Find columns
        for col, orig_col in zip(cols, self.data.columns):
            if any(pattern in col for pattern in true_patterns):
                self.col_map['true'] = orig_col
            elif any(pattern in col for pattern in pred_patterns):
                self.col_map['predicted'] = orig_col
            elif any(pattern in col for pattern in time_patterns):
                self.col_map['time'] = orig_col

        # If auto-detection fails, use first few columns
        if 'true' not in self.col_map:
            self.col_map['true'] = self.data.columns[0]
        if 'predicted' not in self.col_map:
            self.col_map['predicted'] = self.data.columns[1]
        if 'time' not in self.col_map and len(self.data.columns) > 2:
            self.col_map['time'] = self.data.columns[2]

        print(f"🎯 Column mapping: {self.col_map}")

    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        try:
            # Extract data
            true_values = self.data[self.col_map['true']].dropna()
            pred_values = self.data[self.col_map['predicted']].dropna()

            # Ensure same length
            min_len = min(len(true_values), len(pred_values))
            true_values = true_values.iloc[:min_len]
            pred_values = pred_values.iloc[:min_len]

            # Remove failed predictions (if marked as string/nan)
            valid_mask = pd.notna(true_values) & pd.notna(pred_values)

            # Try to convert to numeric, handle string failures
            try:
                pred_numeric = pd.to_numeric(pred_values, errors='coerce')
                valid_mask = valid_mask & pd.notna(pred_numeric)
                pred_values = pred_numeric[valid_mask]
                true_values = true_values[valid_mask]
            except:
                pass

            print(f"📊 Valid samples: {len(true_values)}")

            # Calculate errors
            abs_errors = np.abs(pred_values - true_values)

            # Auto-detect scale range
            scale_range = true_values.max() - true_values.min()
            print(f"📏 Scale range: {scale_range:.2f}")

            # Relative errors (% of full scale)
            rel_errors = (abs_errors / scale_range) * 100

            # Store data for plotting
            self.plot_data = {
                'true_values': true_values,
                'pred_values': pred_values,
                'abs_errors': abs_errors,
                'rel_errors': rel_errors,
                'scale_range': scale_range
            }

            # Calculate comprehensive metrics
            self.results = {
                'total_samples': len(self.data),
                'valid_samples': len(true_values),
                'failure_rate': (len(self.data) - len(true_values)) / len(self.data) * 100,

                # Accuracy metrics
                'mean_abs_error': abs_errors.mean(),
                'median_abs_error': abs_errors.median(),
                'std_abs_error': abs_errors.std(),
                'max_abs_error': abs_errors.max(),

                # Relative error metrics (key for gauge reading)
                'mean_rel_error': rel_errors.mean(),
                'median_rel_error': rel_errors.median(),
                'std_rel_error': rel_errors.std(),
                'max_rel_error': rel_errors.max(),

                # Accuracy within thresholds (paper standard)
                'accuracy_1_percent': (rel_errors <= 1.0).mean() * 100,
                'accuracy_2_percent': (rel_errors <= 2.0).mean() * 100,
                'accuracy_5_percent': (rel_errors <= 5.0).mean() * 100,
                'accuracy_10_percent': (rel_errors <= 10.0).mean() * 100,

                # R² and correlation
                'r_squared': np.corrcoef(true_values, pred_values)[0, 1] ** 2,
                'correlation': np.corrcoef(true_values, pred_values)[0, 1],

                # Scale info
                'scale_min': true_values.min(),
                'scale_max': true_values.max(),
                'scale_range': scale_range
            }

            # Processing time metrics (if available)
            if 'time' in self.col_map:
                time_data = pd.to_numeric(self.data[self.col_map['time']], errors='coerce').dropna()
                if len(time_data) > 0:
                    self.results.update({
                        'mean_processing_time': time_data.mean(),
                        'median_processing_time': time_data.median(),
                        'std_processing_time': time_data.std(),
                        'max_processing_time': time_data.max(),
                        'min_processing_time': time_data.min()
                    })
                    self.plot_data['processing_times'] = time_data

            print("✅ Metrics calculated successfully")
            return True

        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return False

    def print_results(self):
        """Print comprehensive benchmark results"""
        print("\n" + "=" * 60)
        print("🎯 GAUGE READING BENCHMARK RESULTS")
        print("=" * 60)

        # Data overview
        print(f"\n📊 DATA OVERVIEW:")
        print(f"   Total samples: {self.results['total_samples']:,}")
        print(f"   Valid samples: {self.results['valid_samples']:,}")
        print(f"   Failure rate: {self.results['failure_rate']:.1f}%")
        print(f"   Scale range: {self.results['scale_min']:.1f} - {self.results['scale_max']:.1f}")

        # Core accuracy metrics
        print(f"\n🎯 ACCURACY METRICS:")
        print(f"   Mean Relative Error: {self.results['mean_rel_error']:.2f}%")
        print(f"   Median Relative Error: {self.results['median_rel_error']:.2f}%")
        print(f"   Mean Absolute Error: {self.results['mean_abs_error']:.3f}")
        print(f"   Max Absolute Error: {self.results['max_abs_error']:.3f}")

        # Accuracy thresholds (paper benchmarks)
        print(f"\n📏 ACCURACY WITHIN THRESHOLDS:")
        print(f"   Within 1%: {self.results['accuracy_1_percent']:.1f}%")
        print(f"   Within 2%: {self.results['accuracy_2_percent']:.1f}%")  # Paper target
        print(f"   Within 5%: {self.results['accuracy_5_percent']:.1f}%")
        print(f"   Within 10%: {self.results['accuracy_10_percent']:.1f}%")

        # Statistical measures
        print(f"\n📈 STATISTICAL MEASURES:")
        print(f"   R^2 Score: {self.results['r_squared']:.4f}")
        print(f"   Correlation: {self.results['correlation']:.4f}")

        # Performance vs Paper Benchmark
        print(f"\n🏆 PAPER BENCHMARK COMPARISON:")
        paper_target = 2.0  # Paper achieves <2% relative error
        if self.results['mean_rel_error'] <= paper_target:
            print(f"   ✅ PASSED - Mean error {self.results['mean_rel_error']:.2f}% <= {paper_target}%")
        else:
            print(f"   ❌ FAILED - Mean error {self.results['mean_rel_error']:.2f}% > {paper_target}%")

        if self.results['accuracy_2_percent'] >= 90:
            print(f"   ✅ PASSED - 2% accuracy {self.results['accuracy_2_percent']:.1f}% >= 90%")
        else:
            print(f"   ❌ FAILED - 2% accuracy {self.results['accuracy_2_percent']:.1f}% < 90%")

        # Processing time (if available)
        if 'mean_processing_time' in self.results:
            print(f"\n⏱️  PROCESSING TIME:")
            print(f"   Mean: {self.results['mean_processing_time']:.0f}ms")
            print(f"   Median: {self.results['median_processing_time']:.0f}ms")
            print(f"   Max: {self.results['max_processing_time']:.0f}ms")

        print("=" * 60)

    def create_plots(self, save_dir="benchmark_plots"):
        """Create comprehensive benchmark plots"""
        try:
            # Create output directory
            Path(save_dir).mkdir(exist_ok=True)

            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))

            # Extract data
            true_vals = self.plot_data['true_values']
            pred_vals = self.plot_data['pred_values']
            abs_errors = self.plot_data['abs_errors']
            rel_errors = self.plot_data['rel_errors']

            # 1. True vs Predicted scatter plot
            ax1 = plt.subplot(2, 3, 1)
            plt.scatter(true_vals, pred_vals, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

            # Perfect prediction line
            min_val, max_val = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect', linewidth=2)

            # Best fit line
            z = np.polyfit(true_vals, pred_vals, 1)
            p = np.poly1d(z)
            plt.plot(true_vals, p(true_vals), 'b-', alpha=0.8, linewidth=2)

            plt.xlabel('True Value (psi)')
            plt.ylabel('Predicted Value (psi)')
            plt.title('True vs Predicted')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add R² annotation (fix font rendering)
            plt.text(0.05, 0.95, f'R^2 = {self.results["r_squared"]:.3f}',
                     transform=ax1.transAxes, fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

            # 2. Error distribution histogram
            ax2 = plt.subplot(2, 3, 2)
            plt.hist(rel_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(2, color='red', linestyle='--', linewidth=2, label='2% target')
            plt.axvline(5, color='orange', linestyle='--', linewidth=2, label='5% target')
            plt.axvline(rel_errors.mean(), color='black', linestyle='-', linewidth=2,
                        label=f'Mean: {rel_errors.mean():.1f}%')
            plt.xlabel('Relative Error (%)')
            plt.ylabel('Count')
            plt.title('Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 3. Error vs True Value
            ax3 = plt.subplot(2, 3, 4)
            plt.scatter(true_vals, abs_errors, alpha=0.6, s=50, color='red', edgecolors='white', linewidth=0.5)
            plt.xlabel('True Value (psi)')
            plt.ylabel('Absolute Error (psi)')
            plt.title('Error vs True Value')
            plt.grid(True, alpha=0.3)

            # 4. Processing time distribution (if available)
            if 'processing_times' in self.plot_data:
                ax4 = plt.subplot(2, 3, 5)
                times = self.plot_data['processing_times']
                plt.hist(times, bins=30, alpha=0.7, color='green', edgecolor='black')
                plt.axvline(times.mean(), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {times.mean():.0f}ms')
                plt.xlabel('Inference Time (ms)')
                plt.ylabel('Count')
                plt.title('Processing Time Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)

            # 5. Accuracy vs Threshold
            ax5 = plt.subplot(2, 3, 6)
            thresholds = [1, 2, 5, 10, 20]
            accuracies = [(rel_errors <= th).mean() * 100 for th in thresholds]

            plt.plot(thresholds, accuracies, 'o-', linewidth=2, markersize=8)
            plt.axhline(90, color='red', linestyle='--', alpha=0.7, label='90% target')
            plt.xlabel('Error Threshold (%)')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy vs Error Threshold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 6. Overall title and metrics
            ax6 = plt.subplot(2, 3, 3)
            ax6.axis('off')

            # Summary metrics text
            summary_text = f"""
BENCHMARK SUMMARY
Scale Range: {self.results['scale_min']:.1f} - {self.results['scale_max']:.1f} psi

KEY METRICS:
• Mean Rel. Error: {self.results['mean_rel_error']:.2f}%
• Accuracy within 2%: {self.results['accuracy_2_percent']:.1f}%
• Accuracy within 5%: {self.results['accuracy_5_percent']:.1f}%
• R^2 Score: {self.results['r_squared']:.3f}
• Samples: {self.results['valid_samples']:,}
• Failure Rate: {self.results['failure_rate']:.1f}%

PAPER BENCHMARK:
Target: <2% relative error
Status: {'✅ PASSED' if self.results['mean_rel_error'] <= 2.0 else '❌ FAILED'}
            """

            if 'mean_processing_time' in self.results:
                summary_text += f"\n• Avg. Time: {self.results['mean_processing_time']:.0f}ms"

            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", alpha=0.9))

            plt.suptitle(
                f'Calibrated Gauge Test Results ({self.results["scale_min"]:.1f}-{self.results["scale_max"]:.1f} psi)',
                fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"{save_dir}/gauge_benchmark_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {plot_file}")

            plt.show()

        except Exception as e:
            print(f"❌ Error creating plots: {e}")

    def save_results(self, output_file="benchmark_results.txt"):
        """Save detailed results to text file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(output_file, 'w') as f:
                f.write("GAUGE READING BENCHMARK RESULTS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Input file: {self.excel_file}\n\n")

                # Write all metrics
                for key, value in self.results.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            print(f"💾 Results saved to: {output_file}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Gauge Reading Benchmark Evaluator')
    parser.add_argument('excel_file', help='Path to Excel file with test results')
    parser.add_argument('--output-dir', default='benchmark_output',
                        help='Output directory for plots and results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to text file')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.excel_file).exists():
        print(f"❌ File not found: {args.excel_file}")
        sys.exit(1)

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Initialize evaluator
    evaluator = GaugeBenchmarkEvaluator(args.excel_file)

    # Load and process data
    if not evaluator.load_data():
        sys.exit(1)

    if not evaluator.calculate_metrics():
        sys.exit(1)

    # Print results
    evaluator.print_results()

    # Generate plots
    if not args.no_plots:
        evaluator.create_plots(args.output_dir)

    # Save results
    if args.save_results:
        result_file = Path(args.output_dir) / "benchmark_results.txt"
        evaluator.save_results(result_file)

    print(f"\n✅ Benchmark evaluation completed!")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()