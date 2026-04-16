#!/usr/bin/env python3
"""
Realtime Chart Viewer for Gauge Reader
Reads gauge data and displays live updating charts
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from collections import deque
import argparse
import os


class GaugeChartViewer:
    def __init__(self, data_file="gauge_readings.json", max_points=100):
        self.data_file = data_file
        self.max_points = max_points

        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.readings = deque(maxlen=max_points)

        # Chart configuration
        self.fig, self.ax1 = plt.subplots(1, 1, figsize=(12, 6))
        self.fig.suptitle('Gauge Reader - Realtime Data', fontsize=16, fontweight='bold')

        # Reading chart (top)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Gauge Reading')
        self.ax1.set_ylabel('Reading (units)', fontsize=12)
        self.ax1.set_title('Gauge Reading Over Time', fontsize=14)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()


        # Statistics
        self.stats_text = self.fig.text(0.02, 0.02, '', fontsize=10,
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # Data reading thread
        self.running = True
        self.data_thread = threading.Thread(target=self.data_reader_loop, daemon=True)

        # Last data
        self.last_reading = None
        self.last_angle = None
        self.data_count = 0

    def start(self):
        """Start the chart viewer"""
        print("🚀 Starting Gauge Chart Viewer...")
        print(f"📂 Reading data from: {self.data_file}")
        print("📊 Chart will update automatically")
        print("🔄 Close window to exit")

        # Start data reading thread
        self.data_thread.start()

        # Start animation
        ani = animation.FuncAnimation(self.fig, self.update_chart,
                                      interval=100, blit=False, cache_frame_data=False)  # ← 100ms thay vì 500ms

        # Show plot
        plt.tight_layout()
        plt.show()

        # Cleanup
        self.running = False

    def data_reader_loop(self):
        """Continuously read data from file"""
        last_size = 0

        while self.running:
            try:
                if os.path.exists(self.data_file):
                    current_size = os.path.getsize(self.data_file)

                    if current_size > last_size:
                        # Read new lines only
                        with open(self.data_file, 'r') as f:
                            f.seek(last_size)  # Start from last position
                            new_lines = f.readlines()

                        for line in new_lines:
                            if line.strip():
                                try:
                                    data = json.loads(line.strip())
                                    timestamp = datetime.fromtimestamp(data.get('timestamp', time.time()))
                                    reading = float(data['reading'])

                                    self.timestamps.append(timestamp)
                                    self.readings.append(reading)

                                    self.last_reading = reading
                                    self.data_count += 1

                                except json.JSONDecodeError:
                                    continue

                        last_size = current_size
                else:
                    # Create fresh start marker
                    fresh_data = {
                        'reading': 0.0,
                        'angle_degrees': 0.0,
                        'timestamp': time.time(),
                        'status': 'fresh_start',
                        'datetime': datetime.now().isoformat()
                    }
                    with open(self.data_file, 'w') as f:
                        json.dump(fresh_data, f)
                        f.write('\n')
                    print(f"📝 Created fresh data file: {self.data_file}")
            except Exception as e:
                print(f"❌ Data reading error: {e}")

            time.sleep(0.05)
    def update_chart(self, frame):
        """Update chart with new data"""
        if len(self.timestamps) < 1:
            return self.line1, self.line2

        try:
            # Convert timestamps to matplotlib format
            times = list(self.timestamps)
            readings_list = list(self.readings)

            # Update reading chart
            self.line1.set_data(times, readings_list)
            self.ax1.relim()
            self.ax1.autoscale_view()

            # Set time window (last 60 seconds)
            if len(times) > 1:
                now = times[-1]
                start_time = now - timedelta(seconds=30)
                self.ax1.set_xlim(start_time, now)

            # Format x-axis
            self.ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            self.fig.autofmt_xdate()



            # Update statistics
            if readings_list:
                current_reading = readings_list[-1]
                avg_reading = np.mean(readings_list)
                min_reading = np.min(readings_list)
                max_reading = np.max(readings_list)

                stats_text = f"Current: {current_reading:.2f} | Avg: {avg_reading:.2f} | " \
                             f"Min: {min_reading:.2f} | Max: {max_reading:.2f} | " \
                             f"Points: {len(readings_list)}"

                self.stats_text.set_text(stats_text)

                # Force refresh để update nhanh hơn
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

                # Color coding based on value
                if current_reading > avg_reading * 1.5:
                    self.line1.set_color('red')
                elif current_reading > avg_reading * 1.2:
                    self.line1.set_color('orange')
                else:
                    self.line1.set_color('blue')

        except Exception as e:
            print(f"❌ Chart update error: {e}")

        return self.line1

class GaugeDataLogger:
    """Helper class to log data from gauge reader"""

    def __init__(self, data_file="gauge_readings.json", max_points=100):
        self.data_file = data_file
        self.max_points = max_points

        # Clear old data for fresh start
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
            print(f"🧹 Cleared old data file: {self.data_file}")

        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.readings = deque(maxlen=max_points)
        self.angles = deque(maxlen=max_points)

    def log_reading(self, reading, angle_degrees, timestamp=None):
        """Log a single reading to file"""
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
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Logging error: {e}")
            return False


def generate_demo_data(filename="gauge_readings.json", duration=60):
    """Generate demo data for testing the chart"""
    print(f"🎭 Generating demo data for {duration} seconds...")
    logger = GaugeDataLogger(filename)

    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            # Generate realistic gauge data
            t = time.time() - start_time
            base_reading = 1.5 + 0.5 * np.sin(t * 0.1)  # Slow sine wave
            noise = np.random.normal(0, 0.05)  # Small random noise
            reading = max(0, base_reading + noise)

            # Corresponding angle (simulate gauge needle)
            angle = 45 + reading * 30  # 45° to 135° range

            logger.log_reading(reading, angle)

            print(f"📊 Demo: {reading:.2f} units, {angle:.1f}°")
            time.sleep(1)  # 1 second intervals

    except KeyboardInterrupt:
        print("\n🛑 Demo data generation stopped")


def main():
    parser = argparse.ArgumentParser(description='Gauge Reader Chart Viewer')
    parser.add_argument('--file', '-f', default='gauge_readings.json',
                        help='Data file to read from (default: gauge_readings.json)')
    parser.add_argument('--demo', '-d', action='store_true',
                        help='Generate demo data for testing')
    parser.add_argument('--demo-duration', type=int, default=60,
                        help='Demo duration in seconds (default: 60)')
    parser.add_argument('--max-points', type=int, default=120,
                        help='Maximum points to display (default: 120)')
    parser.add_argument('--keep-old', action='store_true',
                        help='Keep old data instead of starting fresh')

    args = parser.parse_args()

    if args.demo:
        print("🎭 Demo mode - generating sample data...")
        threading.Thread(target=generate_demo_data,
                         args=(args.file, args.demo_duration), daemon=True).start()
        time.sleep(2)  # Wait for first data point

    # Start chart viewer
    if not args.keep_old:
        # Clear old data for fresh start
        if os.path.exists(args.file):
            os.remove(args.file)
            print(f"🧹 Starting fresh - cleared old data")

    viewer = GaugeChartViewer(args.file, args.max_points)
    viewer.start()


if __name__ == "__main__":
    main()