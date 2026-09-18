
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.tracking.centroid_tracker import CentroidTracker
from app.analytics.trajectory import calculate_distance
from app.visualization.charts import generate_mock_charts

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_mot_pipeline(tracker_type, detections_sequence):
    print(f"\nRunning MOT Pipeline ({tracker_type})...")
    
    tracker = CentroidTracker(max_disappeared=2)
    history = {}
    
    start_time = time.perf_counter()
    
    for frame_dets in detections_sequence:
        objects = tracker.update(frame_dets)
        
        for obj_id, centroid in objects.items():
            if obj_id not in history:
                history[obj_id] = []
            history[obj_id].append(centroid)
            
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Calculate Analytics
    distances = {obj_id: calculate_distance(pts) for obj_id, pts in history.items()}
    total_switches = 0 # Mocked
    
    return {
        'tracker': tracker_type,
        'unique_identities': len(history),
        'avg_distance_px': round(sum(distances.values()) / max(len(distances), 1), 2),
        'id_switches': total_switches,
        'tracking_latency_ms': round(latency_ms, 2)
    }

def main():
    print("🚀 Starting Day 97 Multi-Object Tracking Engine...")
    
    # Mock Detection Sequence: Person walking left to right
    seq = [
        [(100, 100, 150, 200)], # Frame 1
        [(110, 100, 160, 200)], # Frame 2
        [(120, 100, 170, 200)], # Frame 3
        [],                     # Frame 4 (Missed detection)
        [(140, 100, 190, 200)]  # Frame 5 (Tracker should recover)
    ]
    
    results = []
    
    results.append(run_mot_pipeline('Centroid Tracker Baseline', seq))
    results.append(run_mot_pipeline('SORT (Simulated)', seq))
    results.append(run_mot_pipeline('Deep SORT (Simulated)', seq))
    results.append(run_mot_pipeline('ByteTrack (Simulated)', seq))
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(Config.OUTPUT_DIR, 'tracking_experiments.csv'), index=False)
    
    print("\n--- MOT Benchmark Results ---")
    print(df.to_string(index=False))
    
    print("\nGenerating 21 Visualizations...")
    generate_mock_charts(Config.CHARTS_DIR)
    
    print("✅ Multi-Object Tracking Pipeline Complete.")

if __name__ == "__main__":
    main()
