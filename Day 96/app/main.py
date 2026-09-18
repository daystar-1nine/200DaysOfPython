
import os
import sys
# Mock OpenCV and ultralytics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd

from app.config import Config
from app.analytics.line_counter import has_crossed_line
from app.performance.fps import FPSCounter
from app.visualization.charts import generate_mock_charts

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_video_pipeline(exp_id, name, imgsz, frame_skip):
    print(f"\nRunning {name} Pipeline (Size: {imgsz}, Skip: {frame_skip})...")
    
    fps = FPSCounter()
    line_y = 300
    in_count = 0
    out_count = 0
    
    # Mocking video processing loop
    for i in range(1, 101): # 100 frames
        if i % frame_skip != 0:
            continue
            
        # Simulate tracking crossing a line
        prev_p = (100, 290 + i) # Moving down
        curr_p = (100, 295 + i)
        
        cross = has_crossed_line(prev_p, curr_p, line_y)
        if cross == 1:
            out_count += 1
        elif cross == -1:
            in_count += 1
            
        fps.update()
        time.sleep(0.001) # Mock inference latency
        
    final_fps = fps.value()
    
    return {
        'experiment_id': exp_id,
        'model': name,
        'image_size': imgsz,
        'frame_skip': frame_skip,
        'processed_frames': fps.frame_count,
        'in_count': in_count,
        'out_count': out_count,
        'avg_fps': round(final_fps, 2)
    }

def main():
    print("🚀 Starting Day 96 Real-Time Video Analytics Engine...")
    
    results = []
    
    # E1: High Res, No Skip
    results.append(run_video_pipeline('EXP001', 'YOLO Baseline', 640, 1))
    # E2: Low Res, No Skip
    results.append(run_video_pipeline('EXP002', 'YOLO Fast', 320, 1))
    # E3: High Res, Skip 2
    results.append(run_video_pipeline('EXP003', 'YOLO Skip 2', 640, 2))
    # E4: High Res, Skip 3
    results.append(run_video_pipeline('EXP004', 'YOLO Skip 3', 640, 3))
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(Config.OUTPUT_DIR, 'video_experiments.csv'), index=False)
    
    print("\n--- Real-Time Benchmark Results ---")
    print(df.to_string(index=False))
    
    print("\nGenerating 15 Visualizations...")
    generate_mock_charts(Config.CHARTS_DIR)
    
    print("✅ Video Analytics Pipeline Complete.")

if __name__ == "__main__":
    main()
