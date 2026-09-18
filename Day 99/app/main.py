
import os
import sys
import pandas as pd
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.shape.features import extract_shape_features
from app.morphology.clean import clean_mask
from app.visualization.charts import generate_mock_charts

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("🚀 Starting Day 99 Instance Segmentation Analytics Engine...")
    
    # Mocking Instance Masks (100x100)
    print("Generating Mock Instance Masks...")
    mask1 = np.zeros((100, 100))
    cv2.circle(mask1, (30, 30), 15, 1, -1) # Instance 1: Circular Object
    
    mask2 = np.zeros((100, 100))
    cv2.rectangle(mask2, (60, 60), (80, 90), 1, -1) # Instance 2: Rectangular Object
    
    instances = [
        {'id': 1, 'class': 'apple', 'confidence': 0.95, 'mask': mask1},
        {'id': 2, 'class': 'bottle', 'confidence': 0.88, 'mask': mask2}
    ]
    
    report = []
    
    for inst in instances:
        cleaned = clean_mask(inst['mask'], operation='opening')
        features = extract_shape_features(cleaned)
        
        if features:
            features['id'] = inst['id']
            features['class'] = inst['class']
            features['confidence'] = inst['confidence']
            report.append(features)
            
    df = pd.DataFrame(report)
    cols = ['id', 'class', 'confidence', 'area', 'perimeter', 'width', 'height', 'aspect_ratio', 'circularity', 'cx', 'cy']
    df = df[cols]
    
    df.to_csv(os.path.join(Config.OUTPUT_DIR, 'instance_metrics.csv'), index=False)
    
    print("\n--- Object Shape Analytics Report ---")
    print(df.to_string(index=False))
    
    print("\nGenerating 25 Analytics Visualizations...")
    generate_mock_charts(Config.CHARTS_DIR)
    
    print("✅ Instance Segmentation & Shape Analytics Pipeline Complete.")

if __name__ == "__main__":
    main()
