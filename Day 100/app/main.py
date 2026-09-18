
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.data.loader import generate_mock_dataset, load_and_clean
from app.analysis.statistics import generate_summary
from app.features.engineering import prepare_features
from app.models.model import train_and_evaluate
from app.visualization.charts import generate_milestone_charts

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("🚀 Starting Day 100 AI Data Analytics Milestone Pipeline...")
    
    # 1. Data Loader
    raw_path = os.path.join(Config.DATA_DIR, 'raw', 'mock_dataset.csv')
    print("\n1. Generating and Loading Data...")
    generate_mock_dataset(raw_path)
    
    # 2. Cleaning
    print("2. Cleaning Data...")
    df = load_and_clean(raw_path)
    
    # 3. EDA Summary
    print("3. Generating Statistical Summary...")
    print(generate_summary(df))
    
    # 4. Feature Engineering & Split
    print("\n4. Feature Engineering & Splitting...")
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    # 5. Model Training & Evaluation
    print("5. Training Baseline Random Forest Model...")
    model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\n--- Model Evaluation ---")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v}")
        
    # 6. Visualizations
    print("\n6. Generating 12 Milestone Visualizations...")
    generate_milestone_charts(Config.CHARTS_DIR)
    
    print("\n✅ Day 100 Mini-Capstone Complete! 50% Milestone Reached.")

if __name__ == "__main__":
    main()
