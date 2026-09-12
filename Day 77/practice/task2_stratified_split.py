"""Task 2: Stratified vs Unstratified Splitting Demonstration."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def run_task():
    # Imbalanced dataset: 90% Class 0, 10% Class 1
    np.random.seed(42)
    n = 1000
    y = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
    X = np.random.randn(n, 4)
    
    print(f"Overall Class Distribution: Class 0 = {(y==0).mean()*100:.1f}%, Class 1 = {(y==1).mean()*100:.1f}%")
    
    # 1. Unstratified Split
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y, test_size=0.2, random_state=99)
    print(f"Unstratified Test Set:      Class 0 = {(y_test_u==0).mean()*100:.1f}%, Class 1 = {(y_test_u==1).mean()*100:.1f}%")
    
    # 2. Stratified Split
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)
    print(f"Stratified Test Set:        Class 0 = {(y_test_s==0).mean()*100:.1f}%, Class 1 = {(y_test_s==1).mean()*100:.1f}%")

if __name__ == "__main__":
    run_task()
