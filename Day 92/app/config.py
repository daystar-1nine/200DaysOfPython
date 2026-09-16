
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
