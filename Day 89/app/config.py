
import os
class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHARTS_DIR = os.path.join(BASE_DIR, 'output', 'charts')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
