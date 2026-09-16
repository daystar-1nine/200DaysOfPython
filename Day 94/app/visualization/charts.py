
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'class_distribution.png', 'images_per_split.png', 'bbox_width_dist.png', 'bbox_height_dist.png',
        'bbox_area_dist.png', 'sample_gt.png', 'sample_pred.png', 'correct_detections.png',
        'false_positives.png', 'false_negatives.png', 'training_loss.png', 'validation_loss.png',
        'precision_curve.png', 'recall_curve.png', 'map_curve.png', 'confusion_matrix.png',
        'precision_recall_curve.png', 'confidence_dist.png', 'iou_dist.png', 'class_wise_ap.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
