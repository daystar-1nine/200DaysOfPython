
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'class_distribution.png', 'images_per_split.png', 'boxes_per_image.png', 'box_width_dist.png',
        'box_height_dist.png', 'box_area_dist.png', 'training_loss.png', 'validation_loss.png',
        'precision_curve.png', 'recall_curve.png', 'map_50.png', 'map_50_95.png', 'gt_examples.png',
        'pred_examples.png', 'confidence_dist.png', 'iou_dist.png', 'precision_recall_curve.png',
        'class_wise_ap.png', 'confusion_matrix.png', 'detection_count.png', 'conf_vs_precision.png',
        'conf_vs_recall.png', 'conf_vs_f1.png', 'nms_thresh_comp.png', 'img_size_vs_inference.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
