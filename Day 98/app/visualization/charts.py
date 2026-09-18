
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'class_distribution.png', 'image_dimensions.png', 'mask_pixel_dist.png', 'foreground_percentage.png',
        'sample_images.png', 'ground_truth_masks.png', 'image_mask_overlays.png', 'training_loss.png',
        'validation_loss.png', 'training_iou.png', 'validation_iou.png', 'training_dice.png',
        'validation_dice.png', 'prediction_masks.png', 'prediction_overlays.png', 'correct_segmentation.png',
        'boundary_errors.png', 'false_positive_regions.png', 'false_negative_regions.png',
        'iou_distribution.png', 'dice_distribution.png', 'threshold_vs_iou.png', 'threshold_vs_dice.png',
        'class_wise_performance.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
