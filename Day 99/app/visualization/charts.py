
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'images_per_class.png', 'instances_per_class.png', 'instances_per_image.png', 'mask_area_distribution.png',
        'mask_coverage_distribution.png', 'original_images.png', 'ground_truth_masks.png', 'predicted_masks.png',
        'mask_overlays.png', 'instance_colored_masks.png', 'area_distribution.png', 'perimeter_distribution.png',
        'aspect_ratio_distribution.png', 'circularity_distribution.png', 'object_width_distribution.png',
        'object_height_distribution.png', 'iou_distribution.png', 'dice_distribution.png', 'class_wise_iou.png',
        'class_wise_dice.png', 'objects_per_image.png', 'object_centroid_distribution.png', 'object_size_vs_confidence.png',
        'area_vs_confidence.png', 'shape_feature_correlation.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
