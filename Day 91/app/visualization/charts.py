
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'sample_images.png', 'class_distribution.png', 'resized_images.png', 'rgb_conversion.png',
        'training_loss.png', 'validation_loss.png', 'training_accuracy.png', 'validation_accuracy.png',
        'learning_rate.png', 'model_accuracy.png', 'parameter_comparison.png', 'training_time.png',
        'trainable_parameters.png', 'confusion_matrix.png', 'precision_by_class.png', 'recall_by_class.png',
        'f1_by_class.png', 'correct_predictions.png', 'incorrect_predictions.png', 'low_confidence.png',
        'fine_tuning_comparison.png', 'feature_maps.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
