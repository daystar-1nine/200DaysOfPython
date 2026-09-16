
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'convolution_comparison.png', 'depthwise_conv.png', 'pointwise_conv.png', 'depthwise_separable.png',
        'mobilenet_block.png', 'sample_images.png', 'class_distribution.png', 'training_loss.png',
        'validation_loss.png', 'training_accuracy.png', 'validation_accuracy.png', 'learning_rate.png',
        'accuracy_comparison.png', 'f1_comparison.png', 'parameter_comparison.png', 'model_size_comparison.png',
        'training_time.png', 'latency_comparison.png', 'confusion_matrix.png', 'precision_by_class.png',
        'recall_by_class.png', 'f1_by_class.png', 'correct_predictions.png', 'incorrect_predictions.png',
        'confidence_distribution.png', 'accuracy_vs_parameters.png', 'accuracy_vs_latency.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
