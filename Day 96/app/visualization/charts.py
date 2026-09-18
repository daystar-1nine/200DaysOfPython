
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'fps_over_time.png', 'latency_over_time.png', 'avg_fps_by_input.png', 'processing_time_dist.png',
        'objects_per_frame.png', 'class_distribution.png', 'confidence_distribution.png', 'detection_count_time.png',
        'track_duration_dist.png', 'unique_objects.png', 'track_trajectories.png', 'entering_vs_exiting.png',
        'resolution_vs_fps.png', 'confidence_vs_object_count.png', 'model_size_vs_latency.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
