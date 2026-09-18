
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mock_charts(chart_dir):
    charts = [
        'track_count_time.png', 'unique_objects.png', 'active_tracks.png', 'lost_tracks.png',
        'track_duration_dist.png', 'track_length_dist.png', 'individual_trajectories.png',
        'combined_trajectories.png', 'avg_trajectory.png', 'zone_occupancy.png',
        'objects_by_class.png', 'tracks_by_class.png', 'avg_conf_class.png', 'duration_by_class.png',
        'fps_over_time.png', 'detection_latency.png', 'tracking_latency.png', 'e2e_latency.png',
        'id_switches.png', 'track_fragmentation.png', 'association_metrics.png'
    ]
    for c in charts:
        plt.figure()
        plt.plot([0,1])
        plt.savefig(os.path.join(chart_dir, c))
        plt.close()
