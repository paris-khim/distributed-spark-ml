import numpy as np
from scipy import stats

class DriftDetector:
    """Advanced statistical monitoring for feature drift using KS Test."""
    def __init__(self, baseline_data: np.ndarray):
        self.baseline = baseline_data

    def detect_drift(self, current_data: np.ndarray, alpha=0.05):
        """Perform Kolmogorov-Smirnov test to identify data drift."""
        ks_stat, p_value = stats.ks_2samp(self.baseline, current_data)
        drift_detected = p_value < alpha
        return {
            "statistic": float(ks_stat),
            "p_value": float(p_value),
            "is_drifted": bool(drift_detected)
        }
