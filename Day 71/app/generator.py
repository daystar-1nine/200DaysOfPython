"""
Data generator for synthetic A/B experimentation users.
"""

from typing import Optional
import numpy as np
import pandas as pd

class ExperimentGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_users(
        self,
        n_users: int = 10000,
        control_ratio: float = 0.50,
        p_control: float = 0.052,
        p_treatment: float = 0.058,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        
        user_ids = [f"USR-{10000 + i}" for i in range(n_users)]
        groups = rng.choice(["Control", "Treatment"], size=n_users, p=[control_ratio, 1.0 - control_ratio])
        
        conversions = np.zeros(n_users, dtype=int)
        revenues = np.zeros(n_users, dtype=float)
        orders = np.zeros(n_users, dtype=int)
        durations = np.zeros(n_users, dtype=float)
        bounces = np.zeros(n_users, dtype=int)
        refunds = np.zeros(n_users, dtype=int)
        
        for i in range(n_users):
            is_trt = (groups[i] == "Treatment")
            p_conv = p_treatment if is_trt else p_control
            
            conv = int(rng.binomial(n=1, p=p_conv))
            conversions[i] = conv
            
            if conv == 1:
                n_ord = int(rng.choice([1, 2, 3, 4], p=[0.75, 0.18, 0.05, 0.02]))
                orders[i] = n_ord
                base_rev = float(rng.lognormal(mean=3.8, sigma=0.45) * n_ord)
                revenues[i] = round(base_rev, 2)
                dur = float(rng.normal(loc=260.0 if is_trt else 240.0, scale=45.0))
                durations[i] = round(max(15.0, dur), 1)
                bounces[i] = 0
                refunds[i] = int(rng.binomial(n=1, p=0.024 if is_trt else 0.026))
            else:
                orders[i] = 0
                revenues[i] = 0.0
                bounce = int(rng.binomial(n=1, p=0.38 if is_trt else 0.41))
                bounces[i] = bounce
                if bounce == 1:
                    durations[i] = round(float(rng.uniform(5.0, 25.0)), 1)
                else:
                    dur = float(rng.normal(loc=115.0 if is_trt else 110.0, scale=30.0))
                    durations[i] = round(max(5.0, dur), 1)
                refunds[i] = 0
                
        return pd.DataFrame({
            "user_id": user_ids,
            "group": groups,
            "converted": conversions,
            "revenue": revenues,
            "session_duration": np.clip(durations, 5.0, 900.0),
            "orders": orders,
            "bounce": bounces,
            "refunded": refunds
        })
