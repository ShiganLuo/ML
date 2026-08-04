# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
"""Interaction-enhanced robust aggregation strategy.

Adds interaction features to capture non-linear relationships:
- alt_ratio × entropy: compound instability indicator
- del_ratio × ins_ratio: deletion/insertion balance
- depth × alt_ratio: depth-weighted instability
- del_ratio / ins_ratio: deletion/insertion ratio
- unit1_ratio / unit2_ratio: cross-unit-length instability

These interactions may capture signals that linear combinations miss.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from .robust_aggregation import RobustAggregation
from ..utils import _common_features, _unit_len_features


class InteractionRobustAggregation(RobustAggregation):
    """Robust aggregation with interaction features.
    
    Extends RobustAggregation with:
    - Pairwise interactions between key features
    - Ratio features (division)
    - Cross-unit-length features
    """

    _FEATURE_NAMES = RobustAggregation._FEATURE_NAMES + [
        # Interaction features
        'interact_alt_entropy',      # alt_ratio × entropy
        'interact_del_ins',          # del_ratio × ins_ratio
        'interact_depth_alt',        # depth × alt_ratio
        
        # Ratio features
        'ratio_del_ins',             # del_ratio / ins_ratio
        'ratio_alt_entropy',         # alt_ratio / entropy
        
        # Cross-unit features
        'ratio_unit1_unit2_alt',     # alt_unit1 / alt_unit2
        'ratio_unit1_unit3_alt',     # alt_unit1 / alt_unit3
        'ratio_unit2_unit3_alt',     # alt_unit2 / alt_unit3
        
        # Depth-weighted interactions
        'depth_w_interact_alt_entropy',  # depth-weighted (alt × entropy)
        'depth_w_interact_del_ins',      # depth-weighted (del × ins)
        
        # Extreme value features
        'prop_alt_gt_0.9',           # proportion with alt > 0.9
        'prop_entropy_gt_2.5',       # proportion with entropy > 2.5
        'max_entropy',               # maximum entropy
        'min_entropy',               # minimum entropy
        'range_entropy',             # entropy range
    ]

    def aggregate(self, lf: pd.DataFrame) -> Optional[Dict]:
        if len(lf) == 0:
            return None
        
        # Get base robust features
        f = super().aggregate(lf)
        if f is None:
            return None
        
        alt = lf['alt_ratio']
        ent = lf['entropy']
        del_ratio = lf['del_ratio']
        ins_ratio = lf['ins_ratio']
        depth = lf['depth']
        unit_len = lf['unit_len']
        
        # Interaction features (per-locus, then aggregate)
        interact_alt_ent = alt * ent
        interact_del_ins = del_ratio * ins_ratio
        interact_depth_alt = depth * alt
        
        f['interact_alt_entropy'] = float(np.mean(interact_alt_ent))
        f['interact_del_ins'] = float(np.mean(interact_del_ins))
        f['interact_depth_alt'] = float(np.mean(interact_depth_alt))
        
        # Ratio features
        f['ratio_del_ins'] = float(np.mean(del_ratio) / (np.mean(ins_ratio) + 1e-10))
        f['ratio_alt_entropy'] = float(np.mean(alt) / (np.mean(ent) + 1e-10))
        
        # Cross-unit features
        unit1_mask = unit_len == 1
        unit2_mask = unit_len == 2
        unit3_mask = unit_len == 3
        
        alt_unit1 = alt[unit1_mask].mean() if unit1_mask.sum() > 0 else 0
        alt_unit2 = alt[unit2_mask].mean() if unit2_mask.sum() > 0 else 0
        alt_unit3 = alt[unit3_mask].mean() if unit3_mask.sum() > 0 else 0
        
        f['ratio_unit1_unit2_alt'] = float(alt_unit1 / (alt_unit2 + 1e-10))
        f['ratio_unit1_unit3_alt'] = float(alt_unit1 / (alt_unit3 + 1e-10))
        f['ratio_unit2_unit3_alt'] = float(alt_unit2 / (alt_unit3 + 1e-10))
        
        # Depth-weighted interactions
        total_depth = depth.sum()
        f['depth_w_interact_alt_entropy'] = float((interact_alt_ent * depth).sum() / total_depth)
        f['depth_w_interact_del_ins'] = float((interact_del_ins * depth).sum() / total_depth)
        
        # Extreme value features
        f['prop_alt_gt_0.9'] = float((alt > 0.9).mean())
        f['prop_entropy_gt_2.5'] = float((ent > 2.5).mean())
        f['max_entropy'] = float(ent.max())
        f['min_entropy'] = float(ent.min())
        f['range_entropy'] = float(ent.max() - ent.min())
        
        return f

    def get_feature_names(self) -> List[str]:
        return self._FEATURE_NAMES
