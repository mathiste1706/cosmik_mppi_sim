"""Anticipatory cost on the clearance to the human arm (point 3).

Uses only the streamed dynamic spheres (the human arm forecast, one set per rollout step),
never the static world (table, boxes). It adds two terms on top of the existing costs; it
replaces nothing:

* proximity: grows smoothly as the clearance h falls from influence_distance to margin,
      cost = weight * clamp((influence_distance - h) / (influence_distance - margin), 0, clip)^2
  so MPPI starts to move away before the 10 cm CaT margin and the 5 cm primitive cost.

* approach (discrete control barrier function on b = h - margin):
      violation_k = relu((1 - cbf_gamma) * b_k - b_{k+1})
  only counted while h_k < influence_distance. It penalises closing in too fast, whether the
  robot moves toward the arm or the forecast arm moves toward the robot. cbf_gamma in (0, 1]:
  small = strict (b may shrink only slowly), 1 = only forbids entering the margin.

h is the minimum clearance over the robot links, from the same sphere pairs as
primitive_collision (h = -signed overlap). Over/under choice is still left to MPPI.
"""
import torch
import torch.nn as nn


class HumanProximityCost(nn.Module):
    def __init__(self, weight=0.0, influence_distance=0.25, margin=0.10, clip=1.0,
                 cbf_weight=0.0, cbf_gamma=0.2,
                 tensor_args={'device': torch.device('cpu'), 'dtype': torch.float32}):
        super().__init__()
        if influence_distance <= margin:
            raise ValueError(f"influence_distance ({influence_distance}) must be > margin ({margin}).")
        if not 0.0 < cbf_gamma <= 1.0:
            raise ValueError(f"cbf_gamma must be in (0, 1], got {cbf_gamma}.")
        self.tensor_args = tensor_args
        self.weight = float(weight)
        self.influence_distance = float(influence_distance)
        self.margin = float(margin)
        self.clip = float(clip)
        self.cbf_weight = float(cbf_weight)
        self.cbf_gamma = float(cbf_gamma)

    def clearance(self, dynamic_link_dist):
        """dynamic_link_dist [B, H, links] signed overlap (> 0 = penetration) -> clearance [B, H]."""
        return -torch.amax(dynamic_link_dist, dim=-1)

    def forward(self, dynamic_link_dist):
        h = self.clearance(dynamic_link_dist)
        span = self.influence_distance - self.margin
        cost = torch.zeros_like(h)
        if self.weight > 0.0:
            prox = ((self.influence_distance - h) / span).clamp(min=0.0, max=self.clip)
            cost = cost + self.weight * prox * prox
        if self.cbf_weight > 0.0 and h.shape[1] > 1:
            b = h - self.margin
            violation = torch.relu((1.0 - self.cbf_gamma) * b[:, :-1] - b[:, 1:])
            violation = violation * (h[:, :-1] < self.influence_distance)
            # Charged to the step reached (k + 1); step 0 has no predecessor in the rollout.
            cost[:, 1:] = cost[:, 1:] + self.cbf_weight * violation
        return cost
