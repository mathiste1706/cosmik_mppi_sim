#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import torch.nn as nn
# import torch.nn.functional as F
from ...geom.sdf.robot_world import RobotWorldCollisionPrimitive
from .gaussian_projection import GaussianProjection

def sphere_pair_link_distance(robot_spheres, world_spheres):
    """Maximum signed overlap over every robot/world sphere pair for one link.

    The compiled form fuses broadcast, norm and reduction to avoid materializing
    the enormous [particles*horizon, robot spheres, human spheres, xyz] tensor.
    Pair membership, sign convention and geometry are unchanged.
    """
    difference = robot_spheres[:, :, :3].unsqueeze(2) - world_spheres[:, :3].view(1, 1, -1, 3)
    signed = world_spheres[:, 3].view(1, 1, -1) - torch.linalg.vector_norm(difference, dim=-1) + robot_spheres[:, :, 3].unsqueeze(-1)
    return torch.amax(signed, dim=(1, 2))


def sphere_pair_link_distance_timed(robot_spheres, world_spheres):
    """Time-indexed form of sphere_pair_link_distance.

    robot_spheres: [particles, horizon, robot spheres, 4]
    world_spheres: [horizon, world spheres, 4], the obstacle geometry predicted
    for the same rollout step as each robot state. Returns [particles, horizon].
    """
    difference = robot_spheres[..., :3].unsqueeze(3) - world_spheres[:, :, :3].view(1, world_spheres.shape[0], 1, -1, 3)
    signed = (world_spheres[:, :, 3].view(1, world_spheres.shape[0], 1, -1)
              - torch.linalg.vector_norm(difference, dim=-1) + robot_spheres[..., 3].unsqueeze(-1))
    return torch.amax(signed, dim=(2, 3))


class PrimitiveCollisionCost(nn.Module):
    def __init__(self, weight=None, world_params=None, robot_params=None, gaussian_params={},
                 distance_threshold=0.1,
                 violation_clip_max=0.2,
                 violation_scale=0.25,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 fused_dynamic_collision=False, ignore_links=None):
        # was: fused_dynamic_collision=False):
        super(PrimitiveCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        robot_collision_params = robot_params['robot_collision_params']
        self.batch_size = -1
        # BUILD world and robot:
        self.robot_world_coll = RobotWorldCollisionPrimitive(robot_collision_params,
                                                             world_params['world_model'],
                                                             tensor_args=self.tensor_args,
                                                             bounds=robot_params['world_collision_params']['bounds'],
                                                             grid_resolution=robot_params['world_collision_params']['grid_resolution'])
        
        self.n_world_objs = self.robot_world_coll.world_coll.n_objs
        self.t_mat = None
        # Links whose spheres this cost ignores, e.g. the fixed base (panda_link0/link1) for the
        # table: their collision spheres overhang the table top by 2-4 cm in every posture, which
        # added a constant 66.9/step that inflated CaT's reward baseline (robot stalled near targets).
        link_objs = list(robot_collision_params['link_objs'])
        self._ignore_link_idx = [link_objs.index(name) for name in (ignore_links or [])]
        self.distance_threshold = distance_threshold
        self.violation_clip_max = float(violation_clip_max)
        self.violation_scale = float(violation_scale)
        self.fused_dynamic_collision = bool(fused_dynamic_collision)
        # Lazy compilation keeps the controller serializable before worker startup.
        self._dynamic_pair_kernel = None
        self._dynamic_pair_kernel_timed = None
        self._dynamic_world_spheres = None
        self._empty_world_dist = None
        self._dynamic_link_dist = None
        # Point 3: dynamic-only (human arm) signed distance [batch*horizon, links] from the last
        # forward, read by HumanProximityCost; None when no dynamic spheres are streamed.
        self.last_dynamic_link_dist = None
        if self.violation_clip_max <= 0.0:
            raise ValueError(f"violation_clip_max must be > 0, got {self.violation_clip_max}.")
        if self.violation_scale <= 0.0:
            raise ValueError(f"violation_scale must be > 0, got {self.violation_scale}.")

    def __getstate__(self):
        state = super().__getstate__()
        # Recompile in the optimizer worker after restoring a warmed controller.
        state['_dynamic_pair_kernel'] = None
        state['_dynamic_pair_kernel_timed'] = None
        state['last_dynamic_link_dist'] = None  # Point 3: per-call cache, not state.
        return state

    @staticmethod
    def _shape_violation(
        dist,
        distance_threshold: float,
        violation_clip_max: float,
        violation_scale: float,
    ):
        dist = dist + float(distance_threshold)
        dist[dist <= 0.0] = 0.0
        dist[dist > float(violation_clip_max)] = float(violation_clip_max)
        dist = dist / float(violation_scale)
        return torch.sum(dist, dim=-1)

    def set_dynamic_world_spheres(self, dynamic_world_spheres):
        if dynamic_world_spheres is None:
            return False

        # [N, 4]: the same spheres at every rollout step (observed geometry held).
        # [H, N, 4]: spheres predicted for each rollout step h (forecast mode).
        dynamic_world_spheres = torch.as_tensor(dynamic_world_spheres, **self.tensor_args)
        if dynamic_world_spheres.numel() == 0:
            dynamic_world_spheres = dynamic_world_spheres.reshape(0, 4)
        if dynamic_world_spheres.ndim not in (2, 3) or dynamic_world_spheres.shape[-1] != 4:
            raise ValueError(
                f"dynamic_world_spheres must have shape [N, 4] or [H, N, 4], got {tuple(dynamic_world_spheres.shape)}"
            )

        if (
            self._dynamic_world_spheres is None
            or self._dynamic_world_spheres.shape != dynamic_world_spheres.shape
        ):
            self._dynamic_world_spheres = torch.zeros(dynamic_world_spheres.shape, **self.tensor_args)
        self._dynamic_world_spheres.copy_(dynamic_world_spheres)
        return True

    @staticmethod
    def _align_world_horizon(world_spheres, horizon):
        """Match a [H_w, N, 4] forecast to the rollout horizon (truncate, or hold the last step)."""
        if world_spheres.shape[0] >= horizon:
            return world_spheres[:horizon]
        pad = world_spheres[-1:].expand(horizon - world_spheres.shape[0], -1, -1)
        return torch.cat((world_spheres, pad), dim=0)

    def _compute_dynamic_collision_dist_timed(self, link_spheres, dist, horizon, n_links):
        world = self._align_world_horizon(self._dynamic_world_spheres, horizon).contiguous()
        for link_idx in range(n_links):
            # Batch rows are ordered particle-major: row = particle * horizon + step.
            robot_spheres = link_spheres[link_idx].view(-1, horizon, link_spheres[link_idx].shape[1], 4)
            if getattr(self, 'fused_dynamic_collision', False):
                if self._dynamic_pair_kernel_timed is None:
                    self._dynamic_pair_kernel_timed = torch.compile(
                        sphere_pair_link_distance_timed, fullgraph=True, dynamic=True)
                dist[:, link_idx] = self._dynamic_pair_kernel_timed(robot_spheres, world).reshape(-1)
            else:
                dist[:, link_idx] = sphere_pair_link_distance_timed(robot_spheres, world).reshape(-1)
        return dist

    def _compute_dynamic_collision_dist(self, batch_size, n_links, horizon=1):
        if self._dynamic_world_spheres is None or self._dynamic_world_spheres.shape[-2] == 0:
            return None
        link_spheres = self.robot_world_coll.robot_coll.get_batch_robot_link_spheres()
        if link_spheres is None:
            return None

        if self._dynamic_link_dist is None or self._dynamic_link_dist.shape != (batch_size, n_links):
            self._dynamic_link_dist = torch.zeros((batch_size, n_links), **self.tensor_args)
        dist = self._dynamic_link_dist

        if self._dynamic_world_spheres.ndim == 3:
            return self._compute_dynamic_collision_dist_timed(link_spheres, dist, horizon, n_links)

        world_centers = self._dynamic_world_spheres[:, :3].view(1, 1, -1, 3)
        world_radii = self._dynamic_world_spheres[:, 3].view(1, 1, -1)

        for link_idx in range(n_links):
            robot_spheres = link_spheres[link_idx]
            if getattr(self, 'fused_dynamic_collision', False):
                if self._dynamic_pair_kernel is None:
                    self._dynamic_pair_kernel = torch.compile(sphere_pair_link_distance, fullgraph=True, dynamic=True)
                dist[:, link_idx] = self._dynamic_pair_kernel(robot_spheres, self._dynamic_world_spheres)
                continue
            robot_centers = robot_spheres[:, :, :3]
            robot_radii = robot_spheres[:, :, 3].unsqueeze(-1)
            center_distance = torch.norm(robot_centers.unsqueeze(2) - world_centers, dim=-1)
            signed_dist = world_radii - center_distance + robot_radii
            max_over_world = torch.max(signed_dist, dim=-1)[0]
            dist[:, link_idx] = torch.max(max_over_world, dim=-1)[0]
        return dist

    def _compute_signed_link_distance_batch(self, link_pos_batch, link_rot_batch, horizon=1):
        batch_size = link_pos_batch.shape[0]
        n_links = link_pos_batch.shape[1]

        if self.n_world_objs > 0:
            dist = self.robot_world_coll.check_robot_sphere_collisions(link_pos_batch, link_rot_batch)
        else:
            self.robot_world_coll.robot_coll.update_batch_robot_collision_objs(link_pos_batch, link_rot_batch)
            if (
                self._empty_world_dist is None
                or self._empty_world_dist.shape[0] != batch_size
                or self._empty_world_dist.shape[1] != n_links
            ):
                self._empty_world_dist = torch.zeros((batch_size, n_links), **self.tensor_args) - 10.0
            dist = self._empty_world_dist

        dynamic_dist = self._compute_dynamic_collision_dist(batch_size, n_links, horizon)
        self.last_dynamic_link_dist = dynamic_dist  # Point 3 (read before the next forward).
        if dynamic_dist is not None:
            dist = torch.maximum(dist, dynamic_dist)
        return dist

    def forward(
        self,
        link_pos_seq,
        link_rot_seq,
        return_violation=False,
        violation_distance_threshold=None,
        violation_clip_max=None,
        violation_scale=None,
    ):

        
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        if(self.batch_size != batch_size):
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(self.batch_size * horizon, clone_pose=True, clone_points=True)

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        dist = self._compute_signed_link_distance_batch(link_pos_batch, link_rot_batch, horizon)
        dist = dist.view(batch_size, horizon, n_links)  # , self.n_world_objs)
        if self._ignore_link_idx:
            dist = dist.clone()
            dist[..., self._ignore_link_idx] = -1.0  # ignored links: 1 m clear, never a violation
        violation_default = self._shape_violation(
            dist.clone(),
            self.distance_threshold,
            self.violation_clip_max,
            self.violation_scale,
        )

        cost = self.weight * violation_default
        cost = cost.to(inp_device)

        if return_violation:
            if violation_distance_threshold is None:
                threshold = self.distance_threshold
            else:
                threshold = float(violation_distance_threshold)

            if violation_clip_max is None:
                clip_max = self.violation_clip_max
            else:
                clip_max = float(violation_clip_max)
                if clip_max <= 0.0:
                    raise ValueError(f"violation_clip_max must be > 0, got {clip_max}.")

            if violation_scale is None:
                scale = self.violation_scale
            else:
                scale = float(violation_scale)
                if scale <= 0.0:
                    raise ValueError(f"violation_scale must be > 0, got {scale}.")

            if (
                threshold == self.distance_threshold
                and clip_max == self.violation_clip_max
                and scale == self.violation_scale
            ):
                violation_unweighted = violation_default
            else:
                violation_unweighted = self._shape_violation(
                    dist,
                    threshold,
                    clip_max,
                    scale,
                )
            return cost, violation_unweighted.to(inp_device)

        return cost
