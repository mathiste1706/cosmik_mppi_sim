import unittest

import numpy as np
import torch

from storm_kit.mpc.cost.primitive_collision_cost import PrimitiveCollisionCost
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess


class _RobotColl:
    def __init__(self, link_spheres):
        self.link_spheres = link_spheres

    def get_batch_robot_link_spheres(self):
        return self.link_spheres


class _RobotWorldColl:
    def __init__(self, link_spheres):
        self.robot_coll = _RobotColl(link_spheres)


def make_cost(link_spheres):
    """PrimitiveCollisionCost with only the dynamic-sphere state (no URDF/world build)."""
    cost = PrimitiveCollisionCost.__new__(PrimitiveCollisionCost)
    torch.nn.Module.__init__(cost)
    cost.tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
    cost.fused_dynamic_collision = False
    cost._dynamic_pair_kernel = None
    cost._dynamic_pair_kernel_timed = None
    cost._dynamic_world_spheres = None
    cost._dynamic_link_dist = None
    cost.robot_world_coll = _RobotWorldColl(link_spheres)
    return cost


def sphere(x, r=0.1):
    return [x, 0.0, 0.0, r]


class TimedDynamicCollisionTests(unittest.TestCase):
    def test_stationary_forecast_matches_held_spheres(self):
        torch.manual_seed(0)
        particles, horizon = 4, 6
        link_spheres = [torch.rand(particles * horizon, 3, 4), torch.rand(particles * horizon, 2, 4)]
        world = torch.rand(5, 4)
        cost = make_cost(link_spheres)
        cost.set_dynamic_world_spheres(world)
        held = cost._compute_dynamic_collision_dist(particles * horizon, 2, horizon).clone()
        cost.set_dynamic_world_spheres(world.unsqueeze(0).repeat(horizon, 1, 1))
        timed = cost._compute_dynamic_collision_dist(particles * horizon, 2, horizon)
        torch.testing.assert_close(timed, held)

    def test_each_rollout_step_uses_its_own_forecast(self):
        # Particle 0 stays at x=0, particle 1 at x=2; the obstacle is at x=0 then x=2.
        particles, horizon = 2, 2
        robot = torch.tensor([[sphere(0.0)], [sphere(0.0)], [sphere(2.0)], [sphere(2.0)]])
        cost = make_cost([robot])
        cost.set_dynamic_world_spheres(torch.tensor([[sphere(0.0)], [sphere(2.0)]]))
        dist = cost._compute_dynamic_collision_dist(particles * horizon, 1, horizon).view(particles, horizon)
        torch.testing.assert_close(dist, torch.tensor([[0.2, -1.8], [-1.8, 0.2]]))

    def test_single_step_horizon_and_short_forecast_hold_last_step(self):
        cost = make_cost([torch.tensor([[sphere(1.0)], [sphere(1.0)], [sphere(1.0)]])])
        cost.set_dynamic_world_spheres(torch.tensor([[sphere(0.0)], [sphere(1.0)]]))
        # H=1: only the first forecast step is used.
        h1 = cost._compute_dynamic_collision_dist(3, 1, 1).view(3, 1)
        torch.testing.assert_close(h1, torch.full((3, 1), -0.8))
        # Forecast shorter than the horizon: last step is held.
        cost.set_dynamic_world_spheres(torch.tensor([[sphere(1.0)]]))
        cost.robot_world_coll.robot_coll.link_spheres = [torch.tensor([[sphere(1.0)]] * 6)]
        held = cost._compute_dynamic_collision_dist(6, 1, 3).view(2, 3)
        torch.testing.assert_close(held, torch.full((2, 3), 0.2))

    def test_rejects_bad_shapes(self):
        cost = make_cost([torch.zeros(1, 1, 4)])
        with self.assertRaises(ValueError):
            cost.set_dynamic_world_spheres(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            cost.set_dynamic_world_spheres(torch.zeros(2, 2, 2, 4))


def make_process(traj_tstep):
    process = ControlProcess.__new__(ControlProcess)
    process.traj_tstep = torch.as_tensor(traj_tstep)
    process._dynamic_forecast = None
    return process


class ForecastSamplingTests(unittest.TestCase):
    def test_linear_interpolation_at_rollout_times(self):
        process = make_process([0.02, 0.04, 0.06])
        times = np.array([10.0, 10.1])
        spheres = np.array([[sphere(0.0)], [sphere(1.0)]], dtype=np.float32)
        process.set_dynamic_world_forecast(times, spheres)
        sampled = process.sample_dynamic_forecast(10.0)
        self.assertEqual(sampled.shape, (3, 1, 4))
        np.testing.assert_allclose(sampled[:, 0, 0], [0.2, 0.4, 0.6], rtol=1e-5)
        np.testing.assert_allclose(sampled[:, 0, 3], 0.1, rtol=1e-6)

    def test_out_of_range_times_hold_ends(self):
        process = make_process([0.5, 1.0])
        process.set_dynamic_world_forecast([0.0, 0.1], np.array([[sphere(0.0)], [sphere(1.0)]]))
        np.testing.assert_allclose(process.sample_dynamic_forecast(5.0)[:, 0, 0], [1.0, 1.0])
        np.testing.assert_allclose(process.sample_dynamic_forecast(-5.0)[:, 0, 0], [0.0, 0.0])

    def test_invalid_forecast_rejected_and_clear(self):
        process = make_process([0.02])
        with self.assertRaises(ValueError):
            process.set_dynamic_world_forecast([1.0, 1.0], np.zeros((2, 1, 4)))
        with self.assertRaises(ValueError):
            process.set_dynamic_world_forecast([1.0], np.zeros((1, 4)))
        process.set_dynamic_world_forecast([1.0], np.zeros((1, 1, 4)))
        process.set_dynamic_world_forecast(None, None)
        self.assertIsNone(process._dynamic_forecast)


if __name__ == '__main__':
    unittest.main()
