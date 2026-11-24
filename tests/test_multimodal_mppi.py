import unittest

import torch

from storm_kit.mpc.control.mppi import MPPI

HORIZON, D = 8, 2


class _RobotModel:
    """End-effector z moves with joint 0, x with joint 1 (constant Jacobian)."""

    def compute_fk_and_jacobian(self, q, qd, link_name):
        J = torch.tensor([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]]).repeat(q.shape[0], 1, 1)
        return torch.zeros(q.shape[0], 3), torch.eye(3).repeat(q.shape[0], 1, 1), J, torch.zeros_like(J)


class _Dynamics:
    n_dofs = D
    ee_link_name = 'ee'
    robot_model = _RobotModel()


class _TwoWayRollout:
    """Two equally good homotopies: joint-0 acceleration of +1 (up) or -1 (down) at every step.

    Averaging them (single-mode MPPI) lands near 0, which is the worst plan.
    """
    dynamics_model = _Dynamics()

    def __call__(self, state, act_seq):
        a0 = act_seq[..., 0]
        costs = torch.minimum((a0 - 1.0) ** 2, (a0 + 1.0) ** 2) + 0.1 * act_seq[..., 1] ** 2
        return {'costs': costs, 'actions': act_seq, 'ee_pos_seq': torch.zeros(act_seq.shape[0], HORIZON, 3),
                'rollout_time': 0.0}


def make_mppi(multimodal=None, particles=200):
    return MPPI(d_action=D, horizon=HORIZON, init_cov=0.3, init_mean=torch.zeros(HORIZON, D),
                base_action='null', beta=0.1, num_particles=particles, step_size_mean=0.9, step_size_cov=0.0,
                alpha=1, gamma=1.0, kappa=0.0, n_iters=1, action_lows=-3 * torch.ones(D),
                action_highs=3 * torch.ones(D), null_act_frac=0.0, rollout_fn=_TwoWayRollout(),
                cov_type='full_AxA', visual_traj='ee_pos_seq', sample_params={'type': 'random', 'fixed_samples': False, 'seed': 0},
                multimodal=multimodal)


STATE = torch.zeros(1, 2 * D)


class MultimodalMPPITests(unittest.TestCase):
    def test_disabled_keeps_single_mode_sampling(self):
        controller = make_mppi()
        self.assertFalse(controller._mm_enabled)
        self.assertEqual(controller.sample_actions(STATE).shape, (200, HORIZON, D))
        self.assertFalse(hasattr(controller, 'mode_means'))

    def test_families_are_labelled_and_fill_the_batch(self):
        controller = make_mppi({'enabled': True, 'seed_accel': 1.0, 'jacobian_damping': 1e-9})
        acts = controller.sample_actions(STATE)
        self.assertEqual(acts.shape, (200, HORIZON, D))
        counts = torch.bincount(controller._mm_labels, minlength=3).tolist()
        self.assertEqual(sum(counts), 200)
        self.assertEqual(counts[1], counts[2])
        self.assertGreater(min(counts), 2)
        # The up/down seeds themselves are in the batch: joint 0 = +1 / -1 during the first quarter.
        up = acts[controller._mm_labels == 1]
        down = acts[controller._mm_labels == 2]
        self.assertTrue(torch.any(torch.all(torch.isclose(up[:, :2, 0], torch.ones(2)), dim=1)))
        self.assertTrue(torch.any(torch.all(torch.isclose(down[:, :2, 0], -torch.ones(2)), dim=1)))

    def test_families_keep_distinct_homotopies_and_execute_one(self):
        torch.manual_seed(0)
        multi = make_mppi({'enabled': True, 'seed_accel': 1.0, 'seed_sample_fraction': 1.0, 'jacobian_damping': 1e-9})
        for _ in range(15):
            multi.optimize(STATE.clone(), shift_steps=0)
        # Each family converges to its own optimum; the executed plan is one of them, not their average.
        executed = multi.mean_action[:2, 0].mean().item()
        self.assertGreater(abs(executed), 0.8)
        self.assertGreater(multi.mode_means[1, :2, 0].mean().item(), 0.8)
        self.assertLess(multi.mode_means[2, :2, 0].mean().item(), -0.8)
        # The executed plan is the mean of the best-scoring family.
        active = ['direct', 'up', 'down'].index(multi.multimodal_debug['active'])
        torch.testing.assert_close(multi.mean_action, multi.mode_means[active])

    def test_hysteresis_requires_a_clear_improvement(self):
        controller = make_mppi({'enabled': True, 'switch_margin': 0.5, 'switch_patience': 2})
        controller._mm_labels = torch.cat([torch.zeros(4), torch.ones(2), 2 * torch.ones(2)]).long()
        actions = torch.zeros(8, HORIZON, D)
        trajectories = {'actions': actions, 'ee_pos_seq': torch.zeros(8, HORIZON, 3)}
        # Family 'up' slightly better than 'direct' (active): no switch.
        trajectories['costs'] = torch.tensor([10.0] * 4 + [9.0] * 2 + [20.0] * 2).unsqueeze(1).repeat(1, HORIZON)
        controller._update_distribution(trajectories)
        self.assertEqual(controller.multimodal_debug['active'], 'direct')
        # Clearly better once: not yet (patience 2); clearly better twice in a row: switch.
        trajectories['costs'] = torch.tensor([10.0] * 4 + [2.0] * 2 + [20.0] * 2).unsqueeze(1).repeat(1, HORIZON)
        controller._update_distribution(trajectories)
        self.assertEqual(controller.multimodal_debug['active'], 'direct')
        controller._update_distribution(trajectories)
        self.assertEqual(controller.multimodal_debug['active'], 'up')
        self.assertEqual(controller.multimodal_debug['switches'], 1)

    def test_shift_and_reset_cover_every_family(self):
        controller = make_mppi({'enabled': True})
        controller.mode_means[:] = torch.arange(HORIZON, dtype=torch.float32)[None, :, None]
        controller._shift(2)
        self.assertTrue(torch.all(controller.mode_means[:, :-2, 0] == torch.arange(2, HORIZON)))
        self.assertTrue(torch.all(controller.mode_means[:, -2:] == 0))
        controller.reset_mean()
        self.assertTrue(torch.all(controller.mode_means == 0))


if __name__ == '__main__':
    unittest.main()
