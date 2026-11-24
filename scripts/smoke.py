#!/usr/bin/env python3
"""Run and monitor only this simulation's process group; write evidence as JSON.

Use via smoke.sh. Arrival counts require the 6 cm position tolerance (orientation is
recorded, not required); timeout alternations alone cannot pass the reaching check.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray
from linear_feedback_controller_msgs.msg import Control
from tf2_ros import Buffer, TransformListener, TransformException


class Monitor(Node):
    def __init__(self):
        super().__init__('handoff_smoke_monitor')
        self.counts = Counter()
        self.sim_time = 0.0
        self.first_goal_time = None
        self.goal = None
        self.target = None
        self.target_order = []
        self.arrivals = Counter()
        self.visit_reached = False
        self.minimum = {'A': float('inf'), 'B': float('inf')}
        self.nonfinite = False
        self.joint_min = None
        self.joint_max = None
        self.human_x = []
        self.human_capsules = []  # (ns, endpoint a, endpoint b, radius) of the latest human frame
        self.clearance = {}  # human ns -> minimum surface distance to any robot collision sphere [m]
        self.clearance_samples = 0
        self.trace = []  # (sim time, ee xyz, forearm endpoints, overall clearance) for offline analysis
        self.penetration_samples = 0
        spheres = Path(os.environ['STORM_ROOT'])/'content/assets/robot/fer_real/fer_spheres.yml'
        with spheres.open() as f:
            self.robot_spheres = {link.replace('panda', 'fer', 1): (np.array([s['center'] for s in values]),
                                                                    np.array([s['radius'] for s in values]))
                                  for link, values in yaml.safe_load(f)['collision_spheres'].items()}
        self.pose_trace = []  # (sim time, target 0=A 1=B, position error [m], orientation error [rad], tool z xyz)
        self.self_clearance = []  # (sim time, minimum robot self-clearance [m]) at 20 Hz
        self.self_worst_pairs = Counter()  # closest link pair at each self-collision-margin violation
        self.joint_trace = []  # (sim time, q1..q7, qd1..qd7) at 20 Hz, to see which joint sits at its limit
        self.last_q = self.last_qd = None
        self.self_cost = self.load_self_collision_model()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joints_cb, qos_profile_sensor_data)
        self.create_subscription(Control, '/control', self.control_cb, qos_profile_sensor_data)
        self.create_subscription(MarkerArray, '/rtcosmik/collision_markers', self.human_cb, 1)
        self.create_timer(0.05, self.sample)

    def clock_cb(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.counts['clock'] += 1

    def goal_cb(self, msg):
        self.goal = msg.pose
        self.counts['goals'] += 1
        if self.first_goal_time is None:
            self.first_goal_time = self.sim_time
        target = 'A' if msg.pose.position.y < 0 else 'B'
        if target != self.target:
            self.target = target
            self.target_order.append({'target': target, 'sim_time': self.sim_time})
            self.visit_reached = False

    def joints_cb(self, msg):
        self.counts['joint_states'] += 1
        values = np.asarray(msg.position)
        self.nonfinite |= not np.all(np.isfinite(values))
        by_name = dict(zip(msg.name, msg.position))
        if all(f'fer_joint{i}' in by_name for i in range(1, 8)):
            q = np.array([by_name[f'fer_joint{i}'] for i in range(1, 8)])
            self.joint_min = q if self.joint_min is None else np.minimum(self.joint_min, q)
            self.joint_max = q if self.joint_max is None else np.maximum(self.joint_max, q)
            self.last_q = q
            vel = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}
            self.last_qd = np.array([vel.get(f'fer_joint{i}', float('nan')) for i in range(1, 8)])

    def control_cb(self, msg):
        self.counts['control'] += 1
        values = [*msg.feedforward.data, *msg.feedback_gain.data, *msg.initial_state.joint_state.position]
        self.nonfinite |= not np.all(np.isfinite(values))

    def human_cb(self, msg):
        self.counts['human_frames'] += 1
        if msg.markers:
            self.human_x.append(msg.markers[0].pose.position.x)
        capsules = []
        for m in msg.markers:
            if m.type != m.CYLINDER or m.header.frame_id not in ('world', 'fer_link0'):
                continue  # world and fer_link0 coincide in this simulation.
            o = m.pose.orientation
            axis = np.array([2*(o.x*o.z+o.y*o.w), 2*(o.y*o.z-o.x*o.w), 1-2*(o.x*o.x+o.y*o.y)])
            radius = 0.5*max(m.scale.x, m.scale.y)
            half = max(0.5*m.scale.z - radius, 0.0)  # Same centre line as the planner's sphere chain.
            c = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
            capsules.append((m.ns, c - half*axis, c + half*axis, radius))
        self.human_capsules = capsules

    def measure_clearance(self):
        """Minimum surface distance between robot collision spheres and human capsules."""
        if not self.human_capsules:
            return
        centres, radii = [], []
        for link, (local, r) in self.robot_spheres.items():
            try:
                tf = self.tf.lookup_transform('fer_link0', link, Time()).transform
            except TransformException:
                return
            q, t = tf.rotation, tf.translation
            R = np.array([[1-2*(q.y*q.y+q.z*q.z), 2*(q.x*q.y-q.z*q.w), 2*(q.x*q.z+q.y*q.w)],
                          [2*(q.x*q.y+q.z*q.w), 1-2*(q.x*q.x+q.z*q.z), 2*(q.y*q.z-q.x*q.w)],
                          [2*(q.x*q.z-q.y*q.w), 2*(q.y*q.z+q.x*q.w), 1-2*(q.x*q.x+q.y*q.y)]])
            centres.append(local @ R.T + [t.x, t.y, t.z])
            radii.append(r)
        centres, radii = np.concatenate(centres), np.concatenate(radii)
        overall = float('inf')
        for ns, a, b, radius in self.human_capsules:
            ab = b - a
            s = np.clip(((centres - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            d = float(np.min(np.linalg.norm(centres - (a + s[:, None]*ab), axis=1) - radii) - radius)
            self.clearance[ns] = min(self.clearance.get(ns, float('inf')), d)
            overall = min(overall, d)
        self.last_clearance = overall
        self.clearance_samples += 1
        self.penetration_samples += overall < 0.0

    @staticmethod
    def load_self_collision_model():
        """The explicit self-collision model MPPI uses (spheres, SRDF-excluded pairs, margin), from the task YAML."""
        import torch
        from storm_kit.mpc.cost.explicit_self_collision_cost import ExplicitSelfCollisionCost
        task = Path(os.environ['STORM_ROOT'])/'content/configs/mpc/franka_reacher_cat_real.yml'
        exp = yaml.safe_load(task.read_text())
        cfg = exp['cost']['robot_self_collision']
        cost = ExplicitSelfCollisionCost(
            weight=1.0, robot_params={'robot_collision_params': exp['model']['robot_collision_params']},
            tensor_args={'device': torch.device('cpu'), 'dtype': torch.float64},
            distance_threshold=cfg['distance_threshold'], srdf_path=cfg['srdf_path'])
        links = exp['model']['robot_collision_params']['link_objs']
        # STORM link names use the "panda" prefix; this robot's TF frames use "fer".
        frames = [link.replace('panda', 'fer', 1) for link in links]
        return {'cost': cost, 'frames': frames, 'links': links, 'margin': float(cfg['distance_threshold'])}

    def measure_self_clearance(self):
        """Minimum surface distance between the robot's own collision spheres, as MPPI's self-collision check."""
        import torch
        model = self.self_cost
        pos, rot = [], []
        for frame in model['frames']:
            try:
                tf = self.tf.lookup_transform('fer_link0', frame, Time()).transform
            except TransformException:
                return
            q, t = tf.rotation, tf.translation
            rot.append([[1-2*(q.y*q.y+q.z*q.z), 2*(q.x*q.y-q.z*q.w), 2*(q.x*q.z+q.y*q.w)],
                        [2*(q.x*q.y+q.z*q.w), 1-2*(q.x*q.x+q.z*q.z), 2*(q.y*q.z-q.x*q.w)],
                        [2*(q.x*q.z-q.y*q.w), 2*(q.y*q.z+q.x*q.w), 1-2*(q.x*q.x+q.y*q.y)]])
            pos.append([t.x, t.y, t.z])
        c = model['cost']
        pos = torch.tensor(pos, dtype=torch.float64)
        rot = torch.tensor(rot, dtype=torch.float64)
        world = (rot[c.parents] @ c.centers.unsqueeze(-1)).squeeze(-1) + pos[c.parents]
        d = torch.linalg.vector_norm(world[c.first] - world[c.second], dim=-1) - c.radii[c.first] - c.radii[c.second]
        k = int(torch.argmin(d))
        clearance = float(d[k])
        self.self_clearance.append((self.sim_time, clearance))
        if clearance < model['margin']:
            pair = sorted((model['links'][int(c.parents[c.first[k]])], model['links'][int(c.parents[c.second[k]])]))
            self.self_worst_pairs['/'.join(pair)] += 1

    def sample(self):
        if self.goal is None:
            return
        self.measure_clearance()
        self.measure_self_clearance()
        if self.last_q is not None:
            self.joint_trace.append([self.sim_time, *self.last_q, *self.last_qd])
        try:
            tf = self.tf.lookup_transform('fer_link0', 'fer_hand_tcp', Time()).transform
        except TransformException:
            return
        self.counts['ee_samples'] += 1
        p, g = tf.translation, self.goal.position
        forearm = next((c for c in self.human_capsules if c[0].endswith('lowerarm_R')), None)
        # Without a URDF human the forearm columns are NaN; the end-effector path is still recorded.
        ends = [*forearm[1], *forearm[2]] if forearm is not None else [float('nan')] * 6
        self.trace.append([self.sim_time, p.x, p.y, p.z, *ends, getattr(self, 'last_clearance', float('nan'))])
        err = math.sqrt((p.x-g.x)**2 + (p.y-g.y)**2 + (p.z-g.z)**2)
        self.minimum[self.target] = min(self.minimum[self.target], err)
        q, r = tf.rotation, self.goal.orientation
        qa = np.array([q.x, q.y, q.z, q.w]); ra = np.array([r.x, r.y, r.z, r.w])
        cosine = abs(float(qa @ ra)) / (np.linalg.norm(qa) * np.linalg.norm(ra))
        angle = 2 * math.acos(float(np.clip(cosine, 0, 1)))
        # Gripper approach axis (tool z) in the base frame, to see how the orientation is off.
        z_axis = [2*(q.x*q.z+q.y*q.w), 2*(q.y*q.z-q.x*q.w), 1-2*(q.x*q.x+q.y*q.y)]
        self.pose_trace.append([self.sim_time, 0.0 if self.target == 'A' else 1.0, err, angle, *z_axis])
        # Arrival = position only (orientation is recorded in pose_trace, not required), as the task manager.
        if err < 0.06 and not self.visit_reached:
            self.arrivals[self.target] += 1
            self.visit_reached = True


def orientation_summary(node):
    """Per target: smallest orientation error while within the 6 cm arrival radius (arrival needs < 2.0 rad)."""
    if not node.pose_trace:
        return None
    P = np.asarray(node.pose_trace)
    out = {}
    for code, name in ((0.0, 'A'), (1.0, 'B')):
        near = (P[:, 1] == code) & (P[:, 2] < 0.06)
        if not near.any():
            out[name] = {'samples_within_6cm': 0}
            continue
        k = np.argmin(np.where(near, P[:, 3], np.inf))
        out[name] = {'samples_within_6cm': int(near.sum()),
                     'min_orientation_error_rad': float(P[k, 3]),
                     'median_orientation_error_rad': float(np.median(P[near, 3])),
                     'tool_z_axis_at_best': P[k, 4:7].round(3).tolist()}
    return out


def self_collision_summary(node):
    if not node.self_clearance:
        return None
    c = np.array([v for _, v in node.self_clearance])
    margin = node.self_cost['margin']
    return {'min_clearance_m': float(c.min()), 'samples': int(c.size),
            'penetration_samples': int(np.sum(c < 0.0)),
            'margin_violation_samples': int(np.sum(c < margin)), 'margin_m': margin,
            'closest_pairs_when_violating': dict(node.self_worst_pairs.most_common(5))}


def arm_crossings(trace):
    """Each time the end effector crosses the arm's plane (y=0): above, below or beside the forearm."""
    if not trace:
        return []
    T = np.asarray(trace)
    t, ee, a, b, clearance = T[:, 0], T[:, 1:4], T[:, 4:7], T[:, 7:10], T[:, 10]
    out = []
    for i in np.where(np.sign(ee[1:, 1]) * np.sign(ee[:-1, 1]) < 0)[0]:
        lo, hi = sorted([a[i, 0], b[i, 0]])
        if lo <= ee[i, 0] <= hi and b[i, 0] != a[i, 0]:
            z_arm = a[i, 2] + (ee[i, 0] - a[i, 0]) / (b[i, 0] - a[i, 0]) * (b[i, 2] - a[i, 2])
            side = 'above' if ee[i, 2] > z_arm else 'below'
        else:
            z_arm, side = None, 'beside'
        out.append({'sim_time': float(t[i]), 'side': side, 'ee': ee[i].round(3).tolist(),
                    'forearm_z_at_ee_x': None if z_arm is None else round(float(z_arm), 3),
                    'clearance_m': round(float(clearance[i]), 3)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', choices=['true', 'false'], default='true')
    parser.add_argument('--human-model', choices=['capsule', 'urdf'], default='capsule')
    parser.add_argument('--human-motion', choices=['sine', 'static'], default='sine')
    parser.add_argument('--human-prediction', choices=['true', 'false'], default='true')
    parser.add_argument('--mppi-multimodal', choices=['true', 'false'], default='false')
    parser.add_argument('--human-arm-frequency', type=float, default=0.25, help='Hz, URDF human sine motion')
    parser.add_argument('--target-timeout', type=float, default=None,
                        help='seconds on one target before switching (default: task YAML, 10 s)')
    parser.add_argument('--seconds', type=float, default=50)
    parser.add_argument('--wall-limit', type=float, default=240)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Monitor()
    started = time.monotonic()
    with (args.output/'launch.log').open('w') as log:
        proc = subprocess.Popen([str(Path(__file__).with_name('run.sh')), f'synthetic_human:={args.human}',
                                 f'human_model:={args.human_model}', f'human_motion:={args.human_motion}',
                                 f'human_prediction:={args.human_prediction}',
                                 f'mppi_multimodal:={args.mppi_multimodal}',
                                 f'human_arm_frequency:={args.human_arm_frequency}',
                                 *([f'target_timeout_sec:={args.target_timeout}'] if args.target_timeout else [])],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while time.monotonic() - started < args.wall_limit and proc.poll() is None:
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.first_goal_time is not None and node.sim_time-node.first_goal_time >= args.seconds:
                    break
        finally:
            # Only terminate the process group that this script created.
            for sig, wait in [(signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 2)]:
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    continue
                # launch may exit before a spawned child; follow up on the group.
                if sig == signal.SIGKILL:
                    break
                time.sleep(0.5)
    log_text = (args.output/'launch.log').read_text(errors='replace')
    reached = node.arrivals['A'] > 0 and node.arrivals['B'] > 0
    live = node.counts['control'] > 100 and node.counts['ee_samples'] > 100
    duration = 0.0 if node.first_goal_time is None else node.sim_time-node.first_goal_time
    # The URDF human's first marker (torso) never moves; it only has to be present.
    moved = args.human_model == 'urdf' or (len(node.human_x) > 0 and np.ptp(node.human_x) > 0.1)
    fixture_ok = args.human == 'false' or (len(node.human_x) > 100 and moved)
    dynamic_updates = log_text.count('Updated dynamic collision spheres from live marker/pose inputs.')
    ingestion_ok = args.human == 'false' or dynamic_updates > 1
    profile_loaded = ("Loaded STORM task config: storm_task_file='franka_reacher_cat_real.yml'" in log_text
                      and 'mppi.cat_primitive_collision.enabled=True' in log_text)
    result = {
        'pass': bool(reached and live and fixture_ok and ingestion_ok and profile_loaded
                     and not node.nonfinite and duration >= args.seconds),
        'scope': 'Simulation transport and reaching smoke check; does not certify collision avoidance or prediction.',
        'human': args.human, 'wall_seconds': time.monotonic()-started,
        'cat_profile_loaded': profile_loaded, 'dynamic_geometry_update_logs': dynamic_updates,
        'simulated_seconds_after_first_goal': duration,
        'messages': dict(node.counts), 'target_visits': node.target_order,
        'spatial_arrivals': dict(node.arrivals),
        'minimum_target_error_m': {key: value if math.isfinite(value) else None for key, value in node.minimum.items()},
        'timeout_events': log_text.count('[task-event] target-timeout('),
        'nonfinite_state_or_command': bool(node.nonfinite),
        'joint_excursion_rad': None if node.joint_min is None else (node.joint_max-node.joint_min).tolist(),
        'human_model': args.human_model, 'human_motion': args.human_motion,
        'human_prediction': args.human_prediction, 'mppi_multimodal': args.mppi_multimodal,
        'human_arm_frequency_hz': args.human_arm_frequency, 'target_timeout_sec': args.target_timeout,
        'mppi_family_switches': [line.split('[mppi-family] ')[1] for line in log_text.splitlines()
                                 if '[mppi-family]' in line],
        'arm_crossings': arm_crossings(node.trace),
        # Surface distance, robot collision spheres to human capsules, sampled at 20 Hz.
        'min_human_clearance_m': min(node.clearance.values(), default=None),
        'min_clearance_by_human_segment_m': dict(sorted(node.clearance.items(), key=lambda kv: kv[1])[:4]),
        'clearance_samples': node.clearance_samples,
        'penetration_samples': node.penetration_samples,
        # Robot self-collision, same spheres, SRDF exclusions and margin as MPPI's explicit self check.
        'self_collision': self_collision_summary(node),
        'orientation_near_targets': orientation_summary(node),
        'task_events': [line for line in log_text.splitlines() if '[task-event]' in line],
    }
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    if node.trace:
        # columns: t, ee x y z, forearm a xyz, forearm b xyz, clearance
        np.save(args.output/'trace.npy', np.asarray(node.trace))
    if node.pose_trace:
        np.save(args.output/'pose.npy', np.asarray(node.pose_trace))
    if node.self_clearance:
        # columns: sim time, robot self-clearance [m]
        np.save(args.output/'self_clearance.npy', np.asarray(node.self_clearance))
    if node.joint_trace:
        # columns: sim time, q1..q7 [rad], qd1..qd7 [rad/s]
        np.save(args.output/'joints.npy', np.asarray(node.joint_trace))
    print(json.dumps({k: v for k, v in result.items() if k != 'task_events'}, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
