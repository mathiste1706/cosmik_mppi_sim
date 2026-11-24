import sys, types, math, numpy as np
# Stub the ROS modules so the node class can be built without a ROS graph.
for name in ['rclpy', 'rclpy.node', 'builtin_interfaces', 'builtin_interfaces.msg', 'geometry_msgs', 'geometry_msgs.msg',
             'sensor_msgs', 'sensor_msgs.msg', 'tf2_ros', 'visualization_msgs', 'visualization_msgs.msg']:
    sys.modules[name] = types.ModuleType(name)
class Node:
    def __init__(self, *a): self._p = getattr(self, '_p', {})
    def declare_parameter(self, n, v): self._p.setdefault(n, v)
    def get_parameter(self, n): return types.SimpleNamespace(value=self._p[n])
    def create_publisher(self, *a): return None
    def create_timer(self, *a): return None
    def get_clock(self): return None
sys.modules['rclpy.node'].Node = Node
for mod, names in [('builtin_interfaces.msg', ['Time']), ('geometry_msgs.msg', ['TransformStamped']), ('sensor_msgs.msg', ['JointState']),
                   ('tf2_ros', ['StaticTransformBroadcaster']), ('visualization_msgs.msg', ['Marker', 'MarkerArray'])]:
    for n in names: setattr(sys.modules[mod], n, type(n, (), {'__init__': lambda self, *a, **k: None}))
ns = {}
exec(open('urdf_human').read().split("\ndef main")[0], ns)
UH = ns['UrdfHuman']
URDF = '/root/worspace/cosmik_mppi_sim/ros2_ws/src/storm_pick_place_demo/urdf/human/human.urdf'
def make(**over):
    h = UH.__new__(UH); Node.__init__(h)
    h._p.update({'urdf_path': URDF, 'xyz': [-1.0, 0.20, 0.15], 'sine_amplitudes': [0.15, 0.45], **over})
    h.get_logger = lambda: types.SimpleNamespace(info=print)
    h._publish_placement = lambda *a: None
    UH.__init__(h); return h
# 1) defaults: nothing changes
h0 = make()
assert not h0.block_enabled and not h0.jerk_enabled and [s[0] for s in h0.segments] == ['upperarm_R', 'lowerarm_R']
q0 = h0.joint_positions(12.3); print('default ok, segments', [s[0] for s in h0.segments])
# 2) block: left hand path over one period
h = make(block_enabled=True, jerk_enabled=True, jerk_seed=203)
print('segments', [s[0] for s in h.segments], 'moving', sorted(h.moving_segments))
B = np.array([-0.5, 0.5, 0.3])
def lhand(q):
    L = h.kinematics.link_poses(q); T = h.T_place @ L['lowerarmL']; return T[:3, 3] + T[:3, :3] @ np.array([0, -0.30, 0])
for t in [0, 3.0, 3.5, 4.0, 6.0, 8.0, 8.5, 9.0, 11.0]:
    p = lhand(h.joint_positions(t)); print(f't {t:5.1f} block {h.block_fraction(t):.2f} left hand {p.round(3)} dist to B {np.linalg.norm(p - B):.3f}')
# hand height along the reach
zs = [lhand(h.joint_positions(3.0 + u))[2] for u in np.linspace(0, 1, 11)]; print('hand z during reach', np.round(zs, 3))
# 3) jerks
print('jerks in first 60 s', [(round(s, 2), round(a, 3)) for s, a in h.jerks if s < 60])
s0, a0 = h.jerks[0]
for dt in (0, 0.125, 0.25, 0.375, 0.5): print(f'  bump +{dt}: offset {h.jerk_offset(s0 + dt):.3f}')
gaps = np.diff([s for s, _ in h.jerks]); print(f'mean interval {gaps.mean():.2f} s, n in 230 s: {sum(1 for s, _ in h.jerks if s < 230)}')
# forecast k=0 equals observation; forecast keeps the current bump offset
t = s0 + 0.2
obs = h.joint_positions(t); fc0 = h.joint_positions(t, amp_time=t, jerk_time=t); fc1 = h.joint_positions(t + 0.5, amp_time=t, jerk_time=t)
assert abs(obs['right_shoulder_Z'] - fc0['right_shoulder_Z']) < 1e-12
print('forecast +0.5 s keeps offset', round(fc1['right_shoulder_Z'] - h.joint_positions(t + 0.5)['right_shoulder_Z'], 3), 'rad vs truth')
