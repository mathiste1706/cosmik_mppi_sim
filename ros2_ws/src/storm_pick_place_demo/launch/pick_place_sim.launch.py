import os
import tempfile
from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import Command, EnvironmentVariable, FindExecutable, LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from agimus_demos_common.launch_utils import generate_default_franka_args, generate_include_launch
from agimus_demos_common.static_transform_publisher_node import static_transform_publisher_node


def launch_setup(context: LaunchContext, *args, **kwargs) -> list[LaunchDescriptionEntity]:
    if LaunchConfiguration("arm_id").perform(context) != "fer":
        raise ValueError("This baseline is configured for arm_id:=fer.")
    arm_id = LaunchConfiguration("arm_id")
    armed = LaunchConfiguration("armed")
    with_rviz = LaunchConfiguration("with_rviz")
    armed_param = ParameterValue(armed, value_type=bool)

    # Force simulation mode for this launch file (agimus uses Gazebo).
    set_sim_cfg = [
        SetLaunchConfiguration("use_gazebo", "true"),
        SetLaunchConfiguration("use_rviz", with_rviz),
        SetLaunchConfiguration("robot_ip", ""),
        SetLaunchConfiguration("aux_computer_ip", ""),
        SetLaunchConfiguration("on_aux_computer", "false"),
    ]

    rviz_cfg = PathJoinSubstitution([FindPackageShare("storm_pick_place_demo"), "rviz", "pick_place.rviz"])
    lfc_params = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "linear_feedback_controller_params.yaml"]
    )

    franka_robot_launch = generate_include_launch(
        "franka_common_lfc.launch.py",
        extra_launch_arguments={"rviz_config_path": rviz_cfg, "linear_feedback_controller_params": lfc_params},
    )

    env_xacro = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "urdf", "environment.urdf.xacro"]
    )
    environment_description = ParameterValue(
        Command([PathJoinSubstitution([FindExecutable(name="xacro")]), " ", env_xacro]),
        value_type=str,
    )
    environment_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="environment_publisher",
        output="screen",
        remappings=[("robot_description", "environment_description")],
        parameters=[{"robot_description": environment_description, "use_sim_time": True}],
    )

    arm_id_str = context.perform_substitution(arm_id)
    base_frame = f"{arm_id_str}_link0"
    tf_env = static_transform_publisher_node(
        frame_id=base_frame,
        child_frame_id="robot_attachment_link",
        xyz=(0.0, 0.0, 0.0),
        rot_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    wait_for_non_zero_joints_node = Node(
        package="agimus_demos_common",
        executable="wait_for_non_zero_joints_node",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    obj_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "object_manager_params.yaml"]
    )
    object_manager_node = Node(
        package="storm_pick_place_demo",
        executable="object_manager_node",
        name="object_manager",
        parameters=[{"use_sim_time": True}, obj_yaml],
        output="screen",
    )

    storm_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_agimus_bridge"), "config", "storm_mppi_planner_params.yaml"]
    )
    # The planner consumes the forecast only when prediction is enabled (otherwise: held observation).
    use_prediction = (LaunchConfiguration("human_prediction").perform(context) == "true"
                      and LaunchConfiguration("human_model").perform(context) == "urdf")
    forecast_topic = LaunchConfiguration("forecast_topic").perform(context) if use_prediction else ""
    storm_planner_node = Node(
        package="storm_agimus_bridge",
        executable="storm_mppi_planner_node",
        name="storm_mppi_planner",
        parameters=[storm_yaml, {"use_sim_time": True, "armed": armed_param,
            "storm_task_file": LaunchConfiguration("storm_task_file"),
            "max_joint_ref_error_rad": 0.1,
            "collision_markers_topic": LaunchConfiguration("collision_topic"),
            "collision_forecast_topic": forecast_topic,
            "mppi_multimodal": LaunchConfiguration("mppi_multimodal").perform(context) == "true",
            "auto_discover_collision_markers_topic": False,
            "gripper_joint_states_topic": "/joint_states"}],
        output="screen",
        remappings=[("robot_description", "robot_description_with_collision")],
    )

    task_yaml = PathJoinSubstitution(
        [FindPackageShare("storm_pick_place_demo"), "config", "pick_place_task_manager_params.yaml"]
    )
    # Empty keeps the YAML value (10 s): time on one target before switching to the other.
    target_timeout = LaunchConfiguration("target_timeout_sec").perform(context).strip()
    # The task YAML names the node, which outranks both launch dicts (written as "/**") and -p here;
    # a later file with the same node-specific key does win.
    task_override_files = []
    if target_timeout:
        fd, override_path = tempfile.mkstemp(prefix="pick_place_task_override_", suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump({"pick_place_task_manager": {"ros__parameters": {
                "target_timeout_sec": float(target_timeout)}}}, f)
        task_override_files.append(override_path)
    task_node = Node(
        package="storm_pick_place_demo",
        executable="pick_place_task_manager_node",
        name="pick_place_task_manager",
        remappings=[("/fer_gripper/gripper_action", "/gripper_action_controller/gripper_cmd")],
        parameters=[{"use_sim_time": True}, task_yaml, {"armed": armed_param, "auto_start": armed_param},
                    *task_override_files],
        output="screen",
    )

    start_stack_on_ready = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=wait_for_non_zero_joints_node,
            on_exit=[
                object_manager_node,
                storm_planner_node,
                task_node,
            ],
        )
    )

    # Ensure STORM is importable from ROS nodes and from STORM's spawned MPPI optimization process.
    storm_root = os.environ.get("STORM_ROOT", "")
    if not (Path(storm_root) / "storm_kit").is_dir():
        raise RuntimeError("Set STORM_ROOT by sourcing scripts/env.sh first.")
    set_storm_env = [
        SetEnvironmentVariable("STORM_ROOT", storm_root),
        SetEnvironmentVariable(
            "PYTHONPATH",
            [TextSubstitution(text=storm_root), TextSubstitution(text=":"), EnvironmentVariable("PYTHONPATH")],
        ),
    ]

    synthetic_human = LaunchConfiguration("synthetic_human").perform(context) == "true"
    human_model = LaunchConfiguration("human_model").perform(context)
    human_nodes = []
    if synthetic_human and human_model == "capsule":
        human_nodes.append(
            Node(package="storm_pick_place_demo", executable="synthetic_capsules",
                 name="synthetic_capsules", output="screen",
                 parameters=[{"use_sim_time": True, "topic": LaunchConfiguration("collision_topic")}]))
    elif synthetic_human and human_model == "urdf":
        arm_frequency = float(LaunchConfiguration("human_arm_frequency").perform(context))
        if not arm_frequency > 0.0:
            raise ValueError("human_arm_frequency must be > 0 Hz")
        human_urdf = os.path.join(
            get_package_share_directory("storm_pick_place_demo"), "urdf", "human", "human.urdf")
        with open(human_urdf, encoding="utf-8") as f:
            human_description = f.read()
        human_nodes += [
            Node(package="robot_state_publisher", executable="robot_state_publisher",
                 name="robot_state_publisher", namespace="human", output="screen",
                 parameters=[{"robot_description": human_description, "frame_prefix": "human/",
                              "use_sim_time": True}]),
            Node(package="storm_pick_place_demo", executable="urdf_human",
                 name="urdf_human", output="screen",
                 parameters=[{"use_sim_time": True, "topic": LaunchConfiguration("collision_topic"),
                              "forecast_topic": LaunchConfiguration("forecast_topic"),
                              "motion": LaunchConfiguration("human_motion"),
                              "sine_period": 1.0 / arm_frequency,
                              "urdf_path": human_urdf}]),
        ]

    return [
        *set_storm_env,
        *set_sim_cfg,
        *human_nodes,
        franka_robot_launch,
        environment_publisher_node,
        tf_env,
        wait_for_non_zero_joints_node,
        start_stack_on_ready,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("storm_task_file", default_value="franka_reacher_cat_real.yml"),
        DeclareLaunchArgument("synthetic_human", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument(
            "human_model",
            default_value="capsule",
            description="Synthetic human: moving cylinder (baseline) or RT-COSMIK URDF.",
            choices=["capsule", "urdf"],
        ),
        DeclareLaunchArgument(
            "human_motion",
            default_value="sine",
            description="URDF human only: right arm moving up/down as a sinusoid, or a fixed posture.",
            choices=["sine", "static"],
        ),
        DeclareLaunchArgument(
            "target_timeout_sec",
            default_value="",
            description="Seconds on one target before switching to the other (empty: task YAML value, 10 s).",
        ),
        DeclareLaunchArgument(
            "human_arm_frequency",
            default_value="0.25",
            description="URDF human with human_motion:=sine: frequency of the arm's up/down motion [Hz].",
        ),
        DeclareLaunchArgument(
            "human_prediction",
            default_value="true",
            description="URDF human only: MPPI evaluates each rollout step against the forecast human "
            "geometry (true) or holds the latest observed geometry over the horizon (false).",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "mppi_multimodal",
            default_value="false",
            description="Multimodal MPPI: direct/up/down families scored on the same (forecast) costs; "
            "MPPI executes the best family instead of averaging them.",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument("collision_topic", default_value="/rtcosmik/collision_markers"),
        DeclareLaunchArgument("forecast_topic", default_value="/rtcosmik/collision_forecast"),
        DeclareLaunchArgument(
            "armed",
            default_value="true",
            description="Arm the demo (publish commands and operate gripper).",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "with_rviz",
            default_value="true",
            description="Launch RViz2 (disable for headless smoke tests).",
            choices=["true", "false"],
        ),
    ]
    return LaunchDescription(declared_arguments + generate_default_franka_args() + [OpaqueFunction(function=launch_setup)])
