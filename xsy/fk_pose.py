# -*- coding:utf-8 -*-

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pose Authoring Tool using Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
import carb

"""
Usage: python pose_adjust.py  --device cpu
"""

UNITREE_MODEL_DIR = "/home/simon/code/unitree_rl_lab/unitree_model"

G1_29DOF_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ), 
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True, 
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ), 
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": 0.0000,
            "right_hip_pitch_joint": 0.1712,
            "waist_yaw_joint": -0.3491,
            "left_hip_roll_joint": 0.0873,
            "right_hip_roll_joint": -0.1902,
            "waist_roll_joint": 0.0000,
            "left_hip_yaw_joint": 0.3910,
            "right_hip_yaw_joint": -0.3895,
            "waist_pitch_joint": 0.0000,
            "left_knee_joint": 0.0000,
            "right_knee_joint": 0.0874,
            "left_shoulder_pitch_joint": -0.1745,
            "right_shoulder_pitch_joint": 0.2618,
            "left_ankle_pitch_joint": 0.0000,
            "right_ankle_pitch_joint": -0.1745,
            "left_shoulder_roll_joint": 0.3491,
            "right_shoulder_roll_joint": -0.1745,
            "left_ankle_roll_joint": 0.0000,
            "right_ankle_roll_joint": 0.1473,
            "left_shoulder_yaw_joint": 0.5236,
            "right_shoulder_yaw_joint": -0.5236,
            "left_elbow_joint": 0.8727,
            "right_elbow_joint": 1.2217,
            "left_wrist_roll_joint": 0.0000,
            "right_wrist_roll_joint": 0.0000,
            "left_wrist_pitch_joint": 0.0000,
            "right_wrist_pitch_joint": 0.0000,
            "left_wrist_yaw_joint": 0.0000,
            "right_wrist_yaw_joint": -0.2618,
        }, 
    ),
    actuators={
        "body_actuator": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0, 
            damping=0.0,
        ),
    },
)

class PoseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = G1_29DOF_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, 
        gravity=(0.0, 0.0, 0.0)
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = PoseSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()

    robot = scene["robot"]
    
    # IMPORTANT: Reset robot to initial joint positions after sim.reset()
    # This ensures the joint_pos defined in init_state are actually applied
    print("[INFO]: Setting initial joint positions...")
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    
    # Get keyboard interface using carb
    import omni.appwindow
    appwindow = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    
    print("[INFO]: Press 'P' to print all joint positions")
    print("[INFO]: Press 'R' to reset to initial pose")
    print("[INFO]: Press 'ESC' to quit")
    
    # Track previous keyboard state to detect key press
    p_key_pressed_last = False
    r_key_pressed_last = False

    while simulation_app.is_running():
        # Check for P key press
        p_key_pressed = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.P)
        
        if p_key_pressed and not p_key_pressed_last:
            print("\n" + "="*80)
            print("Current Joint Positions:")
            print("="*80)
            
            # Get joint names and positions
            joint_names = robot.data.joint_names
            joint_pos = robot.data.joint_pos[0].cpu().numpy()  # Get first env
            
            # Print in a formatted way
            for i, (name, pos) in enumerate(zip(joint_names, joint_pos)):
                print(f"{i:2d}. {name:30s}: {pos:8.4f} rad ({pos*180/3.14159:8.2f}°)")
            
            print("="*80)
            
            # Also print in dictionary format for easy copying
            print("\nDictionary format:")
            print("{")
            for name, pos in zip(joint_names, joint_pos):
                print(f'    "{name}": {pos:.4f},')
            print("}")
            print("="*80 + "\n")
        
        p_key_pressed_last = p_key_pressed
        
        # Check for R key press to reset to initial pose
        r_key_pressed = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.R)
        
        if r_key_pressed and not r_key_pressed_last:
            print("[INFO]: Resetting to initial pose...")
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        
        r_key_pressed_last = r_key_pressed
        
        # Don't write joint states in the loop - let the physics/actuators handle it
        # Only write data to sync with simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

if __name__ == "__main__":
    main()
    simulation_app.close()