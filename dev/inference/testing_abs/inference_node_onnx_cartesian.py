#!/usr/bin/env python3
"""
ROS2 Inference Node for LeRobot Deployment - Cartesian Mode (ONNX+TensorRT)
Absolute Action Mode

This node subscribes to sensor topics, runs inference using ONNX Runtime with TensorRT,
and publishes actions as TwistStamped.

Includes image preprocessing matching the training pipeline.
Supports Cartesian State (via Forward Kinematics) and Cartesian Actions (TwistStamped).
"""

import sys
import time
import threading
import argparse
import json
import datetime
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import deque

import numpy as np
import cv2
import torch
import torchvision
import onnxruntime as ort
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header


class ONNXTensorRTInference:
    """
    ONNX Runtime with TensorRT backend inference for Cartesian model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        onnx_dir: str,
        device: str = "cuda",
        debug_dir: Optional[Path] = None,
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.onnx_dir = Path(onnx_dir)
        self.debug_dir = debug_dir
        self.debug_step = 0

        # Load ONNX configuration
        config_path = self.onnx_dir / "onnx_config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load image_features from the original model config to determine camera order
        self._load_image_features_config()

        print(f"Loaded ONNX config:")
        print(f"  Horizon: {self.config['horizon']}")
        print(f"  N obs steps: {self.config['n_obs_steps']}")
        print(f"  N action steps: {self.config['n_action_steps']}")
        print(f"  Action dim: {self.config['action_dim']}")
        print(f"  State dim: {self.config['state_dim']}")
        print(f"  Inference steps: {self.config['num_inference_steps']}")
        print(f"  Scheduler: {self.config['noise_scheduler_type']}")

        # Setup ONNX Runtime sessions with TensorRT
        self._setup_onnx_sessions()

        # Setup noise scheduler (DDIM)
        self._setup_scheduler()

        # Setup normalization
        self._setup_normalization()

        # Image preprocessing parameters
        self.crop_box = (260, 135, 224, 224)  # (x, y, w, h) for top view
        self.target_size = (224, 224)

        # Setup center crop if needed
        if self.config.get("crop_shape"):
            self.center_crop = torchvision.transforms.CenterCrop(
                self.config["crop_shape"]
            )
        else:
            self.center_crop = None

        # Left arm joint names
        self.left_arm_joints = [
            "la_shoulder_pan_joint",
            "la_shoulder_lift_joint",
            "la_elbow_joint",
            "la_wrist_1_joint",
            "la_wrist_2_joint",
            "la_wrist_3_joint",
        ]

        # Observation and action queues
        self._queues = {
            "action": deque(maxlen=self.config["n_action_steps"]),
            "observation.state": deque(maxlen=self.config["n_obs_steps"]),
            "observation.images": deque(maxlen=self.config["n_obs_steps"]),
        }

        # Metadata for ROS node
        self.input_mode = "vision_pos"
        self.output_mode = "cartesian"

    def reset(self):
        """Clear observation and action queues."""
        self._queues["action"].clear()
        self._queues["observation.state"].clear()
        self._queues["observation.images"].clear()

    def _setup_onnx_sessions(self):
        """Setup ONNX Runtime sessions with TensorRT provider."""
        print("\nInitializing ONNX Runtime sessions with TensorRT...")
        available_providers = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available_providers and self.device == "cuda":
            trt_options = {
                "device_id": 0,
                "trt_max_workspace_size": 2147483648,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self.onnx_dir / "trt_engines"),
            }
            providers.append(("TensorrtExecutionProvider", trt_options))
        elif "CUDAExecutionProvider" in available_providers and self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        encoder_path = self.onnx_dir / "rgb_encoder.onnx"
        self.rgb_encoder_session = ort.InferenceSession(
            str(encoder_path), sess_options=sess_options, providers=providers
        )
        unet_path = self.onnx_dir / "unet.onnx"
        self.unet_session = ort.InferenceSession(
            str(unet_path), sess_options=sess_options, providers=providers
        )

    def _setup_scheduler(self):
        """Setup DDIM scheduler."""
        scheduler_kwargs = {
            "num_train_timesteps": self.config["num_train_timesteps"],
            "beta_start": self.config["beta_start"],
            "beta_end": self.config["beta_end"],
            "beta_schedule": self.config["beta_schedule"],
            "clip_sample": self.config["clip_sample"],
            "clip_sample_range": self.config["clip_sample_range"],
            "prediction_type": self.config["prediction_type"],
        }
        self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)

    def _load_image_features_config(self):
        """Load the image_features list from the original model config."""
        if not (self.checkpoint_path / "pretrained_model").exists():
            checkpoint_path = self.checkpoint_path / "output" / "checkpoints" / "last"
        else:
            checkpoint_path = self.checkpoint_path

        model_path = checkpoint_path / "pretrained_model"
        config_path = model_path / "config.json"

        with open(config_path, "r") as f:
            model_config = json.load(f)

        self.image_features = [
            "observation.images.sync_left_arm_cam",
            "observation.images.sync_head_cam",
        ]
        self.camera_key_map = {}
        for key in self.image_features:
            if "head" in key:
                self.camera_key_map["head"] = key
            elif "left" in key:
                self.camera_key_map["left"] = key

    def _setup_normalization(self):
        """Setup normalization from policy stats."""
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        if not (self.checkpoint_path / "pretrained_model").exists():
            checkpoint_path = self.checkpoint_path / "output" / "checkpoints" / "last"
        else:
            checkpoint_path = self.checkpoint_path

        model_path = checkpoint_path / "pretrained_model"
        policy = DiffusionPolicy.from_pretrained(str(model_path))

        self.norm_stats = {}
        self.unnorm_stats = {}

        if hasattr(policy, "normalize_inputs") and policy.normalize_inputs is not None:
            for buffer_name in dir(policy.normalize_inputs):
                if buffer_name.startswith("buffer_"):
                    feature_name = buffer_name.replace("buffer_", "").replace("_", ".")
                    buffer = getattr(policy.normalize_inputs, buffer_name)
                    stats = {}
                    if hasattr(buffer, "mean") and hasattr(buffer, "std"):
                        stats["mode"] = "mean_std"
                        stats["mean"] = buffer["mean"].cpu().numpy()
                        stats["std"] = buffer["std"].cpu().numpy()
                    elif hasattr(buffer, "min") and hasattr(buffer, "max"):
                        stats["mode"] = "min_max"
                        stats["min"] = buffer["min"].cpu().numpy()
                        stats["max"] = buffer["max"].cpu().numpy()
                    if stats:
                        self.norm_stats[feature_name] = stats

        if hasattr(policy, "normalize_targets") and policy.normalize_targets is not None:
            for buffer_name in dir(policy.normalize_targets):
                if buffer_name.startswith("buffer_"):
                    feature_name = buffer_name.replace("buffer_", "").replace("_", ".")
                    buffer = getattr(policy.normalize_targets, buffer_name)
                    stats = {}
                    if hasattr(buffer, "mean") and hasattr(buffer, "std"):
                        stats["mode"] = "mean_std"
                        stats["mean"] = buffer["mean"].cpu().numpy()
                        stats["std"] = buffer["std"].cpu().numpy()
                    elif hasattr(buffer, "min") and hasattr(buffer, "max"):
                        stats["mode"] = "min_max"
                        stats["min"] = buffer["min"].cpu().numpy()
                        stats["max"] = buffer["max"].cpu().numpy()
                    if stats:
                        self.unnorm_stats[feature_name] = stats

        del policy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def decode_compressed_image_msg(
        self, compressed_msg, is_top_view: bool = False
    ) -> np.ndarray:
        """Decode ROS CompressedImage message to RGB array with preprocessing."""
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode compressed image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if is_top_view:
            x, y, w, h = self.crop_box
            image_cropped = image_rgb[y : y + h, x : x + w]
            image_processed = cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)
        else:
            image_processed = cv2.resize(
                image_rgb, self.target_size, interpolation=cv2.INTER_AREA
            )
        return image_processed

    def _forwardKinematics(self, theta, tcp=None):
        """Forward kinematics for UR-10e with calibration offsets."""
        a = np.array([0.0000, -0.6127, -0.57155, 0.0000, 0.0000, 0.0000])
        d = np.array([0.1807, 0.0000, 0.0000, 0.17415, 0.11985, 0.11655])
        alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
        
        delta_a = np.array([3.1576640107943976e-05, 0.298634925475782076, 0.227031257526500829, -8.27068507303316573e-05, 3.6195435783833642e-05, 0])
        delta_d = np.array([5.82932048768247668e-05, 362.998939868892023, -614.839459588742898, 251.84113332747981, 0.000164511802564715204, -0.000899906496469232708])
        delta_alpha = np.array([-0.000774756642435869836, 0.00144883356002286951, -0.00181081418698111852, 0.00068792563586761446, 0.000450856239573305118, 0])
        delta_theta = np.array([1.09391516130152855e-07, 1.03245736607748673, 6.17452995676434124, -0.92380698472218048, 6.42771759845617296e-07, -3.18941184192234051e-08])

        a += delta_a
        d += delta_d
        alpha += delta_alpha
        theta = theta.copy() + delta_theta

        ot = np.eye(4)
        for i in range(6):
            ot = ot @ np.array([[np.cos(theta[i]), -(np.sin(theta[i]))*np.cos(alpha[i]), np.sin(theta[i])*np.sin(alpha[i]), a[i]*np.cos(theta[i])],[np.sin(theta[i]),np.cos(theta[i])*np.cos(alpha[i]),-(np.cos(theta[i]))*np.sin(alpha[i]),a[i]*np.sin(theta[i])], [0.0,np.sin(alpha[i]),np.cos(alpha[i]),d[i]],[0.0,0.0,0.0,1.0]])

        return np.array([ot[0,3], ot[1,3], ot[2,3]])

    def extract_cartesian_state_from_joints(self, joint_state_msg) -> np.ndarray:
        """Extract cartesian position (x, y) from JointState message."""
        joint_names = list(joint_state_msg.name)
        positions = np.array(joint_state_msg.position, dtype=np.float32)

        left_arm_indices = []
        for joint_name in self.left_arm_joints:
            if joint_name in joint_names:
                left_arm_indices.append(joint_names.index(joint_name))

        if len(left_arm_indices) < 6:
            raise ValueError(f"Expected 6 left arm joints, found {len(left_arm_indices)}")

        theta = positions[left_arm_indices[:6]]
        cartesian_pose = self._forwardKinematics(theta)
        # Return (-x, -y) as per reference implementation
        return np.array([-cartesian_pose[0], -cartesian_pose[1]], dtype=np.float32)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input (CHW format, normalized)."""
        image_normalized = image.astype(np.float32) / 255.0
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        return image_chw

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state observation."""
        if "observation.state" in self.norm_stats:
            stats = self.norm_stats["observation.state"]
            if stats.get("mode") == "min_max":
                min_val = stats["min"]
                max_val = stats["max"]
                state = (state - min_val) / (max_val - min_val + 1e-8)
                state = state * 2 - 1
            elif stats.get("mode") == "mean_std":
                mean = stats["mean"]
                std = stats["std"]
                state = (state - mean) / (std + 1e-8)
        return state

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image observation."""
        possible_keys = [
            "observation.image",
            "observation.images.sync_left_arm_cam",
            "observation.images.sync_head_cam",
        ]
        for key in possible_keys:
            if key in self.norm_stats:
                stats = self.norm_stats[key]
                if stats.get("mode") == "mean_std":
                    mean = stats["mean"].reshape(3, 1, 1)
                    std = stats["std"].reshape(3, 1, 1)
                    image = (image - mean) / (std + 1e-8)
                break
        return image

    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action."""
        if "action" in self.unnorm_stats:
            stats = self.unnorm_stats["action"]
            if stats.get("mode") == "min_max":
                action = (action + 1) / 2
                min_val = stats["min"]
                max_val = stats["max"]
                action = action * (max_val - min_val) + min_val
            elif stats.get("mode") == "mean_std":
                mean = stats["mean"]
                std = stats["std"]
                action = action * (std + 1e-8) + mean
        return action

    def _encode_images(self, images: np.ndarray) -> np.ndarray:
        """Encode images using ONNX RGB encoder."""
        if torch.is_tensor(images):
            images = images.cpu().numpy()

        batch_size, n_cameras = images.shape[:2]
        images_flat = images.reshape(-1, *images.shape[2:])

        if self.center_crop is not None:
            images_tensor = torch.from_numpy(images_flat)
            images_cropped = self.center_crop(images_tensor)
            images_flat = images_cropped.numpy()

        features_flat = self.rgb_encoder_session.run(
            ["features"], {"image": images_flat}
        )[0]
        features = features_flat.reshape(batch_size, -1)
        return features

    def predict(self, left_image, head_image, cartesian_state) -> np.ndarray:
        """Run inference on preprocessed inputs."""
        left_processed = self.preprocess_image(left_image)
        head_processed = self.preprocess_image(head_image)
        left_normalized = self._normalize_image(left_processed)
        head_normalized = self._normalize_image(head_processed)

        batch_images = {}
        if "left" in self.camera_key_map:
            batch_images[self.camera_key_map["left"]] = left_normalized
        if "head" in self.camera_key_map:
            batch_images[self.camera_key_map["head"]] = head_normalized

        images_stacked = np.stack(
            [batch_images[key] for key in self.image_features], axis=0
        )

        state_normalized = self._normalize_state(cartesian_state)
        self._queues["observation.state"].append(state_normalized)
        self._queues["observation.images"].append(images_stacked)

        if len(self._queues["action"]) == 0:
            state_list = list(self._queues["observation.state"])
            images_list = list(self._queues["observation.images"])

            while len(state_list) < self.config["n_obs_steps"]:
                state_list.insert(0, state_list[0] if state_list else state_normalized)
                images_list.insert(0, images_list[0] if images_list else images_stacked)

            state_list = state_list[-self.config["n_obs_steps"] :]
            images_list = images_list[-self.config["n_obs_steps"] :]

            state_batch = np.stack(state_list, axis=0)[np.newaxis, ...]
            images_batch = np.stack(images_list, axis=0)[np.newaxis, ...]

            action_chunk_unnorm = self._run_inference(images_batch, state_batch)
            self._queues["action"].extend(action_chunk_unnorm[0])

        action = self._queues["action"].popleft()
        return action

    def _run_inference(self, images_batch, state_batch):
        """Core inference logic."""
        batch_size, n_obs_steps, n_cameras, C, H, W = images_batch.shape
        images_reshaped = images_batch.reshape(
            batch_size * n_obs_steps, n_cameras, C, H, W
        )
        features_reshaped = self._encode_images(images_reshaped)
        img_features = features_reshaped.reshape(batch_size, n_obs_steps, -1)

        global_cond_unflat = np.concatenate([state_batch, img_features], axis=2)
        global_cond = global_cond_unflat.reshape(batch_size, -1).astype(np.float32)

        noise = np.random.randn(
            batch_size, self.config["horizon"], self.config["action_dim"]
        ).astype(np.float32)

        sample = noise.copy()
        num_inference_steps = self.config.get(
            "num_inference_steps", self.config["num_train_timesteps"]
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            timestep = np.array([t.item()], dtype=np.int64)
            timestep = np.repeat(timestep, batch_size)
            model_output = self.unet_session.run(
                ["noise_pred"],
                {"sample": sample, "timestep": timestep, "global_cond": global_cond},
            )[0]
            sample = self.noise_scheduler.step(
                torch.from_numpy(model_output), t, torch.from_numpy(sample)
            ).prev_sample.numpy()

        actions = sample
        start = self.config["n_obs_steps"] - 1
        end = start + self.config["n_action_steps"]
        action_chunk = actions[:, start:end]
        action_chunk_unnorm = self._unnormalize_action(action_chunk)
        return action_chunk_unnorm

    def predict_from_ros_messages(
        self, left_compressed_msg, head_compressed_msg, joint_state_msg
    ) -> np.ndarray:
        """Run inference from ROS messages."""
        left_image = self.decode_compressed_image_msg(
            left_compressed_msg, is_top_view=False
        )
        head_image = self.decode_compressed_image_msg(
            head_compressed_msg, is_top_view=True
        )
        cartesian_state = self.extract_cartesian_state_from_joints(joint_state_msg)
        return self.predict(left_image, head_image, cartesian_state)


class InferenceNode(Node):
    """ROS2 node for LeRobot inference deployment with Cartesian FK."""

    def __init__(
        self,
        checkpoint_path: str,
        onnx_dir: str,
        device: str = "cuda",
        inference_frequency: float = 20.0,
        mode: str = "continuous",
        debug: bool = False,
    ):
        super().__init__("lerobot_inference_node_onnx_cartesian")

        self.mode = mode
        self.debug = debug
        self.inference_frequency = inference_frequency
        self.inference_interval = 1.0 / inference_frequency

        self.inference = ONNXTensorRTInference(
            checkpoint_path, onnx_dir, device
        )

        self.latest_messages = {
            "left_image": None,
            "head_image": None,
            "joint_state": None,
        }
        self.latest_message_times = {
            "left_image": 0.0,
            "head_image": 0.0,
            "joint_state": 0.0,
        }
        self.message_lock = threading.Lock()

        self.inference_running = False
        self.inference_thread = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.left_image_sub = self.create_subscription(
            CompressedImage,
            "/sync/emily01/left_arm/color/image_raw/compressed",
            self.left_image_callback,
            sensor_qos,
        )

        self.head_image_sub = self.create_subscription(
            CompressedImage,
            "/sync/emily01/head/color/image_raw/compressed",
            self.head_image_callback,
            sensor_qos,
        )

        self.joint_state_sub = self.create_subscription(
            JointState, "/sync/joint_states", self.joint_state_callback, sensor_qos
        )

        self.action_pub = self.create_publisher(
            TwistStamped, "/la/servo_node/delta_twist_cmds", control_qos
        )

        self.get_logger().info("LeRobot Cartesian ONNX Inference Node ready (Absolute)")

    def left_image_callback(self, msg: CompressedImage):
        with self.message_lock:
            self.latest_messages["left_image"] = msg
            self.latest_message_times["left_image"] = time.time()
        if self.mode == "triggered":
            self.trigger_inference()

    def head_image_callback(self, msg: CompressedImage):
        with self.message_lock:
            self.latest_messages["head_image"] = msg
            self.latest_message_times["head_image"] = time.time()

    def joint_state_callback(self, msg: JointState):
        with self.message_lock:
            self.latest_messages["joint_state"] = msg
            self.latest_message_times["joint_state"] = time.time()

    def trigger_inference(self):
        with self.message_lock:
            required = ["left_image", "head_image", "joint_state"]
            if any(self.latest_messages[m] is None for m in required):
                return
            left_msg = self.latest_messages["left_image"]
            head_msg = self.latest_messages["head_image"]
            joint_msg = self.latest_messages["joint_state"]

        if not self.inference_running:
            threading.Thread(
                target=self.run_inference_threaded,
                args=(left_msg, head_msg, joint_msg),
                daemon=True,
            ).start()

    def start_inference_loop(self):
        if self.mode != "continuous": return
        self.inference_running = True
        self.inference_thread = threading.Thread(target=self.continuous_inference_loop, daemon=True)
        self.inference_thread.start()

    def stop_inference_loop(self):
        self.inference_running = False
        if self.inference_thread: self.inference_thread.join(timeout=2.0)

    def continuous_inference_loop(self):
        while self.inference_running:
            loop_start = time.time()
            try:
                with self.message_lock:
                    required = ["left_image", "head_image", "joint_state"]
                    if all(self.latest_messages[m] is not None for m in required):
                        left_msg = self.latest_messages["left_image"]
                        head_msg = self.latest_messages["head_image"]
                        joint_msg = self.latest_messages["joint_state"]
                        self.run_inference_synchronous(left_msg, head_msg, joint_msg)
            except Exception as e:
                self.get_logger().error(f"Inference loop error: {e}")
            elapsed = time.time() - loop_start
            time.sleep(max(0, self.inference_interval - elapsed))

    def run_inference_threaded(self, left_msg, head_msg, joint_msg):
        self.inference_running = True
        try:
            action = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
            self.publish_action(action)
        except Exception as e:
            self.get_logger().error(f"Threaded inference failed: {e}")
        finally:
            self.inference_running = False

    def run_inference_synchronous(self, left_msg, head_msg, joint_msg):
        try:
            action = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
            self.publish_action(action)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

    def publish_action(self, action: np.ndarray):
        """Publish action as TwistStamped message."""
        if action is None: return
        # action is expected to be 2D for linear.x and linear.y
        if len(action) < 2:
            self.get_logger().error(f"Expected at least 2D action, got {len(action)}D")
            return

        msg = TwistStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        # Apply model output to linear x and y
        # Invert back to match robot frame
        msg.twist.linear.x = float(-action[0])
        msg.twist.linear.y = float(-action[1])
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        
        self.action_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="LeRobot Cartesian ONNX Inference Node")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--mode", type=str, default="continuous", choices=["continuous", "triggered"])
    
    parsed_args, unknown = parser.parse_known_args()
    rclpy.init(args=unknown)
    
    try:
        node = InferenceNode(
            checkpoint_path=parsed_args.checkpoint,
            onnx_dir=parsed_args.onnx_dir,
            device=parsed_args.device,
            inference_frequency=parsed_args.frequency,
            mode=parsed_args.mode
        )
        if parsed_args.mode == "continuous":
            node.start_inference_loop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if "node" in locals():
            node.stop_inference_loop()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
