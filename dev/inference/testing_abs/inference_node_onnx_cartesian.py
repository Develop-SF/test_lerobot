#!/usr/bin/env python3
"""
ROS2 Inference Node for LeRobot Deployment - Cartesian Mode (ONNX+TensorRT)
Absolute Action Mode

This node subscribes to sensor topics, runs inference using ONNX Runtime with TensorRT,
and publishes actions as TwistStamped.
"""

import sys
import time
import threading
import argparse
import json
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
    """ONNX Runtime with TensorRT backend inference for Cartesian model."""

    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.onnx_dir = Path(onnx_dir)

        # Load ONNX configuration
        config_path = self.onnx_dir / "onnx_config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self._load_image_features_config()

        print(f"Loaded ONNX config:", flush=True)
        print(f"  Horizon: {self.config.get('horizon')}", flush=True)
        print(f"  N obs steps: {self.config.get('n_obs_steps')}", flush=True)
        print(f"  N action steps: {self.config.get('n_action_steps')}", flush=True)
        print(f"  Action dim: {self.config.get('action_dim')}", flush=True)
        print(f"  State dim: {self.config.get('state_dim')}", flush=True)
        print(f"  Inference steps: {self.config.get('num_inference_steps')}", flush=True)

        self._setup_onnx_sessions()
        self._setup_scheduler()
        self._setup_normalization()

        self.crop_box = (260, 135, 224, 224)
        self.target_size = (224, 224)

        if self.config.get("crop_shape"):
            self.center_crop = torchvision.transforms.CenterCrop(self.config["crop_shape"])
        else:
            self.center_crop = None

        self.left_arm_joints = [
            "la_shoulder_pan_joint", "la_shoulder_lift_joint", "la_elbow_joint",
            "la_wrist_1_joint", "la_wrist_2_joint", "la_wrist_3_joint",
        ]

        self._queues = {
            "action": deque(maxlen=self.config.get("n_action_steps", 1)),
            "observation.state": deque(maxlen=self.config.get("n_obs_steps", 1)),
            "observation.images": deque(maxlen=self.config.get("n_obs_steps", 1)),
        }

    def reset(self):
        self._queues["action"].clear()
        self._queues["observation.state"].clear()
        self._queues["observation.images"].clear()

    def _setup_onnx_sessions(self):
        print("Initializing ONNX Runtime sessions with TensorRT...", flush=True)
        available_providers = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available_providers and self.device == "cuda":
            trt_options = {
                "device_id": 0, "trt_max_workspace_size": 2147483648,
                "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(self.onnx_dir / "trt_engines"),
            }
            providers.append(("TensorrtExecutionProvider", trt_options))
        elif "CUDAExecutionProvider" in available_providers and self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.rgb_encoder_session = ort.InferenceSession(str(self.onnx_dir / "rgb_encoder.onnx"), sess_options=sess_options, providers=providers)
        self.unet_session = ort.InferenceSession(str(self.onnx_dir / "unet.onnx"), sess_options=sess_options, providers=providers)
        print(f"Sessions ready using providers: {self.unet_session.get_providers()}", flush=True)

    def _setup_scheduler(self):
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
        if not (self.checkpoint_path / "pretrained_model").exists():
            checkpoint_path = self.checkpoint_path / "output" / "checkpoints" / "last"
        else:
            checkpoint_path = self.checkpoint_path
        with open(checkpoint_path / "pretrained_model" / "config.json", "r") as f:
            model_config = json.load(f)
        self.image_features = ["observation.images.sync_left_arm_cam", "observation.images.sync_head_cam"]
        self.camera_key_map = {}
        for key in self.image_features:
            if "head" in key: self.camera_key_map["head"] = key
            elif "left" in key: self.camera_key_map["left"] = key

    def _setup_normalization(self):
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        if not (self.checkpoint_path / "pretrained_model").exists():
            checkpoint_path = self.checkpoint_path / "output" / "checkpoints" / "last"
        else:
            checkpoint_path = self.checkpoint_path
        policy = DiffusionPolicy.from_pretrained(str(checkpoint_path / "pretrained_model"))
        self.norm_stats, self.unnorm_stats = {}, {}
        if hasattr(policy, "normalize_inputs") and policy.normalize_inputs is not None:
            for b in dir(policy.normalize_inputs):
                if b.startswith("buffer_"):
                    name = b.replace("buffer_", "").replace("_", "."); buf = getattr(policy.normalize_inputs, b)
                    if hasattr(buf, "mean"): self.norm_stats[name] = {"mode": "mean_std", "mean": buf["mean"].cpu().numpy(), "std": buf["std"].cpu().numpy()}
                    elif hasattr(buf, "min"): self.norm_stats[name] = {"mode": "min_max", "min": buf["min"].cpu().numpy(), "max": buf["max"].cpu().numpy()}
        if hasattr(policy, "normalize_targets") and policy.normalize_targets is not None:
            for b in dir(policy.normalize_targets):
                if b.startswith("buffer_"):
                    name = b.replace("buffer_", "").replace("_", "."); buf = getattr(policy.normalize_targets, b)
                    if hasattr(buf, "mean"): self.unnorm_stats[name] = {"mode": "mean_std", "mean": buf["mean"].cpu().numpy(), "std": buf["std"].cpu().numpy()}
                    elif hasattr(buf, "min"): self.unnorm_stats[name] = {"mode": "min_max", "min": buf["min"].cpu().numpy(), "max": buf["max"].cpu().numpy()}
        del policy; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def decode_compressed_image_msg(self, msg, is_top_view: bool = False) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Decode failed")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if is_top_view:
            x, y, w, h = self.crop_box
            img = cv2.rotate(img[y : y + h, x : x + w], cv2.ROTATE_90_CLOCKWISE)
        else:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img

    def _forwardKinematics(self, theta):
        a = np.array([0.0000, -0.6127, -0.57155, 0.0000, 0.0000, 0.0000])
        d = np.array([0.1807, 0.0000, 0.0000, 0.17415, 0.11985, 0.11655])
        alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
        delta_a = np.array([3.15e-5, 0.2986, 0.2270, -8.27e-5, 3.61e-5, 0])
        delta_d = np.array([5.82e-5, 362.99, -614.83, 251.84, 0.00016, -0.00089])
        delta_alpha = np.array([-0.00077, 0.00144, -0.00181, 0.00068, 0.00045, 0])
        delta_theta = np.array([1.09e-7, 1.032, 6.174, -0.923, 6.42e-7, -3.18e-8])
        a += delta_a; d += delta_d; alpha += delta_alpha; theta = theta.copy() + delta_theta
        ot = np.eye(4)
        for i in range(6):
            ot = ot @ np.array([[np.cos(theta[i]), -(np.sin(theta[i]))*np.cos(alpha[i]), np.sin(theta[i])*np.sin(alpha[i]), a[i]*np.cos(theta[i])],[np.sin(theta[i]),np.cos(theta[i])*np.cos(alpha[i]),-(np.cos(theta[i]))*np.sin(alpha[i]),a[i]*np.sin(theta[i])], [0.0,np.sin(alpha[i]),np.cos(alpha[i]),d[i]],[0.0,0.0,0.0,1.0]])
        return np.array([ot[0,3], ot[1,3], ot[2,3]])

    def extract_cartesian_state_from_joints(self, msg) -> np.ndarray:
        names = list(msg.name); pos = np.array(msg.position, dtype=np.float32)
        idx = [names.index(j) for j in self.left_arm_joints if j in names]
        if len(idx) < 6: raise ValueError("Missing joints")
        cart = self._forwardKinematics(pos[idx[:6]])
        return np.array([-cart[0], -cart[1]], dtype=np.float32)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if "observation.state" in self.norm_stats:
            s = self.norm_stats["observation.state"]; dim = state.shape[-1]
            if s["mode"] == "min_max":
                mi = s["min"][:dim] if s["min"].shape[-1] > dim else s["min"]
                ma = s["max"][:dim] if s["max"].shape[-1] > dim else s["max"]
                state = ((state - mi) / (ma - mi + 1e-8)) * 2 - 1
            elif s["mode"] == "mean_std":
                m = s["mean"][:dim] if s["mean"].shape[-1] > dim else s["mean"]
                st = s["std"][:dim] if s["std"].shape[-1] > dim else s["std"]
                state = (state - m) / (st + 1e-8)
        return state

    def predict(self, left, head, state) -> np.ndarray:
        l_norm = self._normalize_image(np.transpose(left.astype(np.float32)/255.0, (2,0,1)))
        h_norm = self._normalize_image(np.transpose(head.astype(np.float32)/255.0, (2,0,1)))
        img_map = {self.camera_key_map["left"]: l_norm, self.camera_key_map["head"]: h_norm}
        imgs = np.stack([img_map[k] for k in self.image_features], axis=0)
        s_norm = self._normalize_state(state)
        self._queues["observation.state"].append(s_norm); self._queues["observation.images"].append(imgs)

        if len(self._queues["action"]) == 0:
            sl, il = list(self._queues["observation.state"]), list(self._queues["observation.images"])
            while len(sl) < self.config["n_obs_steps"]: sl.insert(0, sl[0]); il.insert(0, il[0])
            sb, ib = np.stack(sl[-self.config["n_obs_steps"]:], axis=0)[np.newaxis, ...], np.stack(il[-self.config["n_obs_steps"]:], axis=0)[np.newaxis, ...]
            # Pad state if model expects 6 but we derived 2
            if sb.shape[-1] < self.config["state_dim"]:
                sb = np.concatenate([sb, np.zeros((sb.shape[0], sb.shape[1], self.config["state_dim"] - sb.shape[-1]), dtype=np.float32)], axis=-1)
            elif sb.shape[-1] > self.config["state_dim"]: sb = sb[..., :self.config["state_dim"]]

            feats = self.rgb_encoder_session.run(["features"], {"image": ib.reshape(-1, *ib.shape[2:])})[0].reshape(1, self.config["n_obs_steps"], -1)
            cond = np.concatenate([sb, feats], axis=2).reshape(1, -1).astype(np.float32)
            sample = np.random.randn(1, self.config["horizon"], self.config["action_dim"]).astype(np.float32)
            self.noise_scheduler.set_timesteps(self.config.get("num_inference_steps", 16))
            for t in self.noise_scheduler.timesteps:
                out = self.unet_session.run(["noise_pred"], {"sample": sample, "timestep": np.array([t.item()], dtype=np.int64), "global_cond": cond})[0]
                sample = self.noise_scheduler.step(torch.from_numpy(out), t, torch.from_numpy(sample)).prev_sample.numpy()
            
            start = self.config["n_obs_steps"] - 1
            chunk = sample[:, start : start + self.config["n_action_steps"]]
            unnorm = chunk
            if "action" in self.unnorm_stats:
                s = self.unnorm_stats["action"]
                if s["mode"] == "min_max": unnorm = ((chunk + 1) / 2) * (s["max"] - s["min"]) + s["min"]
                elif s["mode"] == "mean_std": unnorm = chunk * (s["std"] + 1e-8) + s["mean"]
            self._queues["action"].extend(unnorm[0])
        return self._queues["action"].popleft()

    def _normalize_image(self, img):
        for k in ["observation.image", "observation.images.sync_left_arm_cam", "observation.images.sync_head_cam"]:
            if k in self.norm_stats:
                s = self.norm_stats[k]
                if s["mode"] == "mean_std": img = (img - s["mean"].reshape(3,1,1)) / (s["std"].reshape(3,1,1) + 1e-8)
                break
        return img

    def predict_from_ros(self, l_msg, h_msg, j_msg):
        l = self.decode_compressed_image_msg(l_msg, False); h = self.decode_compressed_image_msg(h_msg, True)
        return self.predict(l, h, self.extract_cartesian_state_from_joints(j_msg))


class InferenceNode(Node):
    def __init__(self, checkpoint, onnx_dir, device="cuda", freq=20.0):
        super().__init__("lerobot_inference_onnx_cartesian")
        self.inference = ONNXTensorRTInference(checkpoint, onnx_dir, device)
        self.interval = 1.0 / freq
        self.latest = {"left": None, "head": None, "joints": None}
        self.lock = threading.Lock(); self.running = False
        
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, "/sync/emily01/left_arm/color/image_raw/compressed", self.cb_left, sensor_qos)
        self.create_subscription(CompressedImage, "/sync/emily01/head/color/image_raw/compressed", self.cb_head, sensor_qos)
        self.create_subscription(JointState, "/sync/joint_states", self.cb_joints, sensor_qos)
        self.pub = self.create_publisher(TwistStamped, "/la/servo_node/delta_twist_cmds", QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10))
        self.get_logger().info("Inference Node Ready (2D Cartesian State -> Twist)", once=True)

    def cb_left(self, msg):
        with self.lock: self.latest["left"] = msg
    def cb_head(self, msg):
        with self.lock: self.latest["head"] = msg
    def cb_joints(self, msg):
        with self.lock: self.latest["joints"] = msg

    def start(self):
        self.running = True; threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            start = time.time()
            with self.lock:
                if all(self.latest[k] is not None for k in ["left", "head", "joints"]):
                    l, h, j = self.latest["left"], self.latest["head"], self.latest["joints"]
                    try:
                        act = self.inference.predict_from_ros(l, h, j)
                        msg = TwistStamped(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = "base_link"
                        msg.twist.linear.x = float(-act[0]); msg.twist.linear.y = float(-act[1])
                        self.pub.publish(msg)
                    except Exception as e: self.get_logger().error(f"Inference failed: {e}")
            while time.time() - start < self.interval: time.sleep(0.001)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--frequency", type=float, default=20.0)
    args = parser.parse_args(); rclpy.init()
    node = InferenceNode(args.checkpoint, args.onnx_dir, args.device, args.frequency)
    node.start(); rclpy.spin(node); rclpy.shutdown()

if __name__ == "__main__": main()
