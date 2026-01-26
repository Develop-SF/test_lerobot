#!/usr/bin/env python3
"""
Evaluate ONNX+TensorRT optimized Cartesian model predictions against ground truth from rosbag data.

This script uses ONNX Runtime with TensorRT backend for optimized inference,
comparing model predictions with actual Cartesian trajectories (x, y) from the rosbag.
"""

# ROS2 imports
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS2 not available. Cannot evaluate with rosbag data.", flush=True)


import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import deque
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# ONNX Runtime imports
import onnxruntime as ort
import torch
import torchvision
import cv2

# Diffusers scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


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
        print(f"  Horizon: {self.config['horizon']}", flush=True)
        print(f"  N obs steps: {self.config['n_obs_steps']}", flush=True)
        print(f"  N action steps: {self.config['n_action_steps']}", flush=True)
        print(f"  Action dim: {self.config['action_dim']}", flush=True)
        print(f"  State dim: {self.config['state_dim']}", flush=True)
        print(f"  Inference steps: {self.config['num_inference_steps']}", flush=True)

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
            "la_shoulder_pan_joint",
            "la_shoulder_lift_joint",
            "la_elbow_joint",
            "la_wrist_1_joint",
            "la_wrist_2_joint",
            "la_wrist_3_joint",
        ]

        self._queues = {
            "action": deque(maxlen=self.config["n_action_steps"]),
            "observation.state": deque(maxlen=self.config["n_obs_steps"]),
            "observation.images": deque(maxlen=self.config["n_obs_steps"]),
        }

    def reset(self):
        self._queues["action"].clear()
        self._queues["observation.state"].clear()
        self._queues["observation.images"].clear()

    def _setup_onnx_sessions(self):
        print("Initializing ONNX Runtime sessions...", flush=True)
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
        self.rgb_encoder_session = ort.InferenceSession(str(encoder_path), sess_options=sess_options, providers=providers)
        unet_path = self.onnx_dir / "unet.onnx"
        self.unet_session = ort.InferenceSession(str(unet_path), sess_options=sess_options, providers=providers)
        print("Sessions ready.", flush=True)

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
            if "head" in key: self.camera_key_map["head"] = key
            elif "left" in key: self.camera_key_map["left"] = key

    def _setup_normalization(self):
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
                        stats["mode"] = "mean_std"; stats["mean"] = buffer["mean"].cpu().numpy(); stats["std"] = buffer["std"].cpu().numpy()
                    elif hasattr(buffer, "min") and hasattr(buffer, "max"):
                        stats["mode"] = "min_max"; stats["min"] = buffer["min"].cpu().numpy(); stats["max"] = buffer["max"].cpu().numpy()
                    if stats: self.norm_stats[feature_name] = stats
        if hasattr(policy, "normalize_targets") and policy.normalize_targets is not None:
            for buffer_name in dir(policy.normalize_targets):
                if buffer_name.startswith("buffer_"):
                    feature_name = buffer_name.replace("buffer_", "").replace("_", ".")
                    buffer = getattr(policy.normalize_targets, buffer_name)
                    stats = {}
                    if hasattr(buffer, "mean") and hasattr(buffer, "std"):
                        stats["mode"] = "mean_std"; stats["mean"] = buffer["mean"].cpu().numpy(); stats["std"] = buffer["std"].cpu().numpy()
                    elif hasattr(buffer, "min") and hasattr(buffer, "max"):
                        stats["mode"] = "min_max"; stats["min"] = buffer["min"].cpu().numpy(); stats["max"] = buffer["max"].cpu().numpy()
                    if stats: self.unnorm_stats[feature_name] = stats
        del policy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def decode_compressed_image_msg(self, compressed_msg, is_top_view: bool = False) -> np.ndarray:
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None: raise ValueError("Failed to decode compressed image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if is_top_view:
            x, y, w, h = self.crop_box
            image_cropped = image_rgb[y : y + h, x : x + w]
            image_processed = cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)
        else:
            image_processed = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        return image_processed

    def _forwardKinematics(self, theta, tcp=None):
        a = np.array([0.0000, -0.6127, -0.57155, 0.0000, 0.0000, 0.0000])
        d = np.array([0.1807, 0.0000, 0.0000, 0.17415, 0.11985, 0.11655])
        alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
        delta_a = np.array([3.1576640107943976e-05, 0.298634925475782076, 0.227031257526500829, -8.27068507303316573e-05, 3.6195435783833642e-05, 0])
        delta_d = np.array([5.82932048768247668e-05, 362.998939868892023, -614.839459588742898, 251.84113332747981, 0.000164511802564715204, -0.000899906496469232708])
        delta_alpha = np.array([-0.000774756642435869836, 0.00144883356002286951, -0.00181081418698111852, 0.00068792563586761446, 0.000450856239573305118, 0])
        delta_theta = np.array([1.09391516130152855e-07, 1.03245736607748673, 6.17452995676434124, -0.92380698472218048, 6.42771759845617296e-07, -3.18941184192234051e-08])
        a += delta_a; d += delta_d; alpha += delta_alpha; theta = theta.copy() + delta_theta
        ot = np.eye(4)
        for i in range(6):
            ot = ot @ np.array([[np.cos(theta[i]), -(np.sin(theta[i]))*np.cos(alpha[i]), np.sin(theta[i])*np.sin(alpha[i]), a[i]*np.cos(theta[i])],[np.sin(theta[i]),np.cos(theta[i])*np.cos(alpha[i]),-(np.cos(theta[i]))*np.sin(alpha[i]),a[i]*np.sin(theta[i])], [0.0,np.sin(alpha[i]),np.cos(alpha[i]),d[i]],[0.0,0.0,0.0,1.0]])
        return np.array([ot[0,3], ot[1,3], ot[2,3]])

    def extract_cartesian_state_from_joints(self, joint_state_msg) -> np.ndarray:
        joint_names = list(joint_state_msg.name)
        positions = np.array(joint_state_msg.position, dtype=np.float32)
        indices = [joint_names.index(j) for j in self.left_arm_joints if j in joint_names]
        if len(indices) < 6: raise ValueError("JointState missing joints")
        cartesian_pose = self._forwardKinematics(positions[indices[:6]])
        return np.array([-cartesian_pose[0], -cartesian_pose[1]], dtype=np.float32)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image_normalized = image.astype(np.float32) / 255.0
        return np.transpose(image_normalized, (2, 0, 1))

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state observation with dimension mismatch handling."""
        if "observation.state" in self.norm_stats:
            stats = self.norm_stats["observation.state"]
            dim = state.shape[-1]
            
            if stats.get("mode") == "min_max":
                min_val = stats["min"][:dim] if stats["min"].shape[-1] > dim else stats["min"]
                max_val = stats["max"][:dim] if stats["max"].shape[-1] > dim else stats["max"]
                state = (state - min_val) / (max_val - min_val + 1e-8)
                state = state * 2 - 1
            elif stats.get("mode") == "mean_std":
                mean = stats["mean"][:dim] if stats["mean"].shape[-1] > dim else stats["mean"]
                std = stats["std"][:dim] if stats["std"].shape[-1] > dim else stats["std"]
                state = (state - mean) / (std + 1e-8)
        return state

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        for key in ["observation.image", "observation.images.sync_left_arm_cam", "observation.images.sync_head_cam"]:
            if key in self.norm_stats:
                stats = self.norm_stats[key]
                if stats.get("mode") == "mean_std":
                    image = (image - stats["mean"].reshape(3, 1, 1)) / (stats["std"].reshape(3, 1, 1) + 1e-8)
                break
        return image

    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        if "action" in self.unnorm_stats:
            stats = self.unnorm_stats["action"]
            if stats.get("mode") == "min_max":
                action = (action + 1) / 2
                action = action * (stats["max"] - stats["min"]) + stats["min"]
        return action

    def _encode_images(self, images: np.ndarray) -> np.ndarray:
        if torch.is_tensor(images): images = images.cpu().numpy()
        batch_size, n_cameras = images.shape[:2]
        images_flat = images.reshape(-1, *images.shape[2:])
        if self.center_crop:
            images_flat = self.center_crop(torch.from_numpy(images_flat)).numpy()
        features_flat = self.rgb_encoder_session.run(["features"], {"image": images_flat})[0]
        return features_flat.reshape(batch_size, -1)

    def predict(self, left_image, head_image, state) -> np.ndarray:
        """Run inference on preprocessed inputs."""
        left_normalized = self._normalize_image(self.preprocess_image(left_image))
        head_normalized = self._normalize_image(self.preprocess_image(head_image))
        batch_images = {}
        if "left" in self.camera_key_map: batch_images[self.camera_key_map["left"]] = left_normalized
        if "head" in self.camera_key_map: batch_images[self.camera_key_map["head"]] = head_normalized
        images_stacked = np.stack([batch_images[key] for key in self.image_features], axis=0)
        
        # Handle normalization with potential dimension mismatch
        state_normalized = self._normalize_state(state)
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

            # Ensure state_batch matches model input dimension
            model_state_dim = self.config["state_dim"]
            if state_batch.shape[-1] < model_state_dim:
                padding = np.zeros((state_batch.shape[0], state_batch.shape[1], model_state_dim - state_batch.shape[-1]), dtype=np.float32)
                state_batch = np.concatenate([state_batch, padding], axis=-1)
            elif state_batch.shape[-1] > model_state_dim:
                state_batch = state_batch[..., :model_state_dim]

            features_reshaped = self._encode_images(images_batch.reshape(-1, *images_batch.shape[2:]))
            img_features = features_reshaped.reshape(1, self.config["n_obs_steps"], -1)
            global_cond = np.concatenate([state_batch, img_features], axis=2).reshape(1, -1).astype(np.float32)

            sample = noise = np.random.randn(1, self.config["horizon"], self.config["action_dim"]).astype(np.float32)
            self.noise_scheduler.set_timesteps(self.config.get("num_inference_steps", 16))
            for t in self.noise_scheduler.timesteps:
                model_output = self.unet_session.run(["noise_pred"], {"sample": sample, "timestep": np.array([t.item()], dtype=np.int64), "global_cond": global_cond})[0]
                sample = self.noise_scheduler.step(torch.from_numpy(model_output), t, torch.from_numpy(sample)).prev_sample.numpy()
            
            start = self.config["n_obs_steps"] - 1
            action_chunk = sample[:, start : start + self.config["n_action_steps"]]
            self._queues["action"].extend(self._unnormalize_action(action_chunk)[0])

        return self._queues["action"].popleft()

    def predict_from_ros_messages(self, left_compressed_msg, head_compressed_msg, joint_state_msg) -> np.ndarray:
        left = self.decode_compressed_image_msg(left_compressed_msg, is_top_view=False)
        head = self.decode_compressed_image_msg(head_compressed_msg, is_top_view=True)
        cart_state = self.extract_cartesian_state_from_joints(joint_state_msg)
        return self.predict(left, head, cart_state)


class PredictionEvaluator:
    """Evaluate Cartesian model predictions."""

    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda"):
        self.inference = ONNXTensorRTInference(checkpoint_path, onnx_dir, device)
        self.topics = {
            "left_image": "/sync/emily01/left_arm/color/image_raw/compressed",
            "head_image": "/sync/emily01/head/color/image_raw/compressed",
            "joint_state": "/sync/joint_states",
            "action_command": "/sync/la/servo_node/delta_twist_cmds",
        }

    def load_rosbag_data(self, rosbag_path: str, max_samples: int = 100) -> Dict[str, List]:
        if not ROS_AVAILABLE: raise RuntimeError("ROS2 not available")
        storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        reader = rosbag2_py.SequentialReader(); reader.open(storage_options, converter_options)
        
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        messages = {topic: [] for topic in self.topics.values()}
        print(f"Reading bag... Looking for topics: {list(self.topics.values())}", flush=True)
        
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic in messages:
                if len(messages[topic]) >= max_samples:
                    if all(len(msgs) >= max_samples for msgs in messages.values()):
                        break
                    continue
                    
                msg_type_str = topic_types.get(topic)
                if not msg_type_str: continue
                msg_type = get_message(msg_type_str)
                messages[topic].append(deserialize_message(data, msg_type))
                
        for t, msgs in messages.items():
            print(f"Collected {len(msgs)} messages for topic: {t}", flush=True)
        return messages

    def evaluate(self, rosbag_path: str, num_samples: int = 50) -> Dict:
        print(f"Loading data from: {rosbag_path}", flush=True)
        messages = self.load_rosbag_data(rosbag_path, num_samples)
        print("Data loading complete.", flush=True)
        
        counts = {t: len(msgs) for t, msgs in messages.items()}
        print(f"Synchronized message counts: {counts}", flush=True)
        
        min_count = min(counts.values())
        num_samples = min(num_samples, min_count)
        if num_samples == 0:
            raise ValueError(f"No synchronized messages. Counts: {counts}")

        predictions, ground_truth, times = [], [], []
        print(f"\nStarting evaluation loop for {num_samples} samples...", flush=True)
        for i in range(num_samples):
            try:
                print(f"  [{i+1}/{num_samples}] Processing sample...", end="\r", flush=True)
                left = messages[self.topics["left_image"]][i]
                head = messages[self.topics["head_image"]][i]
                joint = messages[self.topics["joint_state"]][i]
                action_msg = messages[self.topics["action_command"]][i]
                
                start = time.perf_counter()
                pred = self.inference.predict_from_ros_messages(left, head, joint)
                times.append(time.perf_counter() - start)
                gt = np.array([-action_msg.twist.linear.x, -action_msg.twist.linear.y], dtype=np.float32)
                predictions.append(pred[:2])
                ground_truth.append(gt)
                
                if (i + 1) % 10 == 0:
                    avg_t = np.mean(times) * 1000
                    print(f"\n  [{i+1}/{num_samples}] Done. Avg inference time: {avg_t:.2f} ms", flush=True)
            except Exception as e:
                print(f"\n  [{i+1}/{num_samples}] Failed: {e}", flush=True)
        
        print(f"\nEvaluation loop complete. Calculating metrics...", flush=True)
        predictions, ground_truth = np.array(predictions), np.array(ground_truth)
        results = {
            "mae": np.mean(np.abs(predictions - ground_truth), axis=0),
            "rmse": np.sqrt(np.mean((predictions - ground_truth)**2, axis=0)),
            "predictions": predictions,
            "ground_truth": ground_truth,
            "inference_times": times
        }
        print("Metrics calculation complete.", flush=True)
        return results

    def plot_results(self, results: Dict, save_path: str = None):
        preds, gt = results["predictions"], results["ground_truth"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, label in enumerate(["Linear X", "Linear Y"]):
            axes[i].plot(gt[:, i], label="Ground Truth")
            axes[i].plot(preds[:, i], label="Prediction")
            axes[i].set_title(label); axes[i].legend()
            axes[i].text(0.05, 0.95, f"MAE: {results['mae'][i]:.4f}\nRMSE: {results['rmse'][i]:.4f}", transform=axes[i].transAxes, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.5))
        plt.tight_layout()
        if save_path: plt.savefig(save_path); print(f"Plot saved to: {save_path}", flush=True)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cartesian ONNX predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx-dir", type=str, required=True)
    parser.add_argument("--rosbag", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()
    
    if not ROS_AVAILABLE: return
    evaluator = PredictionEvaluator(args.checkpoint, args.onnx_dir, args.device)
    results = evaluator.evaluate(args.rosbag, args.num_samples)
    print(f"\nOverall MAE: {np.mean(results['mae']):.4f}, RMSE: {np.mean(results['rmse']):.4f}", flush=True)
    print(f"Avg Inference: {np.mean(results['inference_times'])*1000:.2f} ms", flush=True)
    if args.plot: evaluator.plot_results(results, args.save_plot)

if __name__ == "__main__":
    main()
