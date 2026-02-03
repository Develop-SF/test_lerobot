import gradio as gr
import psutil
import pynvml
import threading
import time
import subprocess
import os
import sys
import uuid
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Configuration & State ---

@dataclass
class Job:
    id: str
    name: str
    command: List[str]
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED, STOPPED
    log_file: str = ""
    created_at: float = field(default_factory=time.time)
    process: Optional[subprocess.Popen] = None
    
    def to_dict(self):
        return {
            "ID": self.id,
            "Name": self.name,
            "Status": self.status,
            "Command": " ".join(self.command[:3]) + "...",
            "Created": datetime.fromtimestamp(self.created_at).strftime("%H:%M:%S")
        }

class JobScheduler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobScheduler, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.pending_jobs: List[Job] = []
        self.active_job: Optional[Job] = None
        self.all_jobs: Dict[str, Job] = {} # History
        self.lock = threading.Lock()
        self.stop_flag = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
        # Ensure logs directory exists
        os.makedirs("gui_logs", exist_ok=True)

    def submit_job(self, name: str, command: List[str]) -> str:
        job_id = str(uuid.uuid4())[:8]
        log_file = os.path.join("gui_logs", f"{job_id}_{name}.log")
        
        job = Job(id=job_id, name=name, command=command, log_file=log_file)
        
        with self.lock:
            if len(self.pending_jobs) >= 10:
                 return None
            self.pending_jobs.append(job)
            
        self.all_jobs[job_id] = job
        print(f"Job {name} ({job_id}) enqueued.")
        return job_id

    def remove_job(self, job_id: str):
        with self.lock:
            for i, job in enumerate(self.pending_jobs):
                if job.id == job_id:
                    removed_job = self.pending_jobs.pop(i)
                    removed_job.status = "CANCELLED"
                    return f"Job {job_id} removed from queue."
            
            if self.active_job and self.active_job.id == job_id:
                self.stop_active_job()
                return f"Job {job_id} stopped."
                
            return "Job not found in pending queue."

    def stop_active_job(self):
        if self.active_job and self.active_job.process:
            try:
                # Use psutil to kill the process and all its children
                parent = psutil.Process(self.active_job.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"Error while killing process: {e}")
            
            self.active_job.status = "STOPPED"
            with open(self.active_job.log_file, "a") as f:
                f.write("\n\n--- JOB MANUALLY STOPPED ---\n")
    
    def get_job_list(self):
        # Return list of dicts for DataFrame
        jobs_list = sorted(self.all_jobs.values(), key=lambda x: x.created_at, reverse=True)
        return [j.to_dict() for j in jobs_list]

    def _worker(self):
        print("Scheduler worker started.")
        while True:
            job = None
            with self.lock:
                if self.pending_jobs:
                    job = self.pending_jobs.pop(0)
            
            if not job:
                time.sleep(1)
                continue
                
            self.active_job = job
            job.status = "RUNNING"
            print(f"Starting job: {job.name}")
            
            with open(job.log_file, "w") as f:
                f.write(f"Command: {' '.join(job.command)}\n\n")
            
            try:
                with open(job.log_file, "a") as f:
                    job.process = subprocess.Popen(
                        job.command,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=os.getcwd(), 
                        env=os.environ.copy()
                    )
                    
                job.process.wait()
                
                if job.status != "STOPPED":
                     if job.process.returncode == 0:
                        job.status = "COMPLETED"
                     else:
                        job.status = "FAILED"
                        
            except Exception as e:
                job.status = "FAILED"
                with open(job.log_file, "a") as f:
                    f.write(f"\nExample execution failed: {str(e)}\n")
            finally:
                self.active_job = None

SCHEDULER = JobScheduler()

class SystemMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.has_gpu = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except:
            self.has_gpu = False
            self.gpu_count = 0
            
    def get_stats(self):
        # CPU
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_usage = f"{ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f} GB"
        
        stats = [
            f"CPU: {cpu_usage}%",
            f"RAM: {ram_usage}"
        ]
        
        # GPU
        if self.has_gpu:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                    
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # mW to W
                
                mem_str = f"{mem.used / (1024**2):.0f}/{mem.total / (1024**2):.0f} MB"
                stats.append(f"GPU {i} ({name}): {util}% Util | {mem_str} VRAM | {power:.1f} W")
                
        return " | ".join(stats)

MONITOR = SystemMonitor()

# --- Helpers ---

# Global to track the last displayed job to avoid clearing logs
LAST_DISPLAYED_JOB_ID = None

def get_job_logs():
    global LAST_DISPLAYED_JOB_ID
    
    job = None
    if SCHEDULER.active_job:
        job = SCHEDULER.active_job
        LAST_DISPLAYED_JOB_ID = job.id
    else:
        # If no active job, show the last known job (either the one that just finished or the latest overall)
        if LAST_DISPLAYED_JOB_ID and LAST_DISPLAYED_JOB_ID in SCHEDULER.all_jobs:
            job = SCHEDULER.all_jobs[LAST_DISPLAYED_JOB_ID]
        else:
            jobs = sorted(SCHEDULER.all_jobs.values(), key=lambda x: x.created_at, reverse=True)
            if not jobs:
                return "No jobs run yet."
            job = jobs[0]
            LAST_DISPLAYED_JOB_ID = job.id
    
    try:
        if not os.path.exists(job.log_file):
            return f"Log file not found for job {job.id} ({job.name})"
        with open(job.log_file, "r", errors="replace") as f:
            lines = f.readlines()
            # Show last 500 lines to keep UI snappy
            return "".join(lines[-500:])
    except Exception as e:
        return f"Error reading logs: {str(e)}"

def update_queue_display():
    return SCHEDULER.get_job_list()

def update_monitor():
    return MONITOR.get_stats()

def remove_job_from_queue(job_id_to_remove):
    if not job_id_to_remove:
        return "Please enter a Job ID.", update_queue_display()
    
    msg = SCHEDULER.remove_job(job_id_to_remove)
    return msg, update_queue_display()

def stop_active_job_ui():
    if SCHEDULER.active_job:
        job_id = SCHEDULER.active_job.id
        SCHEDULER.stop_active_job()
        return f"Stopped active job: {job_id}", update_queue_display()
    return "No active job to stop.", update_queue_display()

def get_active_job_progress():
    if not SCHEDULER.active_job:
        if LAST_DISPLAYED_JOB_ID and LAST_DISPLAYED_JOB_ID in SCHEDULER.all_jobs:
            job = SCHEDULER.all_jobs[LAST_DISPLAYED_JOB_ID]
            return f"{job.status}: {job.name}"
        return "Not running"
    
    try:
        with open(SCHEDULER.active_job.log_file, "r", errors="replace") as f:
            lines = f.readlines()
            if not lines:
                return "Starting..."
            
            # Look for progress patterns in the last 50 lines for better coverage
            for line in reversed(lines[-50:]):
                if "📁 Processing bag:" in line:
                    return line.strip()
                if "Writing" in line and "/" in line:
                    return line.strip()
                if "Extracted:" in line:
                    return line.strip()
                if "step:" in line and "/" in line:
                    return "Training " + line.strip()
            
            return f"Running: {SCHEDULER.active_job.name}"
    except:
        return "Reading logs..."

def confirm_overwrite_dialog(path):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = messagebox.askyesno("Overwrite Dataset?", f"The directory {path} already exists.\n\nDo you want to PERMANENTLY ERASE it and overwrite?")
    root.destroy()
    return result

def open_file_dialog(is_dir=False, initial_dir=None):
    # This runs ON THE SERVER
    # Since this is a local app, it pops up on the user's screen.
    root = tk.Tk()
    root.withdraw() # Hide main window
    root.attributes('-topmost', True) # Bring to front
    
    if is_dir:
        path = filedialog.askdirectory(initialdir=initial_dir)
    else:
        path = filedialog.askopenfilename(initialdir=initial_dir)
        
    root.destroy()
    return path

# --- Task Command Generators ---

def generate_convert_cmd(rosbag_path, base_output_dir, system_setup, dataset_name, input_mode, output_mode, fps, obs_topics, act_topics, num_workers, force):
    script = "rosbag_to_lerobot_rosbag2.py"
    if "7-DoF" in system_setup:
        script = "rosbag_to_lerobot_rosbag2_7DoF.py"
    elif "Cropped" in system_setup:
        script = "rosbag_to_lerobot_cropped_trimmed.py"
    
    # Automatically create a subfolder with the dataset name
    full_output_dir = os.path.join(base_output_dir, dataset_name)
    
    cmd = [
        sys.executable,
        f"dev/data_processing/conversion/{script}",
        rosbag_path,
        "--output-dir", full_output_dir,
        "--dataset-name", dataset_name,
        "--input-mode", input_mode,
        "--output-mode", output_mode,
        "--num-workers", str(int(num_workers)),
    ]
    
    # Topics
    if obs_topics:
        obs_list = [t.strip() for t in obs_topics.split(",") if t.strip()]
        if obs_list:
            cmd.extend(["--observation-topics"] + obs_list)
    
    if act_topics:
        act_list = [t.strip() for t in act_topics.split(",") if t.strip()]
        if act_list:
            cmd.extend(["--action-topics"] + act_list)

    if force:
        cmd.append("--force")
    
    # Always pass FPS parameter
    cmd.extend(["--fps", str(fps)])
    
    if "7-DoF" in system_setup:
        # Use dataset_name for task name as requested
        cmd.extend(["--task", dataset_name])
        
    return cmd

def generate_train_cmd(dataset_root, repo_id, output_dir, policy_type, action_mode, arm_dim, backbone, batch_size, steps, workers, obs_horizon, pred_horizon, action_horizon, job_name):
    
    cmd = []
    
    if "Relative" in action_mode:
        stats_path = os.path.join(dataset_root, "relative_stats.json")
        config_path = os.path.join(dataset_root, "training_config.json")
        
        cmd = [sys.executable, "lerobot/scripts/train_with_relative_actions.py"]
        cmd.extend(["--config_path", config_path])
        cmd.extend(["--output_dir", output_dir])
        cmd.append("--use_relative_actions")
        cmd.extend(["--arm_dim", str(arm_dim)])
        cmd.extend(["--obs_horizon", str(obs_horizon)])
        cmd.extend(["--relative_stats_path", stats_path])
        
    else:
        cmd = [sys.executable, "-m", "lerobot.scripts.train"]
        cmd.extend(["--dataset.repo_id", repo_id])
        cmd.extend(["--dataset.root", dataset_root])
        cmd.extend(["--policy.type", policy_type])
        cmd.extend(["--policy.vision_backbone", backbone])
        cmd.extend(["--policy.horizon", str(pred_horizon)])
        cmd.extend(["--policy.n_action_steps", str(action_horizon)])
        cmd.extend(["--policy.n_obs_steps", str(obs_horizon)])
        cmd.extend(["--batch_size", str(batch_size)])
        cmd.extend(["--num_workers", str(workers)])
        cmd.extend(["--steps", str(steps)])
        cmd.extend(["--output_dir", output_dir])
        cmd.extend(["--job_name", job_name])
    
    return cmd

def generate_export_cmd(checkpoint_dir, output_dir, model_type, arm_dim, opset):
    script_dir = "dev/inference/testing_abs" if "Absolute" in model_type else "dev/inference/testing_rel"
    script = "convert_to_onnx_7DoF.py" if "Absolute" in model_type else "convert_to_onnx_rel.py"
    
    cmd = [sys.executable, f"{script_dir}/{script}"]
    
    if "Absolute" in model_type:
        cmd.extend(["--checkpoint", checkpoint_dir])
        cmd.extend(["--output", output_dir])
    else:
        cmd.extend(["--checkpoint", checkpoint_dir])
        cmd.extend(["--output-dir", output_dir])
        cmd.extend(["--arm-dim", str(arm_dim)])
        cmd.extend(["--opset-version", str(opset)])
        
    return cmd

def generate_eval_cmd(checkpoint_dir, onnx_dir, rosbag, eval_type, num_samples):
    script_dir = "dev/inference/testing_abs" if "Absolute" in eval_type else "dev/inference/testing_rel"
    script = "evaluate_predictions_onnx_7DoF.py" if "Absolute" in eval_type else "evaluate_predictions_onnx_rel.py"
    
    cmd = [sys.executable, f"{script_dir}/{script}"]
    cmd.extend(["--checkpoint", checkpoint_dir])
    cmd.extend(["--onnx-dir", onnx_dir])
    cmd.extend(["--rosbag", rosbag])
    cmd.extend(["--num-samples", str(num_samples)])
    cmd.append("--plot")
    
    save_path = f"eval_plot_{int(time.time())}.png"
    cmd.extend(["--save-plot", save_path])
    
    return cmd, save_path

# --- UI Layout ---

def create_out_input(label, value=None):
    """Creates a Textbox for output paths (cannot upload to output)."""
    return gr.Textbox(label=label, value=value)

def build_ui():
    # Calculate max workers based on CPU count (80% of available cores)
    max_workers = max(1, int(0.8 * os.cpu_count()))
    
    with gr.Blocks(theme=gr.themes.Soft(), title="LeRobot Training GUI") as app:
        
        with gr.Row(variant="panel"):
            gr.Markdown("## 🤖 LeRobot Training GUI")
            monitor_box = gr.Markdown("Loading hardware stats...", elem_id="monitor")
            monitor_timer = gr.Timer(1)
            monitor_timer.tick(update_monitor, outputs=[monitor_box])

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    # --- TAB 1: CONVERT ---
                    with gr.Tab("1. Convert"):
                        gr.Markdown("Convert ROS bags to LeRobot dataset format.")
                        
                        with gr.Row():
                            with gr.Column(scale=4):
                                rosbag_path = gr.Textbox(label="Rosbag Path (Dataset Root Directory)", placeholder="Click Browse to select...")
                            with gr.Column(scale=1, min_width=100):
                                btn_browse_rosbag = gr.Button("Browse", size="sm")
                            output_convert = create_out_input(label="Output Directory", value="/mnt/nas/dataset/robot_learning/lerobot/")
                        
                        with gr.Row():
                            system_type = gr.Dropdown([
                                "Standard (6-DoF)", 
                                "7-DoF (UR10e + Gripper)",
                                "Standard (Cropped/Trimmed)"
                            ], label="System Type", value="7-DoF (UR10e + Gripper)")
                            dataset_name = gr.Textbox(label="Dataset / Task Name", value="my_experiment")
                        
                        with gr.Accordion("Topic Configuration", open=True):
                            with gr.Row():
                                obs_topics = gr.Textbox(
                                    label="Observation Topics (comma separated)", 
                                    value="/sync/emily01/front/color/image_raw/compressed, /sync/emily01/head/color/image_raw/compressed, /sync/joint_states",
                                    placeholder="/topic1, /topic2"
                                )
                                act_topics = gr.Textbox(
                                    label="Action Topics (comma separated)", 
                                    value="/sync/ra_trajectory_controller/joint_trajectory, /sync/right_gripper_cmd",
                                    placeholder="/topic_act1"
                                )

                        with gr.Row():
                            input_mode = gr.Dropdown(["vision_pos", "vision_only", "pos_only", "vision_pos_vel"], label="Input Mode", value="vision_pos")
                            output_mode = gr.Dropdown(["pos_vel", "pos_only"], label="Output Mode", value="pos_only")
                            fps_slider = gr.Slider(1, 60, value=20, label="FPS")
                        
                        with gr.Row():
                            num_workers_slider = gr.Slider(1, max_workers, value=min(4, max_workers), step=1, label="Parallel Workers")
                            gr.Markdown(f"⚠️ **Warning:** Higher values speed up conversion but use more RAM. Each worker loads one episode into memory. (Max: {max_workers} = 80% of {os.cpu_count()} cores)")
                        
                        with gr.Row():
                            btn_start_convert = gr.Button("🚀 Start Conversion", variant="primary")
                            btn_stop_convert = gr.Button("⏹️ Stop Conversion", variant="stop")
                        convert_status = gr.Textbox(label="Submission Status")

                        def on_system_type_change(st):
                            if "7-DoF" in st:
                                obs = "/sync/emily01/front/color/image_raw/compressed, /sync/emily01/head/color/image_raw/compressed, /sync/joint_states"
                                act = "/sync/ra_trajectory_controller/joint_trajectory, /sync/right_gripper_cmd"
                            else:
                                obs = "/sync/emily01/left_arm/color/image_raw/compressed, /sync/emily01/head/color/image_raw/compressed, /sync/joint_states"
                                act = "/sync/la_trajectory_controller/joint_trajectory"
                            return obs, act
                        
                        system_type.change(on_system_type_change, inputs=[system_type], outputs=[obs_topics, act_topics])

                        def on_browse_rosbag():
                            path = open_file_dialog(is_dir=True, initial_dir="/mnt/nas/rosbags")
                            return path if path else ""
                        
                        btn_browse_rosbag.click(on_browse_rosbag, outputs=[rosbag_path])
                        
                        def on_convert(rosbag_dir, od, st, dn, im, om, fps, obs, act, num_workers, progress=gr.Progress()):
                            if not rosbag_dir:
                                yield "Error: No rosbag path specified."; return
                            if not os.path.exists(rosbag_dir):
                                yield f"Error: Path does not exist: {rosbag_dir}"; return
                            
                            # Count total bags
                            total_bags = 0
                            try:
                                from pathlib import Path
                                for item in Path(rosbag_dir).iterdir():
                                    if item.is_dir():
                                        total_bags += 1
                            except:
                                total_bags = 1 # Fallback
                            
                            if total_bags == 0:
                                yield "Error: No bag directories found in source path."; return
                                
                            # Check if output directory already exists
                            full_output_dir = os.path.join(od, dn)
                            force = False
                            if os.path.exists(full_output_dir):
                                if confirm_overwrite_dialog(full_output_dir):
                                    force = True
                                else:
                                    yield f"Conversion aborted."; return
                                
                            cmd = generate_convert_cmd(rosbag_dir, od, st, dn, im, om, fps, obs, act, num_workers, force)
                            job_id = SCHEDULER.submit_job(f"Convert_{dn}", cmd)
                            if not job_id:
                                yield "Queue Full!"; return
                            yield f"Job {job_id} submitted. Check progress bar/logs."

                            # Simple native progress polling
                            last_count = -1
                            while True:
                                job = SCHEDULER.all_jobs.get(job_id)
                                if not job: break
                                
                                count = 0
                                if os.path.exists(job.log_file):
                                    try:
                                        with open(job.log_file, "r", errors="replace") as f:
                                            # More efficient counting for large logs
                                            for line in f:
                                                if "📁 Processing bag:" in line:
                                                    count += 1
                                    except:
                                        pass
                                
                                if count != last_count:
                                    progress(count / total_bags, desc=f"Converting {count}/{total_bags} bags")
                                    last_count = count
                                
                                if job.status in ["COMPLETED", "FAILED", "STOPPED", "CANCELLED"]:
                                    yield f"Finished: {job.status}"
                                    break
                                time.sleep(1)
                        
                        btn_start_convert.click(
                            on_convert, 
                            [rosbag_path, output_convert, system_type, dataset_name, input_mode, output_mode, fps_slider, obs_topics, act_topics, num_workers_slider], 
                            [convert_status]
                        )
                        btn_stop_convert.click(stop_active_job_ui, outputs=[convert_status])

                    # --- TAB 2: TRAIN ---
                    with gr.Tab("2. Train"):
                        gr.Markdown("Queue training jobs.")
                        gr.Markdown("[Monitor on Weights & Biases](https://wandb.ai/home)")
                    
                        with gr.Accordion("Dataset Config", open=True):
                            with gr.Row():
                                with gr.Column(scale=4):
                                    train_root_path = gr.Textbox(label="Dataset Root Path", placeholder="Click Browse to select...")
                                with gr.Column(scale=1, min_width=100):
                                    btn_browse_train_root = gr.Button("Browse", size="sm")
                            train_repo = gr.Textbox(label="Dataset Repo ID (Folder Name)")
                            train_out = create_out_input(label="Output Model Directory")
                            train_job_name = gr.Textbox(label="Job Name (for logs/W&B)", value="diffusion_test")
                        
                        with gr.Accordion("Hyperparameters", open=False):
                            policy_type = gr.Dropdown(["diffusion", "act"], label="Policy", value="diffusion")
                            action_mode = gr.Dropdown(["Relative (Recommended)", "Absolute"], label="Action Mode", value="Absolute")
                            arm_dim = gr.Slider(6, 7, value=7, step=1, label="Arm Dim")
                            backbone = gr.Dropdown(["resnet18", "resnet50"], label="Backbone", value="resnet18")
                            batch_size = gr.Slider(8, 256, value=32, step=8, label="Batch Size")
                            steps = gr.Number(value=100000, label="Training Steps")
                            workers = gr.Slider(1, 16, value=4, step=1, label="Num Workers")
                            with gr.Row():
                                obs_h = gr.Slider(1, 10, value=2, label="Obs Horizon")
                                pred_h = gr.Slider(16, 128, value=32, label="Pred Horizon")
                                act_h = gr.Slider(4, 64, value=16, label="Action Horizon")

                        with gr.Row():
                            btn_train = gr.Button("🚀 Start Training", variant="primary")
                            btn_stop_train = gr.Button("⏹️ Stop Training", variant="stop")
                        train_msg = gr.Textbox(label="Status", visible=True)

                        def on_browse_train_root():
                            path = open_file_dialog(is_dir=True)
                            return path if path else ""
                        
                        btn_browse_train_root.click(on_browse_train_root, outputs=[train_root_path])
                        
                        def on_train(dataset_root, repo, out, p_type, act, arm, back, bs, stp, wrk, oh, ph, ah, jn):
                            if not dataset_root:
                                return "Error: No dataset root path specified."
                            if not os.path.exists(dataset_root):
                                return f"Error: Path does not exist: {dataset_root}"
                            cmd = generate_train_cmd(dataset_root, repo, out, p_type, act, arm, back, bs, stp, wrk, oh, ph, ah, jn)
                            job_id = SCHEDULER.submit_job(f"Train_{jn}", cmd)
                            return f"Started {job_id}"
                        
                        btn_train.click(on_train, 
                                        [train_root_path, train_repo, train_out, policy_type, action_mode, arm_dim, backbone, batch_size, steps, workers, obs_h, pred_h, act_h, train_job_name],
                                        [train_msg])
                        btn_stop_train.click(stop_active_job_ui, outputs=[train_msg])

                    # --- TAB 3: EXPORT ---
                    with gr.Tab("3. Export"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                ckpt_dir_path = gr.Textbox(label="Checkpoint Directory", placeholder="Click Browse to select...")
                            with gr.Column(scale=1, min_width=100):
                                btn_browse_ckpt = gr.Button("Browse", size="sm")
                        export_out = create_out_input(label="Onnx Output Directory")
                        with gr.Row():
                            exp_type = gr.Dropdown(["Relative", "Absolute (7-DoF)"], label="Model Type", value="Absolute (7-DoF)")
                            exp_arm = gr.Dropdown([6, 7], value=7, label="Arm Dim (if relative)")
                            opset = gr.Dropdown([17, 14], value=17, label="Opset")
                        with gr.Row():
                            btn_export = gr.Button("🚀 Start Export", variant="primary")
                            btn_stop_export = gr.Button("⏹️ Stop Export", variant="stop")
                        exp_status = gr.Textbox(label="Status")
                        
                        def on_browse_ckpt():
                            path = open_file_dialog(is_dir=True)
                            return path if path else ""
                        btn_browse_ckpt.click(on_browse_ckpt, outputs=[ckpt_dir_path])
                        
                        def on_export(ckpt_path, out, mtype, arm, ops):
                            if not ckpt_path: return "Error: No checkpoint path specified."
                            cmd = generate_export_cmd(ckpt_path, out, mtype, arm, ops)
                            job_id = SCHEDULER.submit_job(f"Export", cmd)
                            return f"Started: {job_id}"
                        btn_export.click(on_export, [ckpt_dir_path, export_out, exp_type, exp_arm, opset], exp_status)
                        btn_stop_export.click(stop_active_job_ui, outputs=[exp_status])

                    # --- TAB 4: EVAL ---
                    with gr.Tab("4. Evaluate"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column(scale=4):
                                        eval_ckpt_path = gr.Textbox(label="Original Checkpoint Path", placeholder="Click Browse...")
                                    with gr.Column(scale=1, min_width=100):
                                        btn_browse_eval_ckpt = gr.Button("Browse", size="sm")
                                with gr.Row():
                                    with gr.Column(scale=4):
                                        eval_onnx_path = gr.Textbox(label="ONNX Directory", placeholder="Click Browse...")
                                    with gr.Column(scale=1, min_width=100):
                                        btn_browse_eval_onnx = gr.Button("Browse", size="sm")
                                with gr.Row():
                                    with gr.Column(scale=4):
                                        eval_bag_path = gr.Textbox(label="Test Rosbag Path", placeholder="Click Browse...")
                                    with gr.Column(scale=1, min_width=100):
                                        btn_browse_eval_bag = gr.Button("Browse", size="sm")
                                eval_type = gr.Dropdown(["Relative", "Absolute (7-DoF)"], label="Eval Type", value="Absolute (7-DoF)")
                                num_samples = gr.Slider(10, 500, value=100, label="Num Samples")
                                with gr.Row():
                                    btn_eval = gr.Button("🚀 Start Eval", variant="primary")
                                    btn_stop_eval = gr.Button("⏹️ Stop Eval", variant="stop")
                                eval_status = gr.Textbox(label="Status")
                            with gr.Column():
                                eval_plot = gr.Image(label="Evaluation Plot")
                        
                        def on_browse_eval_ckpt(): return open_file_dialog(is_dir=True) or ""
                        def on_browse_eval_onnx(): return open_file_dialog(is_dir=True) or ""
                        def on_browse_eval_bag(): return open_file_dialog(is_dir=True) or ""
                        
                        btn_browse_eval_ckpt.click(on_browse_eval_ckpt, outputs=[eval_ckpt_path])
                        btn_browse_eval_onnx.click(on_browse_eval_onnx, outputs=[eval_onnx_path])
                        btn_browse_eval_bag.click(on_browse_eval_bag, outputs=[eval_bag_path])
                        
                        def on_eval(ckpt_path, onnx_path, bag_path, etype, ns):
                            if not ckpt_path or not onnx_path or not bag_path: return "Error: Missing inputs."
                            cmd, save_path = generate_eval_cmd(ckpt_path, onnx_path, bag_path, etype, ns)
                            job_id = SCHEDULER.submit_job("Evaluate", cmd)
                            return f"Started: {job_id}. Plot will be saved to {save_path}"
                        btn_eval.click(on_eval, [eval_ckpt_path, eval_onnx_path, eval_bag_path, eval_type, num_samples], eval_status)
                        btn_stop_eval.click(stop_active_job_ui, outputs=[eval_status])

            # --- GLOBAL MANAGEMENT PANEL (RIGHT) ---
            with gr.Column(scale=2):
                gr.Markdown("### 🛠️ Global Logs & Monitor")
                
                with gr.Group():
                    gr.Markdown("#### Active Job Progress")
                    progress_display = gr.Textbox(label="Progress Indicator", value="Not running", interactive=False)

                gr.Markdown("#### Live Logs")
                log_display = gr.Code(language="shell", label="Combined Output", lines=25, elem_id="log_code")
                # Custom CSS to force height and scrollbar
                app.css = "#log_code { height: 500px !important; overflow-y: auto !important; }"

                # Timer and Events
                log_timer = gr.Timer(2)
                log_timer.tick(get_job_logs, outputs=[log_display])
                log_timer.tick(get_active_job_progress, outputs=[progress_display])

    return app

if __name__ == "__main__":
    ui = build_ui()
    ui.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
