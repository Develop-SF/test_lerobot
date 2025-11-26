# ROS Inference Node Testing Guide - Approach Plate Pipeline

This guide walks you through testing the complete ROS inference pipeline with the cropped/rotated image preprocessing.

## Prerequisites

1. ROS2 environment is set up
2. LeRobot conda environment is activated
3. Rosbag data is available in `/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate`

## Image Processing Pipeline

This pipeline applies specific preprocessing to match the training data:

- **Head Camera (Top View)**: 
  - Crop to (x=247, y=122, w=193, h=237)
  - Rotate 90° clockwise
  - Final size: 237×193

- **Left Arm Camera**: 
  - Resize to 237×193

## Testing Steps

### Step 1: Launch ROS Inference Node (Terminal 1)

```bash
# Navigate to the inference directory
cd /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot/testing

# Option A: Use the launch script
./run_inference.sh

# Option B: Manual launch with custom parameters
conda activate lerobot
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python inference_node.py --checkpoint .. --device cuda --max-rate 30.0
```

**Expected Output:**
```
LeRobot Inference Node started (Approach Plate Pipeline)
Image preprocessing:
  - Head camera: Crop (247,122,193,237) + Rotate 90° CW → 237x193
  - Left camera: Resize → 237x193
Subscribing to:
  - /sync/emily01/left_arm/color/image_raw/compressed
  - /sync/emily01/head/color/image_raw/compressed
  - /sync/joint_states
Publishing to:
  - /left_arm/joint_trajectory
Press Ctrl+C to stop
```

### Step 2: Monitor Output Topic (Terminal 2)

```bash
# Monitor the trajectory commands being published
ros2 topic echo /left_arm/joint_trajectory
```

### Step 3: Play Rosbag Data (Terminal 3)

```bash
# Navigate to rosbag directory
cd /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate

# Play a rosbag (replace with actual rosbag name)
ros2 bag play sync_approach_plate_YYYY_MM_DD-HH_MM_SS \
    --topics /sync/emily01/left_arm/color/image_raw/compressed \
             /sync/emily01/head/color/image_raw/compressed \
             /sync/joint_states \
    --loop \
    --rate 1.0
```

## Expected Behavior

### Terminal 1 (Inference Node):
- Should show "LeRobot Inference Node ready" 
- No error messages
- May show inference timing information if enabled

### Terminal 2 (Topic Monitor):
You should see JointTrajectory messages like:
```yaml
header:
  stamp:
    sec: 1234567890
    nanosec: 123456789
  frame_id: base_link
joint_names:
- la_shoulder_pan_joint
- la_shoulder_lift_joint
- la_elbow_joint
- la_wrist_1_joint
- la_wrist_2_joint
- la_wrist_3_joint
points:
- positions: [-0.134, -1.770, -1.938, -1.234, 0.567, 2.890]
  velocities: []
  accelerations: []
  effort: []
  time_from_start:
    sec: 0
    nanosec: 100000000
```

### Terminal 3 (Rosbag Player):
- Shows playback progress
- Lists topics being published
- Runs in loop mode for continuous testing

## Advanced Testing

### Test with Rosbag Data (Offline)

Test the inference pipeline without running ROS nodes:

```bash
cd /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot/testing

# Test with a specific rosbag
python test_rosbag.py \
    --rosbag /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate/sync_approach_plate_YYYY_MM_DD-HH_MM_SS \
    --num-tests 10 \
    --device cuda
```

**Expected Output:**
```
Loading messages from: /path/to/rosbag
Loaded messages - Left: 50, Head: 50, Joints: 50
Running 10 inference tests...
Image preprocessing: Head camera cropped (247,122,193,237) + rotated 90° CW → 237x193
                     Left camera resized → 237x193
Test 1: SUCCESS - Pos: [-0.134 -1.770 -1.938] Time: 6.2ms
Test 2: SUCCESS - Pos: [-0.135 -1.771 -1.939] Time: 5.8ms
...
==================================================
TEST SUMMARY - APPROACH PLATE PIPELINE
==================================================
Total tests: 10
Successful: 10
Failed: 0
Success rate: 100.0%
Average inference time: 6.1ms
```

### Evaluate Predictions

Compare model predictions with ground truth:

```bash
cd /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot/testing

# Evaluate predictions
python evaluate_predictions.py \
    --rosbag /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate/sync_approach_plate_YYYY_MM_DD-HH_MM_SS \
    --num-samples 50 \
    --plot \
    --save-plot evaluation_results.png
```

This will:
- Load 50 samples from the rosbag
- Run inference on each sample
- Compare predictions with ground truth
- Calculate MAE, RMSE, and other metrics
- Generate plots (if --plot is specified)

## Troubleshooting

### Common Issues:

1. **"No module named 'trajectory_msgs'"**
   ```bash
   sudo apt install ros-humble-trajectory-msgs
   ```

2. **"Inference failed" errors**
   - Check GPU availability: `nvidia-smi`
   - Verify checkpoint path exists
   - Ensure all topics are being published

3. **No trajectory messages received**
   - Verify topic names match exactly
   - Check if all three input topics are active: `ros2 topic list`
   - Ensure sync node is working

4. **Image size mismatch errors**
   - Verify the rosbag was created with the same camera setup
   - Check that image topics contain valid compressed images

### Debug Commands:

```bash
# List active topics
ros2 topic list

# Check topic frequencies
ros2 topic hz /sync/emily01/left_arm/color/image_raw/compressed
ros2 topic hz /left_arm/joint_trajectory

# Check topic info
ros2 topic info /left_arm/joint_trajectory

# Monitor node status
ros2 node list
ros2 node info /lerobot_inference_node

# Check image dimensions (requires image_view)
ros2 run image_view image_view --ros-args --remap image:=/sync/emily01/left_arm/color/image_raw
```

## Performance Monitoring

Monitor inference performance:
```bash
# Check message rates
ros2 topic hz /left_arm/joint_trajectory

# Monitor system resources
htop
nvidia-smi -l 1  # For GPU usage (updates every second)

# Check inference latency
ros2 topic echo /left_arm/joint_trajectory --field header.stamp
```

## Test Success Criteria

1. **Inference Node Starts**: No errors on startup
2. **Topics Active**: All three input topics receiving data
3. **Trajectory Output**: JointTrajectory messages published to `/left_arm/joint_trajectory`
4. **Real-time Performance**: Trajectory published at reasonable rate (~10-30 Hz)
5. **Valid Data**: Trajectory contains 6 joint positions with proper joint names
6. **Image Processing**: Images correctly cropped, rotated, and resized

## Comparison with Original Pipeline

| Aspect | Original (approach_new) | New (approach_plate) |
|--------|------------------------|----------------------|
| Head Camera | Resize to 180×320 | Crop (247,122,193,237) + Rotate 90° CW → 237×193 |
| Left Camera | Resize to 180×320 | Resize to 237×193 |
| Final Size | 180×320 | 237×193 |
| Aspect Ratio | 16:9 | ~1:1 |

## Next Steps

Once testing is successful:
1. Integrate with robot controller
2. Subscribe to `/left_arm/joint_trajectory` in your control system
3. Deploy in production environment
4. Monitor performance and adjust rate limits as needed
5. Consider adding safety checks and error handling

## Example Integration Code

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.subscription = self.create_subscription(
            JointTrajectory,
            '/left_arm/joint_trajectory',
            self.trajectory_callback,
            10
        )
    
    def trajectory_callback(self, msg):
        if msg.points:
            positions = msg.points[0].positions  # 6D left arm positions
            joint_names = msg.joint_names
            
            # Send to robot controller
            self.get_logger().info(f'Received trajectory: {positions}')
            # self.robot_controller.execute_trajectory(joint_names, positions)

def main():
    rclpy.init()
    controller = RobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
