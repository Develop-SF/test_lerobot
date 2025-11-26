# Deployment Guide for Approach Plate Model

This guide explains how to deploy the trained LeRobot model from the approach_plate dataset.

## Quick Start

### 1. Test the Inference Module

First, verify that the inference module works correctly:

```bash
cd /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot/testing

# Run the test script
python test_inference.py --checkpoint ../output/checkpoints/last --device cuda
```

This will:
- Test image preprocessing pipeline
- Load the trained model
- Verify input/output dimensions
- Run a test inference

### 2. Deploy with ROS2

Once testing is successful, deploy the model with ROS2:

```bash
# Continuous mode (recommended for real-time control)
python inference_node.py \
    --checkpoint /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot \
    --device cuda \
    --frequency 20.0 \
    --mode continuous
```

## Deployment Modes

### Continuous Mode (Recommended)

Runs inference at a fixed frequency (e.g., 20 Hz):

```bash
python inference_node.py \
    --checkpoint /path/to/checkpoint \
    --device cuda \
    --frequency 20.0 \
    --mode continuous
```

**Pros**:
- Predictable timing
- Consistent control frequency
- Better for closed-loop control

**Cons**:
- May process stale sensor data if sensors are slower than inference frequency

### Triggered Mode

Runs inference whenever new sensor data arrives:

```bash
python inference_node.py \
    --checkpoint /path/to/checkpoint \
    --device cuda \
    --mode triggered
```

**Pros**:
- Always uses latest sensor data
- Adapts to sensor frequency

**Cons**:
- Variable inference frequency
- May be too fast or too slow depending on sensor rate

## Image Preprocessing Details

The inference code applies the **exact same preprocessing** as the training pipeline:

### Top View (Head Camera)
1. **Crop**: Extract region `(x=247, y=122, w=193, h=237)` from 640×480 image
2. **Rotate**: Rotate 90° clockwise
3. **Result**: 237×193 image

### Left Arm Camera
1. **Resize**: Resize from 640×480 to 237×193
2. **Result**: 237×193 image

Both cameras output **237×193** resolution to the model.

## ROS2 Topics

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/sync/emily01/left_arm/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | Left arm camera |
| `/sync/emily01/head/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | Head camera (top view) |
| `/sync/joint_states` | `sensor_msgs/JointState` | Joint states (if needed by model) |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/left_arm/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Predicted joint commands |

## Model Modes

The inference system automatically detects the model's input and output modes from the checkpoint.

### Input Modes

- **`vision_only`**: Only camera images (no joint states required)
- **`vision_pos`**: Camera images + joint positions (6D)
- **`vision_pos_vel`**: Camera images + joint positions + velocities (12D)

### Output Modes

- **`pos_only`**: Joint positions only (6D)
- **`pos_vel`**: Joint positions + velocities (12D)

## Troubleshooting

### Issue: Model not found

**Error**: `FileNotFoundError: No valid checkpoint found`

**Solution**: Check that the checkpoint path is correct. The path should point to either:
- The output directory (e.g., `approach_plate_lerobot/`)
- The checkpoints directory (e.g., `approach_plate_lerobot/output/checkpoints/last`)

### Issue: Image size mismatch

**Error**: `Image size doesn't match expected`

**Solution**: This usually means the preprocessing is not being applied correctly. Verify:
1. The `is_top_view` flag is set correctly when decoding images
2. The crop box parameters match the training pipeline
3. The target size is 237×193

### Issue: CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: 
1. Use CPU instead: `--device cpu`
2. Reduce inference frequency: `--frequency 10.0`
3. Close other GPU-intensive applications

### Issue: Inference too slow

**Symptoms**: Inference loop running slower than target frequency

**Solutions**:
1. Reduce inference frequency: `--frequency 10.0`
2. Use GPU if available: `--device cuda`
3. Check if other processes are using GPU
4. Consider using triggered mode instead

### Issue: Robot not moving

**Possible causes**:
1. Check that the action publisher topic is correct (`/left_arm/joint_trajectory`)
2. Verify that the robot controller is subscribed to the topic
3. Check that the model is producing non-zero actions
4. Verify that all required sensor topics are publishing data

## Performance Optimization

### GPU Optimization

```bash
# Use CUDA with optimized settings
export CUDA_VISIBLE_DEVICES=0
python inference_node.py --checkpoint /path/to/checkpoint --device cuda
```

### CPU Optimization

```bash
# Use all CPU cores
export OMP_NUM_THREADS=8
python inference_node.py --checkpoint /path/to/checkpoint --device cpu
```

### Frequency Tuning

Start with a conservative frequency and increase gradually:

```bash
# Start conservative
python inference_node.py --frequency 10.0 --mode continuous

# Increase if system can handle it
python inference_node.py --frequency 20.0 --mode continuous

# Maximum (if system is fast enough)
python inference_node.py --frequency 30.0 --mode continuous
```

## Monitoring

### Check Inference Performance

Add logging to monitor inference timing:

```python
# In inference_node.py, the loop already logs warnings if running slow
# Watch for: "Inference loop running slower than X Hz"
```

### Check Topic Rates

```bash
# Check camera topic rates
ros2 topic hz /sync/emily01/left_arm/color/image_raw/compressed
ros2 topic hz /sync/emily01/head/color/image_raw/compressed

# Check action publishing rate
ros2 topic hz /left_arm/joint_trajectory
```

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

## Safety Considerations

1. **Always test in simulation first** before deploying on real hardware
2. **Start with low inference frequency** (e.g., 5-10 Hz) and increase gradually
3. **Monitor robot behavior** closely during initial deployment
4. **Have emergency stop ready** in case of unexpected behavior
5. **Verify action limits** are within safe ranges for your robot

## Next Steps

After successful deployment:

1. **Collect performance metrics**: Track success rate, inference time, etc.
2. **Fine-tune parameters**: Adjust inference frequency, control gains, etc.
3. **Iterate on training**: Use deployment insights to improve training data
4. **Scale up**: Once stable, increase frequency or add more complex tasks

## Support

For issues or questions:
1. Check the logs for error messages
2. Review the `PIPELINE_COMPARISON.md` for pipeline differences
3. Verify preprocessing matches training pipeline
4. Test with the `test_inference.py` script first
