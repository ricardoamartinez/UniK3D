# Performance Improvements: Decoupling Rendering from Model Inference

## Overview
The following changes have been implemented to ensure that the 3D scene rendering is completely independent from the model inference speed, preventing lag when the model outputs are slow.

## Key Changes

### 1. Reduced Queue Size (main_viewer.py)
- Changed queue size from 10 to 2 frames
- Enables more aggressive frame dropping when the model is slow
- Prevents accumulation of stale frames

### 2. Frame Dropping with Statistics (main_viewer.py)
- Implemented queue draining in the update method
- Only processes the latest frame, dropping intermediate frames
- Tracks and reports dropped frames for monitoring
- Added dropped frames counter to the overlay display

### 3. Non-blocking Queue Operations (inference_logic.py)
- Changed from blocking `put()` to non-blocking `put_nowait()`
- Frames are dropped at the producer side when queue is full
- Prevents inference thread from blocking on queue operations

### 4. Frame Timing Tracking (main_viewer.py)
- Added vertex update interval tracking
- Monitors the actual time between model updates
- Enables future interpolation features

### 5. Visual Feedback
- Added "Dropped Frames" counter to the overlay
- Shows when frame dropping is active
- Helps users understand performance characteristics

## Benefits

1. **Smooth Rendering**: The 3D scene renders at maximum FPS regardless of model speed
2. **No Accumulation**: Old frames are dropped instead of queuing up
3. **Real-time Feedback**: Users can see when frames are being dropped
4. **Responsive Controls**: Camera movement and UI remain responsive even when model is slow

## Usage

The system automatically manages frame dropping. Users will see:
- Smooth camera movement even when model inference is slow
- "Dropped: X" counter in the overlay when frames are being dropped
- Consistent rendering performance independent of model complexity

## Future Enhancements

1. **Frame Interpolation**: Could add interpolation between frames for even smoother motion
2. **Adaptive Quality**: Could reduce model input resolution when dropping too many frames
3. **Predictive Rendering**: Could predict future frames based on camera motion

## Technical Details

### Queue Management
```python
# Main thread drains queue and uses only latest frame
latest_data = None
frames_in_queue = 0
try:
    while True:
        data = self._data_queue.get_nowait()
        frames_in_queue += 1
        latest_data = data  # Keep only the latest
except queue.Empty:
    pass
```

### Non-blocking Producer
```python
# Inference thread uses non-blocking put
try:
    data_queue.put_nowait(data)
except queue.Full:
    print(f"Warning: Viewer queue full, dropping frame {frame_count}")
```

This architecture ensures that rendering performance is never limited by model inference speed. 