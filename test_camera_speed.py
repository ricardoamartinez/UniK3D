#!/usr/bin/env python3
"""
Quick camera speed test - finds the fastest OpenCV backend for camera access.
"""

import time
import cv2

def test_camera_backend(backend, name):
    """Test a specific camera backend and return timing info."""
    print(f"\n🔍 Testing {name} backend...")
    
    start_time = time.time()
    cap = None
    
    try:
        # Test opening camera
        open_start = time.time()
        cap = cv2.VideoCapture(0, backend)
        open_time = time.time() - open_start
        print(f"  📹 Open time: {open_time:.3f}s")
        
        if not cap.isOpened():
            print(f"  ❌ Failed to open camera")
            return None, open_time, 0, 0
        
        # Test configuration
        config_start = time.time()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        config_time = time.time() - config_start
        print(f"  ⚙️ Config time: {config_time:.3f}s")
        
        # Test frame capture
        frame_start = time.time()
        ret, frame = cap.read()
        frame_time = time.time() - frame_start
        print(f"  📸 Frame time: {frame_time:.3f}s")
        
        if not ret or frame is None:
            print(f"  ❌ Failed to capture frame")
            return None, open_time, config_time, frame_time
        
        total_time = time.time() - start_time
        print(f"  ✅ SUCCESS! Total time: {total_time:.3f}s")
        
        return cap, open_time, config_time, frame_time
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"  ❌ Error: {e} (after {error_time:.3f}s)")
        return None, error_time, 0, 0
    finally:
        if cap:
            cap.release()

def main():
    print("🚀 Camera Speed Test - Finding fastest backend")
    print("=" * 50)
    
    # Test different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MSMF"), 
        (cv2.CAP_V4L2, "Video4Linux2"),
        (cv2.CAP_ANY, "Auto-detect"),
    ]
    
    results = []
    
    for backend, name in backends:
        try:
            cap, open_time, config_time, frame_time = test_camera_backend(backend, name)
            total_time = open_time + config_time + frame_time
            
            results.append({
                'backend': backend,
                'name': name,
                'success': cap is not None,
                'open_time': open_time,
                'config_time': config_time,
                'frame_time': frame_time,
                'total_time': total_time
            })
        except Exception as e:
            print(f"  ❌ Backend {name} threw exception: {e}")
    
    # Show results
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)
    
    successful = [r for r in results if r['success']]
    if successful:
        # Sort by total time
        successful.sort(key=lambda x: x['total_time'])
        
        print("✅ Working backends (fastest first):")
        for i, result in enumerate(successful):
            print(f"{i+1}. {result['name']:15} | Total: {result['total_time']:.3f}s | "
                  f"Open: {result['open_time']:.3f}s | Frame: {result['frame_time']:.3f}s")
        
        best = successful[0]
        print(f"\n🏆 WINNER: {best['name']} (Total: {best['total_time']:.3f}s)")
        
    else:
        print("❌ No backends worked!")
    
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\n❌ Failed backends:")
        for result in failed:
            print(f"   {result['name']}: {result['open_time']:.3f}s")

if __name__ == "__main__":
    main() 