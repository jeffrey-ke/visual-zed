# Save as test_zed_minimal.py
import pyzed.sl as sl

print("Testing ZED SDK directly...")
print(f"SDK Version: {sl.Camera.get_sdk_version()}")

# Try to list devices
cameras = sl.Camera.get_device_list()
print(f"Devices found: {len(cameras)}")
for i, cam in enumerate(cameras):
    print(f"  Camera {i}: {cam.serial_number}")

# Try to open
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode = sl.DEPTH_MODE.PERFORMANCE

status = zed.open(init)
print(f"Open status: {status}")

if status == sl.ERROR_CODE.SUCCESS:
    print("✓ Camera opened successfully!")
    zed.close()
else:
    print(f"✗ Failed: {status}")