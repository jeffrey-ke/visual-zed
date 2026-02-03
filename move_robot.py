import argparse
import sys
import os

# Add parent directory to path to import xarm_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import util.misc_utils.xarm_utils as xu
    from xarm.wrapper import XArmAPI
except ImportError as e:
    print(f"Error importing xarm utilities: {e}")
    print("Make sure xarm-python-sdk is installed: pip install xarm-python-sdk")
    sys.exit(1)


def change_base_position(ip: str, offset_mm: float, axis: int = 0, speed: int = 50, wait: bool = True):
    """
    Change the base link (belt) position by the specified offset along the rail axis.
    
    Args:
        ip: IP address of the xArm
        offset_mm: Offset value in millimeters (positive or negative)
        axis: Axis index (0=X, 1=Y) - typically 0 for standard belt configuration
        speed: Movement speed (mm/s)
        wait: Whether to wait for movement to complete
    """
    axis_names = ['X', 'Y', 'Z']
    
    print(f"Connecting to xArm at {ip}...")
    
    # Initialize arm connection
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)  # Position mode
    arm.set_state(state=0)  # Ready state
    
    print(f"Getting current position...")
    
    # Get current position of the end-effector (in base frame)
    code, current_pos = arm.get_position()
    
    if code != 0:
        print(f"Error getting current position: {code}")
        arm.disconnect()
        return False
    
    # current_pos is [x, y, z, roll, pitch, yaw] in mm and degrees
    print(f"Current position: {current_pos[:3]} mm")
    print(f"  ({axis_names[0]}={current_pos[0]:.1f}, {axis_names[1]}={current_pos[1]:.1f}, {axis_names[2]}={current_pos[2]:.1f})")
    
    # Calculate new position by adding offset to the belt axis only
    new_pos = current_pos.copy()
    new_pos[axis] += offset_mm
    
    print(f"\nMoving belt along {axis_names[axis]}-axis by {offset_mm:+.1f} mm")
    print(f"Target position: {new_pos[:3]} mm")
    print(f"  ({axis_names[0]}={new_pos[0]:.1f}, {axis_names[1]}={new_pos[1]:.1f}, {axis_names[2]}={new_pos[2]:.1f})")
    
    # Move to new position
    print(f"\nExecuting movement at {speed} mm/s...")
    code = arm.set_position(*new_pos, speed=speed, wait=wait)
    
    if code != 0:
        print(f"Error moving to new position: {code}")
        arm.disconnect()
        return False
    
    print("✓ Belt movement completed successfully!")
    
    # Verify new position
    code, final_pos = arm.get_position()
    if code == 0:
        print(f"\nFinal position: {final_pos[:3]} mm")
        print(f"  ({axis_names[0]}={final_pos[0]:.1f}, {axis_names[1]}={final_pos[1]:.1f}, {axis_names[2]}={final_pos[2]:.1f})")
        actual_change = final_pos[axis] - current_pos[axis]
        print(f"  Actual {axis_names[axis]}-axis change: {actual_change:+.1f} mm")
    
    # Disconnect
    arm.disconnect()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Change xArm base link (belt) position by a specified offset along the rail",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Move belt 100mm forward
  python change_base_position.py --ip 192.168.1.xxx --offset 100
  
  # Move belt 50mm backward
  python change_base_position.py --ip 192.168.1.xxx --offset -50
  
  # Move with custom speed (100 mm/s)
  python change_base_position.py --ip 192.168.1.xxx --offset 100 --speed 100
  
  # If belt is mounted on Y-axis instead of X-axis
  python change_base_position.py --ip 192.168.1.xxx --offset 100 --axis 1
        """
    )
    
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="IP address of the xArm robot"
    )
    
    parser.add_argument(
        "--offset",
        type=float,
        required=True,
        help="Offset value in millimeters (positive or negative)"
    )
    
    parser.add_argument(
        "--axis",
        type=int,
        default=0,
        choices=[0, 1],
        help="Belt axis: 0=X (default), 1=Y"
    )
    
    parser.add_argument(
        "--speed",
        type=int,
        default=50,
        help="Movement speed in mm/s (default: 50)"
    )
    
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for movement to complete"
    )
    
    args = parser.parse_args()
    
    axis_names = ['X', 'Y']
    
    print("=" * 60)
    print("xArm Belt Position Changer")
    print("=" * 60)
    print(f"Robot IP:    {args.ip}")
    print(f"Belt Axis:   {axis_names[args.axis]}")
    print(f"Offset:      {args.offset:+.1f} mm")
    print(f"Speed:       {args.speed} mm/s")
    print("=" * 60)
    print()
    
    # Execute the base position change
    success = change_base_position(
        ip=args.ip,
        offset_mm=args.offset,
        axis=args.axis,
        speed=args.speed,
        wait=not args.no_wait
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Belt position change completed successfully!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Belt position change failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()