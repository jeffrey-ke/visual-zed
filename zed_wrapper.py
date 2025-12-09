"""
ZED Mini Camera Wrapper with Fault Tolerance
Implements resilient capture operations for ZED camera systems.
"""

import time
import random
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from contextlib import contextmanager
from copy import deepcopy
import pyzed.sl as sl

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: Core State and Error Management
# ============================================================================

class CameraState(Enum):
    """Explicit state machine for camera lifecycle."""
    UNINITIALIZED = "uninitialized"
    OPENING = "opening"
    READY = "ready"
    CAPTURING = "capturing"
    ERROR = "error"
    RECOVERING = "recovering"
    CLOSED = "closed"


class ErrorSeverity(Enum):
    """Classification of errors by recoverability."""
    TRANSIENT = "transient"      # Retry immediately (1-2 attempts)
    RECOVERABLE = "recoverable"  # Needs camera reset/reopen
    FATAL = "fatal"              # Manual intervention required


class CameraError(Exception):
    """Base exception for all camera errors."""
    pass


class CameraInitError(CameraError):
    """Camera failed to initialize."""
    pass


class CameraTimeoutError(CameraError):
    """Operation timed out."""
    pass


class CircuitOpenError(CameraError):
    """Circuit breaker is open - too many failures."""
    pass


class UnrecoverableCameraError(CameraError):
    """Camera cannot be recovered."""
    pass


def classify_error(error_code: sl.ERROR_CODE) -> ErrorSeverity:
    """
    Map ZED SDK error codes to recovery strategies.

    Reference: ZED SDK error codes documentation
    TRANSIENT: Temporary issues, retry immediately
    RECOVERABLE: Need to close and reopen camera
    FATAL: Hardware/driver issues needing manual intervention
    """
    transient_errors = {
        sl.ERROR_CODE.FAILURE,  # Generic failure, often transient
    }

    recoverable_errors = {
        sl.ERROR_CODE.CAMERA_NOT_DETECTED,
        sl.ERROR_CODE.INVALID_FUNCTION_CALL,  # Often precedes disconnect
        sl.ERROR_CODE.CAMERA_REBOOTING,
    }

    # Everything else is fatal until proven otherwise
    if error_code in transient_errors:
        return ErrorSeverity.TRANSIENT
    elif error_code in recoverable_errors:
        return ErrorSeverity.RECOVERABLE
    else:
        return ErrorSeverity.FATAL


# ============================================================================
# SECTION 2: Sequence Management and State Tracking
# ============================================================================

@dataclass
class CaptureResult:
    """Result of a single capture operation."""
    position_index: int
    timestamp: float
    image_left: Optional[Any] = None
    image_right: Optional[Any] = None
    depth_image: Optional[Any] = None
    depth_map: Optional[Any] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class CaptureSequence:
    """
    Tracks state of multi-capture sequence.
    Enables resume-on-failure without losing progress.
    """
    num_captures: int
    completed_captures: Dict[int, CaptureResult] = field(default_factory=dict)
    failed_captures: List[int] = field(default_factory=list)
    current_index: int = 0
    max_retries_per_capture: int = 3

    def is_complete(self) -> bool:
        """Check if sequence is complete."""
        return self.current_index >= self.num_captures

    def get_progress(self) -> Tuple[int, int]:
        """Return (completed, total) count."""
        return len(self.completed_captures), self.num_captures

    def mark_capture_complete(self, index: int, result: CaptureResult):
        """Mark a capture as successfully completed."""
        self.completed_captures[index] = result

    def mark_capture_failed(self, index: int):
        """Mark a capture as failed after max retries."""
        self.failed_captures.append(index)


# ============================================================================
# SECTION 3: Recovery Strategies and Circuit Breaker
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern: stop attempting operations if failure rate too high.
    Prevents wasting time on permanently broken hardware.

    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: Too many failures, reject all operations
    - HALF_OPEN: Testing if system recovered after timeout
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: float = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
        self._lock = threading.Lock()

    def call(self, func: Callable) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            if self.state == "open":
                # Check if we should try again
                if time.time() - self.last_failure_time > self.timeout_seconds:
                    logger.info("Circuit breaker entering half-open state")
                    self.state = "half-open"
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker open after {self.failure_count} failures"
                    )

        try:
            result = func()
            # Success - reset
            with self._lock:
                self.failure_count = 0
                self.state = "closed"
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures"
                    )
            raise


def retry_with_backoff(
    operation: Callable,
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 16.0
) -> Any:
    """
    Retry operation with exponential backoff and jitter.

    Delays: 1s, 2s, 4s, 8s, 16s (with random jitter)
    Jitter prevents thundering herd when multiple cameras fail simultaneously.
    """
    for attempt in range(max_attempts):
        try:
            return operation()
        except CameraError as e:
            if attempt == max_attempts - 1:
                logger.error(f"All {max_attempts} retry attempts failed")
                raise

            # Exponential backoff with jitter
            delay = min(initial_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 1)
            total_delay = delay + jitter

            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                f"Retrying in {total_delay:.1f}s"
            )
            time.sleep(total_delay)


# ============================================================================
# SECTION 4: Main ZedWrapper Class
# ============================================================================

class ZedWrapper:
    """
    Fault-tolerant wrapper for ZED Mini camera operations.

    Features:
    - Explicit state management
    - Circuit breaker for repeated failures
    - Sequence-aware recovery (resume after failure)
    - Multiple error handling strategies
    - Guaranteed resource cleanup
    - Comprehensive diagnostics

    Usage:
        with ZedWrapper() as zed:
            zed.validate_camera_ready()
            results = zed.capture_sequence(positions)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.camera: Optional[sl.Camera] = None
        self.init_params: Optional[sl.InitParameters] = None
        self.runtime_params: Optional[sl.RuntimeParameters] = None

        self._state = CameraState.UNINITIALIZED
        self._state_lock = threading.Lock()
        self._circuit_breaker = CircuitBreaker()

        self._error_history: List[tuple[float, sl.ERROR_CODE]] = []
        self._last_successful_grab_time: Optional[float] = None
        self._consecutive_failures = 0

        self._config = config or {}

    # ------------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------------

    def _transition_to(self, new_state: CameraState, allowed_from: Optional[List[CameraState]] = None):
        """Transition to new state with validation."""
        with self._state_lock:
            if allowed_from and self._state not in allowed_from:
                raise ValueError(
                    f"Invalid state transition: {self._state} -> {new_state}. "
                    f"Allowed from: {allowed_from}"
                )
            logger.debug(f"State transition: {self._state} -> {new_state}")
            self._state = new_state

    @property
    def state(self) -> CameraState:
        """Get current camera state."""
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------------
    # Initialization and Cleanup (Context Manager)
    # ------------------------------------------------------------------------

    def __enter__(self):
        """Context manager entry - opens camera."""
        self.open_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - guarantees cleanup even on exception."""
        try:
            if self.camera is not None:
                logger.info("Closing camera...")
                self.camera.close()
        except Exception as e:
            logger.error(f"Error during camera cleanup: {e}")
        finally:
            self.camera = None
            self._transition_to(CameraState.CLOSED)

    def _create_init_parameters(self) -> sl.InitParameters:
        """Create initialization parameters with defaults."""
        init = sl.InitParameters()
        init.camera_resolution = self._config.get('resolution', sl.RESOLUTION.HD720)
        init.depth_mode = self._config.get('depth_mode', sl.DEPTH_MODE.NEURAL)
        init.coordinate_units = self._config.get('units', sl.UNIT.MILLIMETER)
        init.depth_minimum_distance = self._config.get('min_depth', 0.05)
        # Neural depth optimization can take 12+ minutes on first run
        init.open_timeout_sec = self._config.get('open_timeout_sec', 720)
        return init

    def open_camera(self, init_params: Optional[sl.InitParameters] = None):
        """
        Open camera with comprehensive error handling.

        Raises:
            CameraInitError: If camera fails to initialize
        """
        if self.state != CameraState.UNINITIALIZED:
            raise ValueError(f"Camera already opened (state: {self.state})")

        self._transition_to(CameraState.OPENING)

        try:
            self.camera = sl.Camera()
            self.init_params = init_params or self._create_init_parameters()

            logger.info("Opening ZED camera...")
            err = self.camera.open(self.init_params)

            # Check for errors (use equality since older SDK doesn't support < >)
            if err != sl.ERROR_CODE.SUCCESS:
                # Check if it's a warning (SDK 4.0+ returns warnings as positive codes)
                # For older SDKs, treat any non-SUCCESS as error
                diagnostic_info = self._gather_diagnostic_info()
                raise CameraInitError(
                    f"Failed to open camera: {err}\n"
                    f"Diagnostics: {diagnostic_info}"
                )

            # Create runtime parameters
            self.runtime_params = sl.RuntimeParameters()

            self._transition_to(CameraState.READY)
            logger.info("Camera opened successfully")

        except Exception as e:
            self.camera = None
            self._transition_to(CameraState.ERROR)
            raise CameraInitError(f"Failed to open camera: {e}") from e

    def _gather_diagnostic_info(self) -> Dict[str, Any]:
        """Collect diagnostic information for debugging initialization failures."""
        info = {}
        try:
            info['sdk_version'] = sl.Camera.get_sdk_version()
        except Exception:
            pass
        try:
            info['devices'] = sl.Camera.get_device_list()
        except Exception:
            pass
        try:
            # is_cuda_available may not exist in older SDK versions
            if hasattr(sl.Camera, 'is_cuda_available'):
                info['cuda_available'] = sl.Camera.is_cuda_available()
        except Exception:
            pass
        return info

    # ------------------------------------------------------------------------
    # Pre-Flight Validation
    # ------------------------------------------------------------------------

    def validate_camera_ready(self):
        """
        Comprehensive check before starting capture sequence.
        Fail-fast: detect problems early before proceeding with captures.

        Raises:
            CameraError: If any validation check fails
        """
        if self.state != CameraState.READY:
            raise ValueError(f"Cannot validate in state {self.state}")

        checks = [
            ("Camera is opened", lambda: self.camera.is_opened()),
            ("USB bandwidth sufficient", self._check_usb_bandwidth),
            ("Can grab test frame", self._test_single_grab),
            ("Can retrieve left image", self._test_retrieve_image),
            ("Can get camera intrinsics", self._test_get_intrinsics),
        ]

        logger.info("Running pre-flight validation checks...")
        for check_name, check_fn in checks:
            try:
                result = check_fn()
                if result is False:
                    raise CameraError(f"Validation failed: {check_name}")
                logger.debug(f"✓ {check_name}")
            except Exception as e:
                raise CameraError(f"Validation failed: {check_name} - {e}") from e

        logger.info("✓ All pre-flight checks passed - camera ready for sequence")

    def _check_usb_bandwidth(self) -> bool:
        """
        Check USB connection speed for ZED camera.
        
        ZED cameras require USB 3.0 for adequate bandwidth:
        - USB 3.0 (SuperSpeed): 5 Gbps - OK for all resolutions
        - USB 2.0 (HighSpeed): 480 Mbps - INSUFFICIENT for HD modes
        
        Returns:
            True if USB 3.0 detected or check cannot be performed
            False if USB 2.0 detected (will likely cause LOW USB BANDWIDTH error)
        """
        import subprocess
        import re
        
        try:
            # Get ZED camera serial from SDK
            info = self.camera.get_camera_information()
            serial = str(info.serial_number)
            
            # Find ZED device in USB tree using lsusb
            # ZED cameras use vendor ID 2b03 (Stereolabs)
            result = subprocess.run(
                ['lsusb', '-d', '2b03:'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning("Could not find ZED camera in USB device list")
                return True  # Can't check, proceed anyway
            
            # Parse bus and device number from lsusb output
            # Format: "Bus 001 Device 002: ID 2b03:f582 ..."
            match = re.search(r'Bus (\d+) Device (\d+)', result.stdout)
            if not match:
                logger.warning("Could not parse USB bus/device info")
                return True
            
            bus = match.group(1)
            
            # Check USB speed from sysfs
            # Path: /sys/bus/usb/devices/X-Y/speed where X is bus number
            # We need to find the device path - search for ZED's vendor
            speed_paths = list(__import__('pathlib').Path('/sys/bus/usb/devices').glob('*/speed'))
            
            for speed_path in speed_paths:
                device_path = speed_path.parent
                vendor_path = device_path / 'idVendor'
                
                if vendor_path.exists():
                    vendor = vendor_path.read_text().strip()
                    if vendor == '2b03':  # Stereolabs vendor ID
                        speed = speed_path.read_text().strip()
                        speed_mbps = int(speed)
                        
                        logger.info(f"ZED camera USB speed: {speed_mbps} Mbps")
                        
                        if speed_mbps < 5000:  # Less than USB 3.0
                            logger.error(
                                f"ZED camera connected at USB 2.0 speed ({speed_mbps} Mbps). "
                                f"USB 3.0 (5000 Mbps) required for reliable operation. "
                                f"Connect to a USB 3.0 port (blue connector)."
                            )
                            return False
                        
                        return True
            
            logger.warning("Could not determine ZED USB speed from sysfs")
            return True  # Can't verify, proceed anyway
            
        except subprocess.TimeoutExpired:
            logger.warning("USB check timed out")
            return True
        except FileNotFoundError:
            logger.warning("lsusb not available, skipping USB speed check")
            return True
        except Exception as e:
            logger.warning(f"USB bandwidth check failed: {e}")
            return True  # Don't block on check failure

    def _test_single_grab(self) -> bool:
        """Test that we can grab a single frame."""
        err = self.camera.grab(self.runtime_params)
        return err == sl.ERROR_CODE.SUCCESS

    def _test_retrieve_image(self) -> bool:
        """Test that we can retrieve image data."""
        image = sl.Mat()
        err = self.camera.retrieve_image(image, sl.VIEW.LEFT)
        return err == sl.ERROR_CODE.SUCCESS

    def _test_get_intrinsics(self) -> bool:
        """Test that we can get camera intrinsics."""
        try:
            info = self.camera.get_camera_information()
            calib = info.camera_configuration.calibration_parameters
            return calib is not None
        except Exception:
            return False

    # ------------------------------------------------------------------------
    # Health Monitoring
    # ------------------------------------------------------------------------

    def is_camera_healthy(self) -> bool:
        """
        Proactive health check without full capture.
        Detects issues before they cause operation failures.
        """
        try:
            if not self.camera or not self.camera.is_opened():
                return False

            # Check for stale timestamps (known issue from forums)
            if self._last_successful_grab_time:
                time_since_last_grab = time.time() - self._last_successful_grab_time
                if time_since_last_grab > 30:
                    logger.warning(
                        f"No successful grabs in {time_since_last_grab:.1f}s - "
                        "camera may be unresponsive"
                    )
                    return False

            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ------------------------------------------------------------------------
    # Core Capture Operations
    # ------------------------------------------------------------------------

    def capture_image(self) -> CaptureResult:
        """
        Capture single image set (left, right, depth).

        Returns:
            CaptureResult with image data

        Raises:
            CameraError: If capture fails
        """
        if self.state not in [CameraState.READY]:
            raise ValueError(f"Cannot capture in state {self.state}")

        self._transition_to(CameraState.CAPTURING, allowed_from=[CameraState.READY])

        try:
            # Execute through circuit breaker
            result = self._circuit_breaker.call(self._do_capture)
            self._transition_to(CameraState.READY)
            return result
        except Exception as e:
            self._transition_to(CameraState.ERROR)
            raise

    def _do_capture(self) -> CaptureResult:
        """
        Internal capture implementation with error classification.
        """
        # Grab frame from camera
        err = self.camera.grab(self.runtime_params)

        # Multi-level error detection (forums: check multiple error codes)
        if err == sl.ERROR_CODE.SUCCESS:
            # Success path
            pass
        elif err == sl.ERROR_CODE.INVALID_FUNCTION_CALL:
            # Known issue: this often precedes CAMERA_NOT_DETECTED
            logger.warning("INVALID_FUNCTION_CALL - checking camera state")
            if not self.camera.is_opened():
                err = sl.ERROR_CODE.CAMERA_NOT_DETECTED

        # Handle errors by severity
        if err != sl.ERROR_CODE.SUCCESS:
            severity = classify_error(err)
            self._record_error(err)

            if severity == ErrorSeverity.TRANSIENT:
                # Retry immediately for transient errors
                logger.info(f"Transient error {err}, retrying once...")
                time.sleep(0.5)
                err = self.camera.grab(self.runtime_params)
                if err != sl.ERROR_CODE.SUCCESS:
                    raise CameraError(f"Grab failed after retry: {err}")

            elif severity == ErrorSeverity.RECOVERABLE:
                # Need full recovery
                raise CameraError(f"Camera requires recovery: {err}")

            else:
                # Fatal error
                raise UnrecoverableCameraError(f"Fatal camera error: {err}")

        # Retrieve images
        image_left = sl.Mat()
        image_right = sl.Mat()
        depth_image = sl.Mat()
        depth_map = sl.Mat()

        self.camera.retrieve_image(image_left, sl.VIEW.LEFT)
        self.camera.retrieve_image(image_right, sl.VIEW.RIGHT)
        self.camera.retrieve_image(depth_image, sl.VIEW.DEPTH)
        self.camera.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        # Convert to numpy arrays
        result = CaptureResult(
            position_index=-1,  # Will be set by sequence manager
            timestamp=time.time(),
            image_left=deepcopy(image_left.get_data()),
            image_right=deepcopy(image_right.get_data()),
            depth_image=deepcopy(depth_image.get_data()),
            depth_map=deepcopy(depth_map.get_data()),
            success=True
        )

        self._last_successful_grab_time = time.time()
        self._consecutive_failures = 0

        return result

    def _record_error(self, err: sl.ERROR_CODE):
        """Record error for diagnostics."""
        self._error_history.append((time.time(), err))
        self._consecutive_failures += 1

        # Keep only recent history
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]

    # ------------------------------------------------------------------------
    # Recovery Operations
    # ------------------------------------------------------------------------

    def _full_recovery(self):
        """
        Attempt full camera recovery: close and reopen.
        Note: ZED Mini does not support software reboot via camera.reboot()
        """
        logger.info("Attempting full camera recovery...")
        self._transition_to(CameraState.RECOVERING)

        try:
            # Close camera
            if self.camera:
                try:
                    self.camera.close()
                except Exception as e:
                    logger.error(f"Error closing camera: {e}")

            # Wait for hardware to reset
            time.sleep(2)

            # Reopen
            self.camera = sl.Camera()
            err = self.camera.open(self.init_params)

            if err != sl.ERROR_CODE.SUCCESS:
                raise CameraError(f"Recovery failed: {err}")

            logger.info("Camera recovery successful")
            self._transition_to(CameraState.READY)

        except Exception as e:
            self._transition_to(CameraState.ERROR)
            raise CameraError(f"Recovery failed: {e}") from e

    # ------------------------------------------------------------------------
    # Sequence Capture with Resilience
    # ------------------------------------------------------------------------

    def capture_sequence(
        self,
        num_captures: int,
        skip_on_failure: bool = False,
        delay_between_captures: float = 0.0
    ) -> CaptureSequence:
        """
        Capture multiple images with fault tolerance.

        Args:
            num_captures: Number of images to capture
            skip_on_failure: If True, skip failed captures. If False, abort on failure.
            delay_between_captures: Optional delay between captures in seconds

        Returns:
            CaptureSequence with results

        Raises:
            UnrecoverableCameraError: If camera cannot be recovered and skip_on_failure=False
        """
        sequence = CaptureSequence(num_captures=num_captures)

        logger.info(f"Starting capture sequence: {num_captures} captures")

        while not sequence.is_complete():
            capture_idx = sequence.current_index

            logger.info(f"Capture {capture_idx + 1}/{num_captures}")

            # Attempt capture with retries
            capture_succeeded = False
            for retry in range(sequence.max_retries_per_capture):
                try:
                    result = self.capture_image()
                    result.position_index = capture_idx
                    sequence.mark_capture_complete(capture_idx, result)
                    capture_succeeded = True
                    break

                except CameraError as e:
                    logger.warning(
                        f"Capture failed at index {capture_idx}, "
                        f"attempt {retry + 1}/{sequence.max_retries_per_capture}: {e}"
                    )

                    # Try recovery if not last retry
                    if retry < sequence.max_retries_per_capture - 1:
                        try:
                            self._full_recovery()
                        except Exception as recovery_error:
                            logger.error(f"Recovery failed: {recovery_error}")

                except CircuitOpenError as e:
                    logger.error(f"Circuit breaker open: {e}")
                    if skip_on_failure:
                        sequence.mark_capture_failed(capture_idx)
                        break
                    else:
                        raise UnrecoverableCameraError(
                            f"Camera failure at capture {capture_idx}: {e}"
                        ) from e

            # Handle persistent failure at this capture
            if not capture_succeeded:
                if skip_on_failure:
                    logger.warning(f"Skipping capture {capture_idx} after max retries")
                    sequence.mark_capture_failed(capture_idx)
                else:
                    completed, total = sequence.get_progress()
                    raise UnrecoverableCameraError(
                        f"Failed to capture index {capture_idx}. "
                        f"Completed {completed}/{total} captures."
                    )

            sequence.current_index += 1

            # Optional delay between captures
            if delay_between_captures > 0 and not sequence.is_complete():
                time.sleep(delay_between_captures)

        completed, total = sequence.get_progress()
        logger.info(
            f"Sequence complete: {completed}/{total} successful, "
            f"{len(sequence.failed_captures)} failed"
        )

        return sequence

    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------

    def get_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsic parameters."""
        if self.state != CameraState.READY:
            raise ValueError(f"Cannot get intrinsics in state {self.state}")

        info = self.camera.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam

        return {
            'fx': calib.fx,
            'fy': calib.fy,
            'cx': calib.cx,
            'cy': calib.cy,
        }


# ============================================================================
# SECTION 5: Usage Example
# ============================================================================

def example_usage():
    """
    Example of how to use ZedWrapper for image capture.
    """
    # Configure camera
    config = {
        'resolution': sl.RESOLUTION.HD720,
        'depth_mode': sl.DEPTH_MODE.NEURAL,
        'units': sl.UNIT.MILLIMETER,
        'min_depth': 0.05,
    }

    # Execute capture with automatic resource management
    try:
        with ZedWrapper(config=config) as zed:
            # Pre-flight validation
            zed.validate_camera_ready()

            # Capture a single image
            result = zed.capture_image()
            print(f"Captured image at {result.timestamp}")
            print(f"Left image shape: {result.image_left.shape}")
            print(f"Right image shape: {result.image_right.shape}")
            print(f"Depth map shape: {result.depth_map.shape}")

            # Or capture a sequence of images
            sequence = zed.capture_sequence(
                num_captures=5,
                skip_on_failure=True,
                delay_between_captures=0.5
            )

            # Process results
            for idx, result in sequence.completed_captures.items():
                print(f"Capture {idx}: captured at {result.timestamp}")

            if sequence.failed_captures:
                print(f"Failed captures: {sequence.failed_captures}")

    except UnrecoverableCameraError as e:
        logger.error(f"Camera failure aborted capture: {e}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    example_usage()