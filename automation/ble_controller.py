"""
BLE Controller module for ESP32 wireless input bridge.

Handles BLE UART communication with ESP32 device to send keyboard and mouse commands.
Uses the bleak library for cross-platform BLE support.
"""

import asyncio
import logging
from typing import Optional
from bleak import BleakClient, BleakScanner


logger = logging.getLogger(__name__)


class BLEController:
    """
    Controller for ESP32 BLE UART communication.

    Manages connection to ESP32 device and sends keyboard/mouse commands
    via BLE UART protocol.
    """

    def __init__(
        self,
        device_name: str,
        service_uuid: str,
        rx_char_uuid: str,
        tx_char_uuid: str
    ):
        """
        Initialize BLE controller.

        Args:
            device_name: Name of ESP32 BLE device to connect to
            service_uuid: BLE service UUID
            rx_char_uuid: RX characteristic UUID (for writing commands)
            tx_char_uuid: TX characteristic UUID (for notifications)
        """
        self.device_name = device_name
        self.service_uuid = service_uuid
        self.rx_char_uuid = rx_char_uuid
        self.tx_char_uuid = tx_char_uuid

        self.client: Optional[BleakClient] = None
        self.device_address: Optional[str] = None
        self.current_mode: str = "mouse"  # Default mode on ESP32 is mouse

        # Tracked pixel position for relative moves (updated by move_mouse_absolute)
        self._current_x: int = 0
        self._current_y: int = 0

    async def scan_and_find_device(self, timeout: float = 10.0) -> Optional[str]:
        """
        Scan for BLE devices and find the target ESP32 device.

        Args:
            timeout: Scan timeout in seconds

        Returns:
            Device address if found, None otherwise
        """
        logger.info(f"Scanning for BLE device: {self.device_name}")

        devices = await BleakScanner.discover(timeout=timeout)

        for device in devices:
            logger.debug(f"Found device: {device.name} ({device.address})")
            if device.name == self.device_name:
                logger.info(f"Found target device: {device.name} at {device.address}")
                return device.address

        logger.warning(f"Device '{self.device_name}' not found")
        return None

    async def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to ESP32 BLE device.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Scan for device
            self.device_address = await self.scan_and_find_device(timeout)
            if not self.device_address:
                logger.error(f"Failed to find device: {self.device_name}")
                return False

            # Connect to device
            logger.info(f"Connecting to {self.device_address}...")
            self.client = BleakClient(self.device_address)
            await self.client.connect()

            if self.client.is_connected:
                logger.info(f"Successfully connected to {self.device_name}")
                return True
            else:
                logger.error(f"Failed to connect to {self.device_address}")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from ESP32 BLE device."""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            logger.info(f"Disconnected from {self.device_name}")
        self.client = None
        self.device_address = None

    async def send_command(self, command: str) -> bool:
        """
        Send a command to ESP32 via BLE UART.

        Args:
            command: Command string to send

        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.client or not self.client.is_connected:
            logger.error("Not connected to BLE device")
            return False

        try:
            # Convert command to bytes
            data = command.encode('utf-8')

            # Write to RX characteristic
            await self.client.write_gatt_char(self.rx_char_uuid, data)
            logger.debug(f"Sent command: {command}")
            return True

        except Exception as e:
            logger.error(f"Failed to send command '{command}': {e}")
            return False

    async def switch_to_mouse_mode(self) -> bool:
        """
        Switch ESP32 to mouse mode.

        Returns:
            True if successful, False otherwise
        """
        success = await self.send_command("mode:mouse")
        if success:
            self.current_mode = "mouse"
            logger.info("Switched to mouse mode")
        return success

    async def switch_to_keyboard_mode(self) -> bool:
        """
        Switch ESP32 to keyboard mode.

        Returns:
            True if successful, False otherwise
        """
        success = await self.send_command("mode:keyboard")
        if success:
            self.current_mode = "keyboard"
            logger.info("Switched to keyboard mode")
        return success

    async def type_text(self, text: str) -> bool:
        """
        Type text using keyboard.

        Args:
            text: Text to type

        Returns:
            True if successful, False otherwise
        """
        return await self.send_command(f"type:{text}")

    async def press_key(self, key: str) -> bool:
        """
        Press a special key.

        Args:
            key: Key name (enter, tab, backspace, delete, esc)

        Returns:
            True if successful, False otherwise
        """
        return await self.send_command(f"key:{key}")

    async def click(self) -> bool:
        """Perform left mouse click."""
        return await self.send_command("click")

    async def double_click(self) -> bool:
        """Perform double left mouse click."""
        ok1 = await self.send_command("click")
        await asyncio.sleep(0.1)
        ok2 = await self.send_command("click")
        return ok1 and ok2

    async def right_click(self) -> bool:
        """Perform right mouse click."""
        return await self.send_command("rclick")

    async def move_mouse(self, x: int = 0, y: int = 0,
                         screen_width: int = 1920, screen_height: int = 1080) -> bool:
        """
        Move mouse cursor by a relative pixel amount.

        Converts pixel delta to HID units and sends ``move:DX,DY`` to the
        firmware, which adds the delta to its own tracked absolute position.
        This is a true relative move independent of the Python-side tracker.

        Args:
            x: Horizontal pixel delta (positive = right, negative = left)
            y: Vertical pixel delta (positive = down, negative = up)

        Returns:
            True if successful, False otherwise
        """
        hid_dx = int(x * 32767 / screen_width)
        hid_dy = int(y * 32767 / screen_height)
        logger.debug(f"Relative mouse move: pixel ({x},{y}) -> HID delta ({hid_dx},{hid_dy})")
        ok = await self.send_command(f"move:{hid_dx},{hid_dy}")
        await asyncio.sleep(0.05)
        return ok

    async def move_mouse_absolute(self, x: int, y: int,
                                   screen_width: int = 1920,
                                   screen_height: int = 1080) -> bool:
        """
        Move mouse to an absolute pixel position using USBHIDAbsoluteMouse.

        Converts pixel coordinates to the 0-32767 range used by the HID
        absolute mouse descriptor and sends a single ``moveto:X,Y`` command.
        Also updates the tracked position so subsequent move_mouse() calls
        use the correct origin.

        Args:
            x: Target pixel X coordinate (0 = left edge)
            y: Target pixel Y coordinate (0 = top edge)
            screen_width: Captured screen width in pixels (default 1920)
            screen_height: Captured screen height in pixels (default 1080)

        Returns:
            True if successful, False otherwise
        """
        self._current_x = max(0, min(screen_width, x))
        self._current_y = max(0, min(screen_height, y))
        hid_x = int(self._current_x * 32767 / screen_width)
        hid_y = int(self._current_y * 32767 / screen_height)
        logger.debug(f"Absolute mouse move: pixel ({self._current_x},{self._current_y}) -> HID ({hid_x},{hid_y})")
        ok = await self.send_command(f"moveto:{hid_x},{hid_y}")
        await asyncio.sleep(0.1)
        return ok

    async def move_mouse_to_position(self, x: int, y: int,
                                      screen_width: int = 1920,
                                      screen_height: int = 1080) -> bool:
        """
        Move mouse to absolute pixel position using USBHIDAbsoluteMouse.

        Args:
            x: Target pixel X from left edge
            y: Target pixel Y from top edge
            screen_width: Captured screen width in pixels
            screen_height: Captured screen height in pixels

        Returns:
            True if successful, False otherwise
        """
        return await self.move_mouse_absolute(x, y, screen_width, screen_height)

    async def scroll(self, amount: int) -> bool:
        """
        Scroll mouse wheel.

        Args:
            amount: Scroll amount (positive = down, negative = up)

        Returns:
            True if successful, False otherwise
        """
        return await self.send_command(f"scroll:{amount}")

    def is_connected(self) -> bool:
        """
        Check if connected to BLE device.

        Returns:
            True if connected, False otherwise
        """
        return self.client is not None and self.client.is_connected

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
