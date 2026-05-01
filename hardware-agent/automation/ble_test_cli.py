"""
BLE Test CLI - Interactive testing tool for ESP32 BLE keyboard and mouse.

Provides a REPL-style interface for manually sending BLE commands to the ESP32
wireless input bridge for testing keyboard and mouse control.

Supports two connection modes:
  --direct  : Connect directly to ESP32 via BLE (default)
  --socket  : Connect via ble_server Unix socket (when ble_server is running)
"""

import cmd
import sys
import asyncio
import argparse
import json
import socket
from pathlib import Path
from typing import Optional

from automation.ble_controller import BLEController
from automation.ble_client import BLEClient as SocketBLEClient
from automation.config import AutomationConfig


class ColoredOutput:
    """Helper class for colored terminal output."""

    # ANSI color codes
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'

    @staticmethod
    def success(message: str) -> str:
        """Format success message in green."""
        return f"{ColoredOutput.GREEN}[SUCCESS]{ColoredOutput.RESET} {message}"

    @staticmethod
    def error(message: str) -> str:
        """Format error message in red."""
        return f"{ColoredOutput.RED}[ERROR]{ColoredOutput.RESET} {message}"

    @staticmethod
    def info(message: str) -> str:
        """Format info message in blue."""
        return f"{ColoredOutput.BLUE}[INFO]{ColoredOutput.RESET} {message}"

    @staticmethod
    def warning(message: str) -> str:
        """Format warning message in yellow."""
        return f"{ColoredOutput.YELLOW}[WARNING]{ColoredOutput.RESET} {message}"

    @staticmethod
    def hint(message: str) -> str:
        """Format hint message in cyan."""
        return f"{ColoredOutput.CYAN}[HINT]{ColoredOutput.RESET} {message}"

    @staticmethod
    def header(text: str, width: int = 60) -> str:
        """Format header with border."""
        border = "━" * width
        return f"\n{border}\n{text}\n{border}\n"


class SocketBLERunner:
    """
    Bridges synchronous CLI commands with ble_server Unix socket.

    Uses BLEClient to communicate with a running ble_server process.
    """

    def __init__(self):
        self.client = SocketBLEClient()

    def connect(self, timeout: float = 10.0) -> bool:
        """Check if ble_server is running and connected."""
        del timeout
        return self.client.is_server_running()

    def disconnect(self) -> None:
        """No-op for socket mode."""
        pass

    def is_connected(self) -> bool:
        """Check if ble_server is running."""
        return self.client.is_server_running()

    def scan_devices(self, timeout: float = 10.0) -> Optional[str]:
        """Not available in socket mode."""
        del timeout
        return None

    def send_command(self, command: str) -> bool:
        """Send raw command via ble_server."""
        return self.client.send_command(command)

    def switch_to_keyboard_mode(self) -> bool:
        return self.client.send_command("mode:keyboard")

    def switch_to_mouse_mode(self) -> bool:
        return self.client.send_command("mode:mouse")

    def type_text(self, text: str) -> bool:
        return self.client.type_text(text)

    def press_key(self, key: str) -> bool:
        return self.client.press_key(key)

    def click(self) -> bool:
        return self.client.click()

    def double_click(self) -> bool:
        return self.client.double_click()

    def right_click(self) -> bool:
        return self.client.right_click()

    def move_mouse(self, x: int, y: int) -> bool:
        return self.client.move_mouse(x, y)

    def move_mouse_absolute(self, x: int, y: int,
                             screen_width: int = 1920, screen_height: int = 1080) -> bool:
        return self.client.move_mouse_absolute(x, y)

    def move_mouse_to_position(self, x: int, y: int,
                                screen_width: int = 1920, screen_height: int = 1080) -> bool:
        return self.client.move_mouse_to_position(x, y, screen_width, screen_height)

    def scroll(self, amount: int) -> bool:
        return self.client.scroll(amount)

    def alt_tab(self) -> bool:
        """Send Alt+Tab shortcut."""
        return self.client.alt_tab()

    def start_logs(self, callback) -> bool:
        """Not available in socket mode."""
        print(ColoredOutput.warning("Log streaming is not available in socket mode. Use direct BLE connection."))
        return False

    def stop_logs(self) -> bool:
        """No-op for socket mode."""
        return True

    def get_device_address(self) -> Optional[str]:
        return "ble_server"

    def get_current_mode(self) -> str:
        return "unknown"

    def cleanup(self):
        pass


class AsyncBLERunner:
    """
    Bridges synchronous CLI commands with async BLE operations.

    Manages the asyncio event loop and BLEController lifecycle.
    """

    def __init__(self, config: AutomationConfig):
        """
        Initialize async BLE runner.

        Args:
            config: Automation configuration instance
        """
        self.config = config
        self.ble = BLEController(
            device_name=config.esp32_device_name,
            service_uuid=config.ble_service_uuid,
            rx_char_uuid=config.ble_rx_char_uuid,
            tx_char_uuid=config.ble_tx_char_uuid
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run_async(self, coro):
        """
        Run async coroutine in synchronous context.

        Args:
            coro: Async coroutine to execute

        Returns:
            Result from coroutine
        """
        return self.loop.run_until_complete(coro)

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to ESP32 BLE device."""
        return self.run_async(self.ble.connect(timeout))

    def disconnect(self) -> None:
        """Disconnect from ESP32 BLE device."""
        self.run_async(self.ble.disconnect())

    def is_connected(self) -> bool:
        """Check if connected to BLE device."""
        return self.ble.is_connected()

    def scan_devices(self, timeout: float = 10.0) -> Optional[str]:
        """Scan for BLE devices."""
        return self.run_async(self.ble.scan_and_find_device(timeout))

    def send_command(self, command: str) -> bool:
        """Send raw BLE UART command."""
        return self.run_async(self.ble.send_command(command))

    def switch_to_keyboard_mode(self) -> bool:
        """Switch to keyboard mode."""
        return self.run_async(self.ble.switch_to_keyboard_mode())

    def switch_to_mouse_mode(self) -> bool:
        """Switch to mouse mode."""
        return self.run_async(self.ble.switch_to_mouse_mode())

    def type_text(self, text: str) -> bool:
        """Type text using keyboard."""
        return self.run_async(self.ble.type_text(text))

    def press_key(self, key: str) -> bool:
        """Press a special key."""
        return self.run_async(self.ble.press_key(key))

    def click(self) -> bool:
        """Perform left mouse click."""
        return self.run_async(self.ble.click())

    def double_click(self) -> bool:
        """Perform double left mouse click."""
        return self.run_async(self.ble.double_click())

    def right_click(self) -> bool:
        """Perform right mouse click."""
        return self.run_async(self.ble.right_click())

    def move_mouse(self, x: int, y: int) -> bool:
        """Move mouse cursor relatively."""
        return self.run_async(self.ble.move_mouse(x, y))

    def move_mouse_absolute(self, x: int, y: int,
                             screen_width: int = 1920, screen_height: int = 1080) -> bool:
        """Move mouse to absolute pixel position via USBHIDAbsoluteMouse."""
        return self.run_async(self.ble.move_mouse_absolute(x, y, screen_width, screen_height))

    def move_mouse_to_position(self, x: int, y: int,
                                screen_width: int = 1920, screen_height: int = 1080) -> bool:
        """Move mouse to absolute pixel position."""
        return self.run_async(self.ble.move_mouse_to_position(x, y, screen_width, screen_height))

    def scroll(self, amount: int) -> bool:
        """Scroll mouse wheel."""
        return self.run_async(self.ble.scroll(amount))

    def alt_tab(self) -> bool:
        """Send Alt+Tab shortcut."""
        return self.run_async(self.ble.alt_tab())

    def start_logs(self, callback) -> bool:
        """Subscribe to BLE TX notifications for firmware logs."""
        return self.run_async(self.ble.start_logs(callback))

    def stop_logs(self) -> bool:
        """Unsubscribe from BLE TX notifications."""
        return self.run_async(self.ble.stop_logs())

    def get_device_address(self) -> Optional[str]:
        """Get connected device address."""
        return self.ble.device_address

    def get_current_mode(self) -> str:
        """Get current input mode."""
        return self.ble.current_mode

    def cleanup(self):
        """Clean up resources."""
        if self.is_connected():
            self.disconnect()
        self.loop.close()


class BLETestShell(cmd.Cmd):
    """Interactive shell for BLE keyboard and mouse testing."""

    intro = ColoredOutput.header("BLE Keyboard & Mouse Test CLI") + \
            "Type 'help' for commands, 'quit' to exit\n"
    prompt = f"{ColoredOutput.CYAN}(BLE Test){ColoredOutput.RESET} "

    def __init__(self, config: AutomationConfig):
        """
        Initialize BLE test shell.

        Args:
            config: Automation configuration instance
        """
        super().__init__()
        self.config = config
        self.runner = AsyncBLERunner(config)

    def default(self, line):
        """
        Handle firmware-protocol syntax typed directly (e.g. key:enter, type:hello).

        cmd.Cmd splits on whitespace, so "key:enter" arrives here as an unknown
        command.  We detect the colon pattern and route it to the correct handler
        so users can type firmware commands directly without the `raw` prefix.
        """
        line = line.strip()
        if ':' in line:
            cmd_part, _, arg_part = line.partition(':')
            cmd_part = cmd_part.lower()
            if cmd_part == 'key':
                self.do_press(arg_part)
                return
            if cmd_part == 'type':
                self.do_type(arg_part)
                return
            if cmd_part == 'moveto':
                # firmware uses "moveto:X,Y"; CLI do_moveto expects "X Y"
                self.do_moveto(arg_part.replace(',', ' '))
                return
            if cmd_part == 'scroll':
                self.do_scroll(arg_part)
                return
            if cmd_part == 'mode':
                if arg_part == 'keyboard':
                    self.do_keyboard('')
                elif arg_part == 'mouse':
                    self.do_mouse('')
                return
            if cmd_part == 'move':
                # move:DX,DY — pass as "DX DY" to do_move
                self.do_move(arg_part.replace(',', ' '))
                return
            if cmd_part == 'up':
                self.do_move(f"0 -{arg_part}")
                return
            if cmd_part == 'down':
                self.do_move(f"0 {arg_part}")
                return
            if cmd_part == 'left':
                self.do_move(f"-{arg_part} 0")
                return
            if cmd_part == 'right':
                self.do_move(f"{arg_part} 0")
                return
        # Fall back to default error message
        print(f"*** Unknown syntax: {line}")
        print("Type 'help' for available commands, or 'raw <cmd>' to send a raw BLE command.")

    # ========== Connection Management ==========

    def do_connect(self, arg):
        """Connect to ESP32 BLE device."""
        print(ColoredOutput.info(f"Scanning for: {self.config.esp32_device_name}"))

        try:
            if self.runner.connect():
                addr = self.runner.get_device_address()
                print(ColoredOutput.success(f"Connected to ESP32 at {addr}"))
            else:
                print(ColoredOutput.error("Failed to connect"))
                detail = self.runner.ble.get_last_error()
                if detail:
                    print(ColoredOutput.error(f"Detail: {detail}"))
                    if "not authorized" in detail.lower():
                        print(ColoredOutput.hint("Allow Bluetooth for this terminal app in macOS Settings > Privacy & Security > Bluetooth"))
                print(ColoredOutput.hint("Make sure:\n" +
                    "  1. ESP32 is powered on\n" +
                    "  2. Bluetooth is enabled on this computer\n" +
                    f"  3. Device name matches: {self.config.esp32_device_name}\n" +
                    "  4. ESP32 is not already connected to another device"))
        except Exception as e:
            print(ColoredOutput.error(f"Connection error: {e}"))

    def help_connect(self):
        """Help for connect command."""
        print("\nConnect to ESP32 BLE device")
        print("Usage: connect")
        print(f"Scans for device: {self.config.esp32_device_name}")

    def do_disconnect(self, arg):
        """Disconnect from ESP32 BLE device."""
        if not self.runner.is_connected():
            print(ColoredOutput.warning("Not connected"))
            return

        try:
            self.runner.disconnect()
            print(ColoredOutput.success("Disconnected from ESP32"))
        except Exception as e:
            print(ColoredOutput.error(f"Disconnect error: {e}"))

    def help_disconnect(self):
        """Help for disconnect command."""
        print("\nDisconnect from ESP32 BLE device")
        print("Usage: disconnect")

    def do_status(self, arg):
        """Show connection status and current mode."""
        border = "━" * 60
        print(f"\n{border}")
        print(f"Connected:      {'Yes' if self.runner.is_connected() else 'No'}")

        if self.runner.is_connected():
            print(f"Device:         {self.config.esp32_device_name}")
            print(f"Address:        {self.runner.get_device_address()}")
            print(f"Current Mode:   {self.runner.get_current_mode().title()}")

        print(f"{border}\n")

    def help_status(self):
        """Help for status command."""
        print("\nShow connection status and current mode")
        print("Usage: status")

    def do_scan(self, arg):
        """Scan for available BLE devices."""
        print(ColoredOutput.info("Scanning for BLE devices..."))

        try:
            addr = self.runner.scan_devices()
            if addr:
                print(ColoredOutput.success(f"Found device at: {addr}"))
            else:
                print(ColoredOutput.warning("Device not found"))
                detail = self.runner.ble.get_last_error()
                if detail:
                    print(ColoredOutput.error(f"Detail: {detail}"))
                    if "not authorized" in detail.lower():
                        print(ColoredOutput.hint("Allow Bluetooth for this terminal app in macOS Settings > Privacy & Security > Bluetooth"))
        except Exception as e:
            print(ColoredOutput.error(f"Scan error: {e}"))

    def help_scan(self):
        """Help for scan command."""
        print("\nScan for available BLE devices")
        print("Usage: scan")
        print(f"Searches for: {self.config.esp32_device_name}")

    # ========== Mode Switching ==========

    def do_keyboard(self, arg):
        """Switch to keyboard mode."""
        if not self._check_connection():
            return

        try:
            if self.runner.switch_to_keyboard_mode():
                print(ColoredOutput.success("Switched to keyboard mode"))
            else:
                print(ColoredOutput.error("Failed to switch mode"))
        except Exception as e:
            print(ColoredOutput.error(f"Mode switch error: {e}"))

    def help_keyboard(self):
        """Help for keyboard command."""
        print("\nSwitch ESP32 to keyboard mode")
        print("Usage: keyboard")
        print("Must be in keyboard mode to use: type, press")

    def do_mouse(self, arg):
        """Switch to mouse mode."""
        if not self._check_connection():
            return

        try:
            if self.runner.switch_to_mouse_mode():
                print(ColoredOutput.success("Switched to mouse mode"))
            else:
                print(ColoredOutput.error("Failed to switch mode"))
        except Exception as e:
            print(ColoredOutput.error(f"Mode switch error: {e}"))

    def help_mouse(self):
        """Help for mouse command."""
        print("\nSwitch ESP32 to mouse mode")
        print("Usage: mouse")
        print("Must be in mouse mode to use: move, moveto, click, scroll, reset")

    # ========== Keyboard Commands ==========

    def do_type(self, arg):
        """Type text using keyboard."""
        if not self._check_connection():
            return

        if not arg:
            print(ColoredOutput.error("No text provided"))
            print(ColoredOutput.hint('Usage: type "your text here"'))
            return

        # Remove quotes if present
        text = arg.strip('"').strip("'")

        try:
            if self.runner.type_text(text):
                print(ColoredOutput.success(f"Typed: {text}"))
            else:
                print(ColoredOutput.error("Failed to type text"))
        except Exception as e:
            print(ColoredOutput.error(f"Type error: {e}"))

    def help_type(self):
        """Help for type command."""
        print("\nType text using keyboard")
        print('Usage: type "text to type"')
        print("Example: type \"Hello World\"")
        print("\nNote: Must be in keyboard mode first (use 'keyboard' command)")

    def do_press(self, arg):
        """Press a special key."""
        if not self._check_connection():
            return

        valid_keys = ['enter', 'return', 'tab', 'backspace', 'delete', 'esc', 'escape']

        if not arg:
            print(ColoredOutput.error("No key specified"))
            print(ColoredOutput.hint(f"Valid keys: {', '.join(valid_keys)}"))
            return

        key = arg.strip().lower()

        # Allow 'return' as alias for 'enter'
        if key == 'return':
            key = 'enter'
        elif key == 'escape':
            key = 'esc'

        if key not in valid_keys:
            print(ColoredOutput.error(f"Unknown key: {key}"))
            print(ColoredOutput.hint(f"Valid keys: {', '.join(valid_keys)}"))
            return

        try:
            if self.runner.press_key(key):
                print(ColoredOutput.success(f"Pressed: {key.title()}"))
            else:
                print(ColoredOutput.error("Failed to press key"))
        except Exception as e:
            print(ColoredOutput.error(f"Press error: {e}"))

    def help_press(self):
        """Help for press command."""
        print("\nPress a special key")
        print("Usage: press <key>")
        print("Valid keys: enter, tab, backspace, delete, esc, escape")
        print("Example: press enter")
        print("\nNote: Must be in keyboard mode first (use 'keyboard' command)")

    def do_esc(self, arg):
        """Press Escape once."""
        del arg
        self.do_press("esc")

    def help_esc(self):
        """Help for esc command."""
        print("\nPress Escape once")
        print("Usage: esc")
        print("\nEquivalent to: press esc")

    def do_alt_tab(self, arg):
        """Send Alt+Tab shortcut to switch windows."""
        del arg
        if not self._check_connection():
            return
        try:
            if hasattr(self.runner, 'alt_tab') and self.runner.alt_tab():
                print(ColoredOutput.success("Sent: Alt+Tab"))
            else:
                print(ColoredOutput.error("Failed to send Alt+Tab"))
        except Exception as e:
            print(ColoredOutput.error(f"Alt+Tab error: {e}"))

    def help_alt_tab(self):
        """Help for alt_tab command."""
        print("\nSend Alt+Tab shortcut to switch windows")
        print("Usage: alt_tab")

    def do_logs(self, arg):
        """Stream firmware logs via BLE TX notifications."""
        del arg
        if not self._check_connection():
            return
        try:
            if hasattr(self.runner, 'start_logs') and self.runner.start_logs(lambda msg: print(f"  [BLE] {msg}")):
                print(ColoredOutput.info("Listening for logs... Press Enter to stop."))
                input()
                self.runner.stop_logs()
                print(ColoredOutput.success("Stopped log streaming"))
            else:
                print(ColoredOutput.error("Failed to start log streaming"))
        except Exception as e:
            print(ColoredOutput.error(f"Log streaming error: {e}"))

    def help_logs(self):
        """Help for logs command."""
        print("\nStream firmware logs via BLE TX notifications")
        print("Usage: logs")
        print("Requires direct BLE connection (not socket mode).")
        print("Shows WiFi connection status, OTA readiness, and other boot logs.")

    def do_scroll_up(self, arg):
        """Scroll up (default 3 units)."""
        amount = 3
        if arg.strip():
            try:
                amount = int(arg.strip())
            except ValueError:
                print(ColoredOutput.error("Invalid scroll amount"))
                return
        self.do_scroll(str(-abs(amount)))

    def help_scroll_up(self):
        """Help for scroll_up command."""
        print("\nScroll up")
        print("Usage: scroll_up [amount]")
        print("Default amount: 3")

    def do_scroll_down(self, arg):
        """Scroll down (default 3 units)."""
        amount = 3
        if arg.strip():
            try:
                amount = int(arg.strip())
            except ValueError:
                print(ColoredOutput.error("Invalid scroll amount"))
                return
        self.do_scroll(str(abs(amount)))

    def help_scroll_down(self):
        """Help for scroll_down command."""
        print("\nScroll down")
        print("Usage: scroll_down [amount]")
        print("Default amount: 3")

    def do_win(self, arg):
        """Press Windows key."""
        del arg
        if not self._check_connection():
            return
        try:
            if self.runner.press_key("win"):
                print(ColoredOutput.success("Pressed: Win"))
            else:
                print(ColoredOutput.error("Failed to press Win"))
        except Exception as e:
            print(ColoredOutput.error(f"Win key error: {e}"))

    def help_win(self):
        """Help for win command."""
        print("\nPress Windows key")
        print("Usage: win")

    def do_win_up(self, arg):
        """Press Win+Up Arrow shortcut (maximize window)."""
        del arg
        if not self._check_connection():
            return
        try:
            if self.runner.press_key("win_up"):
                print(ColoredOutput.success("Sent: Win+Up"))
            else:
                print(ColoredOutput.error("Failed to send Win+Up"))
        except Exception as e:
            print(ColoredOutput.error(f"Win+Up error: {e}"))

    def help_win_up(self):
        """Help for win_up command."""
        print("\nPress Win+Up Arrow shortcut (maximize window)")
        print("Usage: win_up")

    # ========== Mouse Commands ==========

    def do_move(self, arg):
        """Move mouse cursor relatively."""
        if not self._check_connection():
            return

        try:
            parts = arg.split()
            if len(parts) != 2:
                print(ColoredOutput.error("Invalid arguments"))
                print(ColoredOutput.hint("Usage: move <x> <y>"))
                return

            x = int(parts[0])
            y = int(parts[1])

            if self.runner.move_mouse(x, y):
                direction = []
                if x > 0:
                    direction.append(f"right {x}")
                elif x < 0:
                    direction.append(f"left {abs(x)}")
                if y > 0:
                    direction.append(f"down {y}")
                elif y < 0:
                    direction.append(f"up {abs(y)}")

                if direction:
                    print(ColoredOutput.success(f"Moved: {', '.join(direction)}"))
                else:
                    print(ColoredOutput.success("No movement (0, 0)"))
            else:
                print(ColoredOutput.error("Failed to move mouse"))
        except ValueError:
            print(ColoredOutput.error("Invalid coordinates"))
            print(ColoredOutput.hint("Usage: move <x> <y> (integers)"))
        except Exception as e:
            print(ColoredOutput.error(f"Move error: {e}"))

    def help_move(self):
        """Help for move command."""
        print("\nMove mouse cursor relatively")
        print("Usage: move <x> <y>")
        print("  x: Horizontal movement (positive=right, negative=left)")
        print("  y: Vertical movement (positive=down, negative=up)")
        print("Example: move 100 50  (move right 100, down 50)")
        print("\nNote: Must be in mouse mode first (use 'mouse' command)")

    def do_moveto(self, arg):
        """Move mouse to absolute position."""
        if not self._check_connection():
            return

        try:
            parts = arg.split()
            if len(parts) != 2:
                print(ColoredOutput.error("Invalid arguments"))
                print(ColoredOutput.hint("Usage: moveto <x> <y>"))
                return

            x = int(parts[0])
            y = int(parts[1])

            if x < 0 or y < 0:
                print(ColoredOutput.error("Coordinates must be non-negative"))
                return

            print(ColoredOutput.info(f"Moving to absolute position ({x}, {y}) [pixel coords on 1920x1080]..."))

            if self.runner.move_mouse_absolute(x, y):
                print(ColoredOutput.success(f"Moved to position: ({x}, {y})"))
            else:
                print(ColoredOutput.error("Failed to move mouse"))
        except ValueError:
            print(ColoredOutput.error("Invalid coordinates"))
            print(ColoredOutput.hint("Usage: moveto <x> <y> (non-negative integers)"))
        except Exception as e:
            print(ColoredOutput.error(f"Move error: {e}"))

    def help_moveto(self):
        """Help for moveto command."""
        print("\nMove mouse to absolute pixel position (uses USBHIDAbsoluteMouse)")
        print("Usage: moveto <x> <y>")
        print("  x: Pixel X from left edge (0-1919 for 1920px wide screen)")
        print("  y: Pixel Y from top edge  (0-1079 for 1080px tall screen)")
        print("Example: moveto 960 540  (center of 1920x1080 screen)")
        print("\nNote: Requires firmware with USBHIDAbsoluteMouse support")
        print("\nNote: Resets cursor to origin first, then moves to target")
        print("Note: Must be in mouse mode first (use 'mouse' command)")

    def do_click(self, arg):
        """Perform left mouse click."""
        if not self._check_connection():
            return

        try:
            if self.runner.click():
                print(ColoredOutput.success("Clicked (left)"))
            else:
                print(ColoredOutput.error("Failed to click"))
        except Exception as e:
            print(ColoredOutput.error(f"Click error: {e}"))

    def help_click(self):
        """Help for click command."""
        print("\nPerform left mouse click")
        print("Usage: click")
        print("\nNote: Must be in mouse mode first (use 'mouse' command)")

    def do_rclick(self, arg):
        """Perform right mouse click."""
        if not self._check_connection():
            return

        try:
            if self.runner.right_click():
                print(ColoredOutput.success("Clicked (right)"))
            else:
                print(ColoredOutput.error("Failed to right-click"))
        except Exception as e:
            print(ColoredOutput.error(f"Right-click error: {e}"))

    def help_rclick(self):
        """Help for rclick command."""
        print("\nPerform right mouse click")
        print("Usage: rclick")
        print("\nNote: Must be in mouse mode first (use 'mouse' command)")

    def do_scroll(self, arg):
        """Scroll mouse wheel."""
        if not self._check_connection():
            return

        if not arg:
            print(ColoredOutput.error("No scroll amount specified"))
            print(ColoredOutput.hint("Usage: scroll <amount>"))
            return

        try:
            amount = int(arg.strip())

            if self.runner.scroll(amount):
                direction = "down" if amount > 0 else "up"
                print(ColoredOutput.success(f"Scrolled {direction}: {abs(amount)}"))
            else:
                print(ColoredOutput.error("Failed to scroll"))
        except ValueError:
            print(ColoredOutput.error("Invalid scroll amount"))
            print(ColoredOutput.hint("Usage: scroll <amount> (integer)"))
        except Exception as e:
            print(ColoredOutput.error(f"Scroll error: {e}"))

    def help_scroll(self):
        """Help for scroll command."""
        print("\nScroll mouse wheel")
        print("Usage: scroll <amount>")
        print("  Positive values scroll down")
        print("  Negative values scroll up")
        print("Example: scroll 3  (scroll down 3 units)")
        print("Example: scroll -2 (scroll up 2 units)")
        print("\nNote: Must be in mouse mode first (use 'mouse' command)")

    def do_reset(self, arg):
        """Reset tracked position to (0,0) and move cursor to top-left."""
        if not self._check_connection():
            return

        try:
            print(ColoredOutput.info("Moving cursor to top-left (0, 0)..."))

            if self.runner.move_mouse_absolute(0, 0):
                print(ColoredOutput.success("Moved to (0, 0)"))
            else:
                print(ColoredOutput.error("Failed to move mouse"))
        except Exception as e:
            print(ColoredOutput.error(f"Reset error: {e}"))

    def help_reset(self):
        """Help for reset command."""
        print("\nMove cursor to top-left corner (0, 0)")
        print("Usage: reset")
        print("\nNote: Must be in mouse mode first (use 'mouse' command)")

    # ========== Utilities ==========

    def do_raw(self, arg):
        """Send raw BLE UART command."""
        if not self._check_connection():
            return

        if not arg:
            print(ColoredOutput.error("No command specified"))
            print(ColoredOutput.hint("Usage: raw <command>"))
            return

        command = arg.strip()

        try:
            if self.runner.send_command(command):
                print(ColoredOutput.success(f"Sent: {command}"))
            else:
                print(ColoredOutput.error("Failed to send command"))
        except Exception as e:
            print(ColoredOutput.error(f"Send error: {e}"))

    def help_raw(self):
        """Help for raw command."""
        print("\nSend raw BLE UART command to ESP32")
        print("Usage: raw <command>")
        print("Examples:")
        print("  raw mode:keyboard")
        print("  raw type:Hello")
        print("  raw key:enter")
        print("  raw left:100")
        print("  raw click")

    def do_quit(self, arg):
        """Exit the CLI."""
        print("\nGoodbye!")
        return True

    def do_exit(self, arg):
        """Exit the CLI (alias for quit)."""
        return self.do_quit(arg)

    def do_EOF(self, arg):
        """Exit on Ctrl-D."""
        print()  # New line after ^D
        return self.do_quit(arg)

    def help_quit(self):
        """Help for quit command."""
        print("\nExit the BLE Test CLI")
        print("Usage: quit (or exit, or Ctrl-D)")

    # ========== Helper Methods ==========

    def _check_connection(self) -> bool:
        """
        Check if connected to BLE device.

        Returns:
            True if connected, False otherwise
        """
        if not self.runner.is_connected():
            print(ColoredOutput.error("Not connected to BLE device"))
            print(ColoredOutput.hint("Run 'connect' first"))
            return False
        return True

    def postcmd(self, stop, line):
        """Hook method executed after each command."""
        return stop

    def emptyline(self):
        """Do nothing on empty line (override default repeat behavior)."""
        pass

    def cleanup(self):
        """Clean up resources before exit."""
        if self.runner.is_connected():
            print(ColoredOutput.info("Disconnecting..."))
            self.runner.disconnect()
        self.runner.cleanup()


def main():
    """Main entry point for BLE test CLI."""
    parser = argparse.ArgumentParser(
        description="BLE Keyboard & Mouse Test CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--device-name',
        help='Override ESP32 device name'
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=10.0,
        help='Connection timeout in seconds (default: 10)'
    )

    parser.add_argument(
        '--socket',
        action='store_true',
        help='Connect via ble_server Unix socket instead of direct BLE'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = AutomationConfig()

        # Override device name if specified
        if args.device_name:
            config.esp32_device_name = args.device_name

    except Exception as e:
        print(ColoredOutput.error(f"Configuration error: {e}"))
        return 1

    # Create and run shell
    shell = BLETestShell(config)

    # Switch to socket mode if requested
    if args.socket:
        shell.runner = SocketBLERunner()
        shell.prompt = f"{ColoredOutput.CYAN}(BLE Test/socket){ColoredOutput.RESET} "

    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(ColoredOutput.error(f"Unexpected error: {e}"))
        return 1
    finally:
        shell.cleanup()

    return 0


if __name__ == '__main__':
    sys.exit(main())
