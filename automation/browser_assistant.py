"""
Interactive Browser Assistant

Terminal-based chat assistant that controls a remote Windows PC's Chrome browser
via HDMI capture and ESP32 BLE keyboard/mouse emulation.

Features:
- Natural language command interpretation
- Dual YOLO model support (DocLayout-YOLO and YOLOv8 UI detection)
- Browser automation (open Chrome, navigate to URLs)
- Interactive chat interface with real-time feedback
"""

import argparse
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import cv2
import numpy as np

from automation.config import AutomationConfig, load_config
from automation.model_manager import ModelManager, ModelType, DetectionResult
from automation.utils import setup_logging, save_debug_image

logger = logging.getLogger("browser_assistant")


# =============================================================================
# Command Parser
# =============================================================================

class CommandType(Enum):
    """Supported command types"""
    OPEN_BROWSER = "open_browser"
    NAVIGATE = "navigate"
    SWITCH_MODEL = "switch_model"
    CLICK_ELEMENT = "click_element"
    ANALYZE = "analyze"
    CAPTURE = "capture"
    HELP = "help"
    QUIT = "quit"
    UNKNOWN = "unknown"


@dataclass
class Command:
    """Parsed command with type and parameters"""
    type: CommandType
    params: dict


class CommandParser:
    """
    Natural language command parser.

    Parses user input text into structured Command objects that can be
    executed by the browser assistant.
    """

    def parse(self, user_input: str) -> Command:
        """
        Parse user input into a Command object.

        Args:
            user_input: Raw user input text

        Returns:
            Command object with type and parameters
        """
        user_input = user_input.lower().strip()

        # Open browser
        if "open" in user_input and "chrome" in user_input:
            return Command(CommandType.OPEN_BROWSER, {"browser": "chrome"})

        # Navigate to URL
        if any(word in user_input for word in ["goto", "go to", "navigate", "open url"]):
            url = self.extract_url(user_input)
            if url:
                return Command(CommandType.NAVIGATE, {"url": url})

        # Switch model
        if "switch" in user_input or "use" in user_input or "change" in user_input:
            if "doclayout" in user_input or "document" in user_input:
                return Command(CommandType.SWITCH_MODEL, {"model": "doclayout"})
            elif "ui" in user_input or "detection" in user_input or "yolo11" in user_input:
                return Command(CommandType.SWITCH_MODEL, {"model": "ui-detection"})

        # Click element
        if "click" in user_input and "address" in user_input:
            return Command(CommandType.CLICK_ELEMENT, {"element": "address_bar"})

        # Analyze screen
        if "analyze" in user_input or "detect" in user_input or "scan" in user_input:
            return Command(CommandType.ANALYZE, {})

        # Capture screenshot
        if "capture" in user_input or "screenshot" in user_input:
            return Command(CommandType.CAPTURE, {})

        # Help
        if "help" in user_input or user_input == "?":
            return Command(CommandType.HELP, {})

        # Quit
        if user_input in ["quit", "exit", "bye", "q"]:
            return Command(CommandType.QUIT, {})

        # Unknown command
        return Command(CommandType.UNKNOWN, {"raw": user_input})

    def extract_url(self, text: str) -> Optional[str]:
        """
        Extract URL from text.

        Supports both full URLs (https://example.com) and bare domains (example.com).

        Args:
            text: Text containing URL

        Returns:
            Extracted URL with protocol, or None if not found
        """
        # Try to find full URL with protocol
        url_pattern = r'https?://[^\s]+'
        match = re.search(url_pattern, text)
        if match:
            return match.group(0)

        # Try to find domain-like pattern
        domain_pattern = r'\b([a-z0-9-]+\.)+[a-z]{2,}\b'
        match = re.search(domain_pattern, text)
        if match:
            domain = match.group(0)
            # Add protocol if missing
            if not domain.startswith('http'):
                domain = 'https://' + domain
            return domain

        return None


# =============================================================================
# Browser Controller
# =============================================================================

class BrowserController:
    """
    High-level browser control using BLE commands.

    Provides methods for opening Chrome, detecting UI elements, and
    navigating to URLs using the UI detection model.
    """

    def __init__(self, ble_controller, model_manager: ModelManager, capture):
        """
        Initialize Browser Controller.

        Args:
            ble_controller: BLEController instance for sending commands
            model_manager: ModelManager instance for detection
            capture: OpenCV VideoCapture instance
        """
        self.ble = ble_controller
        self.model_manager = model_manager
        self.capture = capture

    async def open_chrome(self) -> bool:
        """
        Open Chrome browser using Windows search.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Opening Chrome browser...")

        try:
            # Press Windows key to open start menu
            await self.ble.send_command("key:windows")
            await asyncio.sleep(1.0)

            # Type "chrome" to search
            await self.ble.type_text("chrome")
            await asyncio.sleep(0.5)

            # Press Enter to launch
            await self.ble.press_key("enter")
            await asyncio.sleep(3.0)  # Wait for Chrome to open

            logger.info("Chrome opened successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to open Chrome: {e}")
            return False

    async def detect_address_bar(self) -> Optional[Tuple[int, int]]:
        """
        Detect address bar location using UI detection model.

        Returns:
            Center coordinates (x, y) of address bar, or None if not found
        """
        logger.debug("Detecting address bar...")

        # Save current model and switch to UI detection
        original_model = self.model_manager.get_current_model_type()
        self.model_manager.switch_model(ModelType.UI_DETECTION)

        try:
            # Capture screen
            ret, frame = self.capture.read()
            if not ret:
                logger.error("Failed to capture screen")
                return None

            # Detect UI elements
            detections = self.model_manager.detect(frame, confidence=0.3)

            # Look for address bar (text input in top portion of screen)
            address_bar_labels = ["text", "input", "textbox", "search", "url"]

            for det in detections:
                if any(label in det.label.lower() for label in address_bar_labels):
                    # Check if it's in top 30% of screen (address bars are usually high)
                    x1, y1, x2, y2 = det.bbox
                    if y1 < frame.shape[0] * 0.3:
                        # Return center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        logger.info(f"Address bar detected at ({center_x}, {center_y})")
                        return (center_x, center_y)

            logger.warning("Address bar not detected")
            return None

        finally:
            # Restore original model
            self.model_manager.switch_model(original_model)

    async def click_address_bar(self) -> bool:
        """
        Detect and click address bar.

        Returns:
            True if successful, False otherwise
        """
        coords = await self.detect_address_bar()
        if coords is None:
            logger.error("Cannot click address bar - not detected")
            return False

        try:
            x, y = coords

            # Move mouse to address bar
            await self.ble.move_mouse_to_position(x, y)
            await asyncio.sleep(0.2)

            # Click
            await self.ble.click()
            await asyncio.sleep(0.3)

            logger.info("Clicked address bar")
            return True

        except Exception as e:
            logger.error(f"Failed to click address bar: {e}")
            return False

    async def navigate_to_url(self, url: str) -> bool:
        """
        Navigate to URL by typing in address bar.

        Args:
            url: URL to navigate to

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Navigating to {url}...")

        try:
            # Click address bar
            if not await self.click_address_bar():
                logger.error("Failed to click address bar")
                return False

            # Clear existing content (Ctrl+A, Delete)
            await self.ble.send_command("key:ctrl+a")
            await asyncio.sleep(0.1)
            await self.ble.send_command("key:delete")
            await asyncio.sleep(0.2)

            # Type URL
            await self.ble.type_text(url)
            await asyncio.sleep(0.5)

            # Press Enter
            await self.ble.press_key("enter")
            await asyncio.sleep(2.0)  # Wait for page to load

            logger.info(f"Navigated to {url}")
            return True

        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return False


# =============================================================================
# Chat Interface
# =============================================================================

class ChatInterface:
    """
    Interactive terminal UI for conversation.

    Provides methods for displaying welcome messages, help text, prompts,
    and results with colored icons.
    """

    def __init__(self):
        """Initialize Chat Interface."""
        self.commands_history = []

    def show_welcome(self):
        """Display welcome message."""
        print("\n" + "="*60)
        print("  Interactive Browser Assistant".center(60))
        print("="*60)
        print("\nControl remote Chrome browser via chat commands.")
        print("Type 'help' for available commands.\n")

    def show_help(self):
        """Display available commands."""
        help_text = """
Available Commands:

Browser Control:
  • open chrome              - Launch Chrome browser
  • goto <url>              - Navigate to URL (e.g., 'goto google.com')
  • click address bar       - Click the address bar

Model Control:
  • switch to doclayout     - Use DocLayout-YOLO model
  • switch to ui detection  - Use YOLOv11 UI detection model

Screen Analysis:
  • analyze                 - Analyze current screen
  • capture                 - Save screenshot

Other:
  • help                    - Show this help message
  • quit                    - Exit assistant

Examples:
  > open chrome
  > goto https://www.google.com
  > switch to ui detection
  > analyze
"""
        print(help_text)

    def get_user_input(self, model_name: str) -> str:
        """
        Get user input with prompt showing current model.

        Args:
            model_name: Name of currently active model

        Returns:
            User input text
        """
        try:
            prompt = f"\n[{model_name}] > "
            user_input = input(prompt).strip()
            return user_input
        except (EOFError, KeyboardInterrupt):
            return "quit"

    def show_result(self, message: str, success: bool = True):
        """
        Display command result with icon.

        Args:
            message: Result message to display
            success: True for success icon, False for error icon
        """
        icon = "✅" if success else "❌"
        print(f"{icon} {message}")

    def show_detections(self, detections: List[DetectionResult]):
        """
        Display detection results.

        Args:
            detections: List of DetectionResult objects
        """
        if not detections:
            print("📊 No detections found")
            return

        print(f"\n📊 Detected {len(detections)} elements:")
        for i, det in enumerate(detections[:10], 1):  # Show top 10
            print(f"  {i}. {det.label} (confidence: {det.confidence:.2f})")

        if len(detections) > 10:
            print(f"  ... and {len(detections) - 10} more")


# =============================================================================
# Browser Assistant Main Class
# =============================================================================

class BrowserAssistant:
    """
    Main browser assistant orchestrator.

    Coordinates all components (model manager, command parser, browser controller,
    chat interface) and implements the main chat loop.
    """

    def __init__(self, config: AutomationConfig, args):
        """
        Initialize Browser Assistant.

        Args:
            config: AutomationConfig object
            args: Command-line arguments
        """
        self.config = config
        self.args = args

        # Components
        self.capture = None
        self.ble = None
        self.model_manager = ModelManager(config)
        self.command_parser = CommandParser()
        self.chat_interface = ChatInterface()
        self.browser_controller = None

        self.running = False

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Browser Assistant...")

        # Initialize video capture
        self.capture = cv2.VideoCapture(self.args.device)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open device {self.args.device}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        logger.info("Video capture initialized")

        # Connect to ESP32 BLE
        from automation.ble_controller import BLEController
        self.ble = BLEController(
            device_name=self.config.esp32_device_name,
            service_uuid=self.config.ble_service_uuid,
            rx_char_uuid=self.config.ble_rx_char_uuid,
            tx_char_uuid=self.config.ble_tx_char_uuid
        )
        await self.ble.connect()
        logger.info("BLE connected")

        # Initialize browser controller
        self.browser_controller = BrowserController(
            self.ble,
            self.model_manager,
            self.capture
        )

        # Load default model
        self.model_manager.load_doclayout_model()
        logger.info("Default model loaded")

        logger.info("✅ Initialization complete")

    async def execute_command(self, command: Command):
        """
        Execute parsed command.

        Args:
            command: Command object to execute
        """
        if command.type == CommandType.OPEN_BROWSER:
            success = await self.browser_controller.open_chrome()
            self.chat_interface.show_result(
                "Chrome opened" if success else "Failed to open Chrome",
                success
            )

        elif command.type == CommandType.NAVIGATE:
            url = command.params["url"]
            success = await self.browser_controller.navigate_to_url(url)
            self.chat_interface.show_result(
                f"Navigated to {url}" if success else "Navigation failed",
                success
            )

        elif command.type == CommandType.SWITCH_MODEL:
            model_name = command.params["model"]
            if model_name == "doclayout":
                self.model_manager.switch_model(ModelType.DOCLAYOUT)
            else:
                self.model_manager.switch_model(ModelType.UI_DETECTION)
            self.chat_interface.show_result(f"Switched to {model_name} model")

        elif command.type == CommandType.CLICK_ELEMENT:
            success = await self.browser_controller.click_address_bar()
            self.chat_interface.show_result(
                "Clicked address bar" if success else "Failed to click",
                success
            )

        elif command.type == CommandType.ANALYZE:
            ret, frame = self.capture.read()
            if ret:
                detections = self.model_manager.detect(frame)
                self.chat_interface.show_detections(detections)
            else:
                self.chat_interface.show_result("Failed to capture", False)

        elif command.type == CommandType.CAPTURE:
            ret, frame = self.capture.read()
            if ret:
                filename = save_debug_image(
                    frame,
                    "chat_capture",
                    self.args.output_dir,
                    timestamp=True
                )
                self.chat_interface.show_result(f"Saved: {filename}")
            else:
                self.chat_interface.show_result("Failed to capture", False)

        elif command.type == CommandType.HELP:
            self.chat_interface.show_help()

        elif command.type == CommandType.QUIT:
            self.chat_interface.show_result("Goodbye!")
            self.running = False

        else:
            self.chat_interface.show_result(
                "Unknown command. Type 'help' for available commands.",
                False
            )

    async def run(self):
        """Main chat loop."""
        try:
            await self.initialize()

            self.chat_interface.show_welcome()
            self.running = True

            while self.running:
                # Get user input
                model_name = self.model_manager.get_current_model_name()
                user_input = self.chat_interface.get_user_input(model_name)

                if not user_input:
                    continue

                # Parse command
                command = self.command_parser.parse(user_input)

                # Execute command
                await self.execute_command(command)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            self.chat_interface.show_result(f"Error: {e}", False)
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")

        if self.ble:
            try:
                await self.ble.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting BLE: {e}")

        if self.capture:
            self.capture.release()

        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Browser Assistant - Control remote Chrome browser via chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./scripts/run_browser_assistant.sh
  ./scripts/run_browser_assistant.sh --device 1 --debug

Chat Commands:
  open chrome              - Launch Chrome browser
  goto google.com          - Navigate to URL
  switch to ui detection   - Change YOLO model
  analyze                  - Analyze current screen
  capture                  - Save screenshot
  help                     - Show all commands
  quit                     - Exit assistant
        """
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Video capture device index (default: 0)"
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="automation_outputs/chat_screenshots",
        help="Screenshot output directory (default: automation_outputs/chat_screenshots)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level, debug_mode=args.debug)

    # Load config
    config = load_config(args.env_file)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run assistant
    assistant = BrowserAssistant(config, args)
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
