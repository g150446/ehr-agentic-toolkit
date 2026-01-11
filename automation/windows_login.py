#!/usr/bin/env python3
"""
Windows PC Auto-Login Script

Automates Windows PC login using:
- HDMI screen capture (MiraBox)
- DocLayout-YOLO + EasyOCR for screen analysis
- ESP32 BLE UART for keyboard/mouse control

Usage:
    python -m automation.windows_login [--password PASSWORD] [--debug] [--no-verify]
    python -m automation.windows_login --test-ble
    python -m automation.windows_login --test-capture
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Import automation modules
from automation.config import load_config, AutomationConfig
from automation.ble_controller import BLEController
from automation.screen_analyzer import (
    capture_screen,
    load_yolo_model,
    load_ocr_reader,
    analyze_layout,
    extract_text,
    find_password_field,
    visualize_detections
)
from automation.utils import (
    setup_logging,
    debug_pause,
    save_debug_image,
    ProgressLogger
)


logger = logging.getLogger("windows_login")


async def test_ble_connection(config: AutomationConfig) -> bool:
    """
    Test BLE connection to ESP32.

    Args:
        config: Automation configuration

    Returns:
        True if test successful, False otherwise
    """
    logger.info("=== BLE Connection Test ===")

    ble = BLEController(
        device_name=config.esp32_device_name,
        service_uuid=config.ble_service_uuid,
        rx_char_uuid=config.ble_rx_char_uuid,
        tx_char_uuid=config.ble_tx_char_uuid
    )

    try:
        # Connect
        if not await ble.connect(timeout=10.0):
            logger.error("Failed to connect to ESP32")
            return False

        logger.info("✓ Connected to ESP32")

        # Test keyboard mode
        await ble.switch_to_keyboard_mode()
        await asyncio.sleep(0.5)
        logger.info("✓ Switched to keyboard mode")

        # Test typing
        await ble.type_text("test")
        await asyncio.sleep(0.5)
        logger.info("✓ Typed test text")

        # Disconnect
        await ble.disconnect()
        logger.info("✓ Disconnected")

        logger.info("=== BLE Test Passed ===")
        return True

    except Exception as e:
        logger.error(f"BLE test failed: {e}")
        return False
    finally:
        if ble.is_connected():
            await ble.disconnect()


def test_screen_capture(config: AutomationConfig) -> bool:
    """
    Test screen capture from HDMI device.

    Args:
        config: Automation configuration

    Returns:
        True if test successful, False otherwise
    """
    logger.info("=== Screen Capture Test ===")

    try:
        frame = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height
        )

        if frame is None:
            logger.error("Failed to capture screen")
            return False

        # Save test image
        output_path = save_debug_image(
            frame,
            "test_capture.jpg",
            config.screenshot_dir,
            timestamp=True
        )

        logger.info(f"✓ Captured {frame.shape[1]}x{frame.shape[0]} frame")
        logger.info(f"✓ Saved to {output_path}")
        logger.info("=== Capture Test Passed ===")
        return True

    except Exception as e:
        logger.error(f"Capture test failed: {e}")
        return False


async def perform_windows_login(
    config: AutomationConfig,
    password: str
) -> bool:
    """
    Perform automated Windows login.

    5-Stage Pipeline:
    1. Initialization - Load models, connect ESP32
    2. Screen Capture - Capture Windows login screen
    3. Screen Analysis - YOLO detection + OCR
    4. Input Control - Send password via BLE
    5. Verification - Verify login success

    Args:
        config: Automation configuration
        password: Login password

    Returns:
        True if login successful, False otherwise
    """
    progress = ProgressLogger(5, logger)

    # ===================================================================
    # Phase 1: Initialization
    # ===================================================================
    logger.info("=" * 70)
    logger.info("Phase 1: Initialization")
    logger.info("=" * 70)

    try:
        # Load YOLO model
        logger.info("Loading DocLayout-YOLO model...")
        model = load_yolo_model(config.doclayout_model_path, config.detection_device)

        # Load OCR reader
        logger.info("Loading EasyOCR reader...")
        ocr_reader = load_ocr_reader(config.ocr_languages, config.ocr_use_gpu)

        # Connect to ESP32
        logger.info("Connecting to ESP32...")
        ble = BLEController(
            device_name=config.esp32_device_name,
            service_uuid=config.ble_service_uuid,
            rx_char_uuid=config.ble_rx_char_uuid,
            tx_char_uuid=config.ble_tx_char_uuid
        )

        if not await ble.connect(timeout=10.0):
            logger.error("Failed to connect to ESP32")
            return False

        # Switch to keyboard mode
        await ble.switch_to_keyboard_mode()
        await asyncio.sleep(0.3)

        progress.step("Initialization complete")
        debug_pause("初期化完了。Enterで続行...", config.debug_mode)

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False

    # ===================================================================
    # Phase 2: Screen Capture (Initial Screen)
    # ===================================================================
    logger.info("=" * 70)
    logger.info("Phase 2: Screen Capture (Initial Screen)")
    logger.info("=" * 70)

    try:
        frame = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height
        )

        if frame is None:
            logger.error("Failed to capture screen")
            await ble.disconnect()
            return False

        # Save initial screen capture
        save_debug_image(frame, "01_initial_screen.jpg", config.screenshot_dir)

        progress.step(f"Screen captured: {frame.shape[1]}x{frame.shape[0]}")
        debug_pause(f"画面キャプチャ完了: {frame.shape}", config.debug_mode)

    except Exception as e:
        logger.error(f"Screen capture failed: {e}")
        await ble.disconnect()
        return False

    # ===================================================================
    # Phase 3: Screen Analysis (Optional - for verification)
    # ===================================================================
    logger.info("=" * 70)
    logger.info("Phase 3: Screen Analysis (Optional)")
    logger.info("=" * 70)

    try:
        # Analyze layout
        regions = analyze_layout(
            frame,
            model,
            confidence=config.detection_confidence,
            image_size=config.detection_image_size,
            device=config.detection_device
        )

        logger.info(f"Detected {len(regions)} regions")

        # Extract text (optional, for debugging)
        if config.debug_mode and len(regions) > 0:
            ocr_results = extract_text(frame, regions, ocr_reader)
            logger.debug(f"Extracted {len(ocr_results)} text segments")

            # Visualize detections
            vis_image = visualize_detections(frame, regions)
            save_debug_image(vis_image, "02_initial_analysis.jpg", config.screenshot_dir)

        progress.step(f"Screen analysis complete: {len(regions)} regions detected")
        debug_pause(f"検出完了: {len(regions)}個の領域", config.debug_mode)

    except Exception as e:
        logger.warning(f"Screen analysis warning: {e}")
        logger.info("Continuing with fallback input method...")

    # ===================================================================
    # Phase 4: Input Control
    # ===================================================================
    logger.info("=" * 70)
    logger.info("Phase 4: Input Control")
    logger.info("=" * 70)

    try:
        # Step 4.1: Press Enter to show password input field
        logger.info("Step 4.1: Pressing Enter to show password field...")
        await ble.send_command("key:enter")
        await asyncio.sleep(1.5)  # Wait for password screen to appear
        debug_pause("Enterキー送信完了。パスワード画面を確認...", config.debug_mode)

        # Capture password screen
        password_screen = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height
        )

        if password_screen is not None:
            save_debug_image(password_screen, "03_password_screen.jpg", config.screenshot_dir)

            # Analyze password screen
            if config.debug_mode:
                password_regions = analyze_layout(
                    password_screen,
                    model,
                    confidence=config.detection_confidence,
                    image_size=config.detection_image_size,
                    device=config.detection_device
                )

                password_ocr = extract_text(password_screen, password_regions, ocr_reader)
                password_field = find_password_field(
                    password_regions,
                    password_ocr,
                    password_screen.shape
                )

                if password_field:
                    logger.info(
                        f"Password field detected: Region {password_field.region.index} "
                        f"at ({password_field.region.center['x']}, {password_field.region.center['y']}) "
                        f"via {password_field.detection_method}"
                    )

                    # Visualize password field
                    vis_password = visualize_detections(
                        password_screen,
                        password_regions,
                        password_field
                    )
                    save_debug_image(vis_password, "04_password_field_detected.jpg", config.screenshot_dir)

                    # Step 4.2: Click password field
                    logger.info("Step 4.2: Clicking password field...")
                    await ble.switch_to_mouse_mode()
                    await asyncio.sleep(0.3)

                    # Move to password field and click
                    field_x = password_field.region.center['x']
                    field_y = password_field.region.center['y']
                    await ble.move_mouse_to_position(field_x, field_y)
                    await ble.click()
                    await asyncio.sleep(0.5)

                    # Switch back to keyboard mode
                    await ble.switch_to_keyboard_mode()
                    await asyncio.sleep(0.3)

                    debug_pause(f"パスワードフィールドをクリック: ({field_x}, {field_y})", config.debug_mode)
                else:
                    logger.warning("Password field not detected - using keyboard-only method")
                    # Field is likely already focused, continue with typing
            else:
                logger.info("Skipping field detection in non-debug mode")

        # Step 4.3: Type password
        logger.info("Step 4.3: Typing password...")
        await ble.type_text(password)
        await asyncio.sleep(1.0)
        debug_pause("パスワード入力完了", config.debug_mode)

        # Step 4.4: Press Enter to login
        logger.info("Step 4.4: Pressing Enter to login...")
        await ble.send_command("key:enter")
        await asyncio.sleep(0.5)

        progress.step("Input control complete")
        debug_pause("入力完了。Enterで続行...", config.debug_mode)

    except Exception as e:
        logger.error(f"Input control failed: {e}")
        await ble.disconnect()
        return False

    # ===================================================================
    # Phase 5: Verification
    # ===================================================================
    logger.info("=" * 70)
    logger.info("Phase 5: Verification")
    logger.info("=" * 70)

    try:
        if config.auto_verify:
            # Wait for login to process
            logger.info("Waiting for login to process...")
            await asyncio.sleep(3.0)

            # Capture post-login screen
            verify_frame = capture_screen(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height
            )

            if verify_frame is not None:
                save_debug_image(verify_frame, "05_post_login_screen.jpg", config.screenshot_dir)

                # Simple verification: check if screen changed
                # (More sophisticated verification could analyze desktop elements)
                import cv2
                diff = cv2.absdiff(frame, verify_frame)
                diff_score = diff.mean()

                logger.info(f"Screen difference score: {diff_score:.2f}")

                if diff_score > 10:  # Threshold for significant change
                    logger.info("✓ Screen changed - login likely successful")
                    success = True
                else:
                    logger.warning("⚠ Screen unchanged - login may have failed")
                    success = False
            else:
                logger.warning("Could not capture verification screen")
                success = None
        else:
            logger.info("Verification skipped (auto_verify=false)")
            success = None

        progress.complete("All phases complete")

    except Exception as e:
        logger.warning(f"Verification warning: {e}")
        success = None
    finally:
        # Cleanup
        await ble.disconnect()

    logger.info("=" * 70)
    if success is True:
        logger.info("✓ Windows login automation completed successfully")
    elif success is False:
        logger.warning("⚠ Windows login automation completed with warnings")
    else:
        logger.info("✓ Windows login automation completed (verification skipped)")
    logger.info("=" * 70)

    return success if success is not None else True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Windows PC Auto-Login via HDMI Capture and ESP32 BLE Control"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Windows login password (overrides .env)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with step-by-step pauses"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip login verification"
    )
    parser.add_argument(
        "--test-ble",
        action="store_true",
        help="Test BLE connection only"
    )
    parser.add_argument(
        "--test-capture",
        action="store_true",
        help="Test screen capture only"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to .env file (default: project root)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.env_file)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Override debug mode if specified
    if args.debug:
        config.debug_mode = True
    elif args.no_debug:
        config.debug_mode = False

    # Override verification if specified
    if args.no_verify:
        config.auto_verify = False

    # Override password if specified
    if args.password:
        config.password = args.password

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"windows_login_{timestamp}.log"
    setup_logging(
        log_level=config.log_level,
        log_file=log_file,
        debug_mode=config.debug_mode
    )

    logger.info("Windows PC Auto-Login Script")
    logger.info(f"Configuration: {config}")

    # Run tests if requested
    if args.test_ble:
        success = asyncio.run(test_ble_connection(config))
        return 0 if success else 1

    if args.test_capture:
        success = test_screen_capture(config)
        return 0 if success else 1

    # Validate password
    if not config.password:
        logger.error("Password not set. Use --password or set WINDOWS_LOGIN_PASSWORD in .env")
        return 1

    # Perform login
    try:
        success = asyncio.run(perform_windows_login(config, config.password))
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
