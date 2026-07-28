#pragma once

// ===========================================================================
//  harness-node configuration
//  Edit values below to customize behavior. No other code changes needed.
// ===========================================================================

// --- Phrase ---------------------------------------------------------------
// Used when no override has been stored via the CONFIG (BLE UART) mode.
// ASCII only (a-z A-Z 0-9 and US-layout symbols / space / tab / newline).
#define DEFAULT_PHRASE       "Hello EHR"

// --- BLE device names (shown to the host when scanning) -------------------
#define BLE_NAME_TYPE        "Harness Node"          // TYPE   mode (HID keyboard)
#define BLE_NAME_CONFIG      "Harness Node [CFG]"    // CONFIG mode (UART)

// --- Typing timing (milliseconds) ----------------------------------------
// Per-character pacing while sending the phrase as a BLE HID keyboard.
#define TYPE_CHAR_DELAY_MS   20   // gap between characters
#define KEY_PRESS_MS         25   // key held down duration
#define KEY_RELEASE_MS       15   // gap between release and next press

// --- Button behavior ------------------------------------------------------
// M5Unified long-press (hold) threshold in ms. A must be held this long
// before the phrase is sent. Default in M5Unified is 500ms.
#define HOLD_THRESHOLD_MS    500

// --- NVS (Preferences) keys for the stored phrase override ----------------
#define NVS_NAMESPACE        "hn"
#define NVS_KEY_PHRASE       "phrase"
