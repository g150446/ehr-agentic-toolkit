/*
  BLE UART Mouse and Keyboard Control

  Controls USB HID Mouse and Keyboard via BLE UART commands.

  Mode Switching Commands:
  - "mode:mouse"     : Switch to mouse mode
  - "mode:keyboard"  : Switch to keyboard mode

  Mouse Commands:
  - "click"       : Click left mouse button
  - "up:10"       : Move cursor up 10 pixels
  - "down:10"     : Move cursor down 10 pixels
  - "left:10"     : Move cursor left 10 pixels
  - "right:10"    : Move cursor right 10 pixels
  - "scroll:5"    : Scroll by 5 units

  Keyboard Commands:
  - "type:Hello"  : Type text string

  Hardware:
  - ESP32-S2 or ESP32-S3 with Native USB support
  - M5AtomS3U tested and working

  IMPORTANT: In Arduino IDE, set Tools > USB Mode > "USB-OTG (TinyUSB)"

  Based on:
  - ButtonMouseControl example
  - KeyboardMessage example
  - BLE UART example
*/

#if !defined(ARDUINO_USB_MODE)
#error This ESP32 SoC has no Native USB interface
#elif ARDUINO_USB_MODE == 1
#error Wrong USB Mode! Please set Tools > USB Mode > "USB-OTG (TinyUSB)" in Arduino IDE
#else

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "USB.h"
#include "USBHIDMouse.h"
#include "USBHIDKeyboard.h"

USBHIDMouse Mouse;
USBHIDKeyboard Keyboard;

// Mode tracking
enum Mode { MOUSE_MODE, KEYBOARD_MODE };
Mode currentMode = MOUSE_MODE;

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// BLE UART Service UUIDs
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println("Device connected");
  };

  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println("Device disconnected");
  }
};

// Parse command and control mouse
void processCommand(String command) {
  command.trim();
  Serial.print("Processing command: ");
  Serial.println(command);

  if (command == "click") {
    // Click left mouse button
    Mouse.click(MOUSE_LEFT);
    Serial.println("Click executed");
  }
  else if (command.startsWith("up:")) {
    int value = command.substring(3).toInt();
    Mouse.move(0, -value, 0);
    Serial.print("Move up: ");
    Serial.println(value);
  }
  else if (command.startsWith("down:")) {
    int value = command.substring(5).toInt();
    Mouse.move(0, value, 0);
    Serial.print("Move down: ");
    Serial.println(value);
  }
  else if (command.startsWith("left:")) {
    int value = command.substring(5).toInt();
    Mouse.move(-value, 0, 0);
    Serial.print("Move left: ");
    Serial.println(value);
  }
  else if (command.startsWith("right:")) {
    int value = command.substring(6).toInt();
    Mouse.move(value, 0, 0);
    Serial.print("Move right: ");
    Serial.println(value);
  }
  else if (command.startsWith("scroll:")) {
    int value = command.substring(7).toInt();
    Mouse.move(0, 0, value);
    Serial.print("Scroll: ");
    Serial.println(value);
  }
  else if (command.startsWith("mode:")) {
    String mode = command.substring(5);
    if (mode == "mouse") {
      currentMode = MOUSE_MODE;
      Serial.println("Switched to MOUSE mode");
    }
    else if (mode == "keyboard") {
      currentMode = KEYBOARD_MODE;
      Serial.println("Switched to KEYBOARD mode");
    }
    else {
      Serial.println("Unknown mode. Use 'mode:mouse' or 'mode:keyboard'");
    }
  }
  else if (command.startsWith("type:")) {
    String text = command.substring(5);
    Keyboard.print(text);
    Serial.print("Typed: ");
    Serial.println(text);
  }
  else {
    Serial.println("Unknown command");
  }
}

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String rxValue = pCharacteristic->getValue();

    if (rxValue.length() > 0) {
      Serial.println("*********");
      Serial.print("Received Value: ");
      Serial.println(rxValue);
      Serial.println("*********");

      // Process the received command
      processCommand(rxValue);
    }
  }
};

void setup() {
  Serial.begin(115200);
  delay(1000); // Wait for serial to be ready
  Serial.println("\n\n=================================");
  Serial.println("BLE UART Mouse & Keyboard Control");
  Serial.println("=================================");
  Serial.println("Board: M5AtomS3U / ESP32-S3");

  // Initialize USB HID Mouse and Keyboard
  Serial.println("Initializing USB Mouse and Keyboard...");
  Mouse.begin();
  Keyboard.begin();
  USB.begin();
  Serial.println("USB Mouse initialized - OK");
  Serial.println("USB Keyboard initialized - OK");

  // Create the BLE Device
  Serial.println("Initializing BLE...");
  BLEDevice::init("BLE Mouse & Keyboard");

  // Create the BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);
  Serial.println("BLE Server created - OK");

  // Create BLE Characteristics
  pTxCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID_TX,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pTxCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID_RX,
    BLECharacteristic::PROPERTY_WRITE
  );
  pRxCharacteristic->setCallbacks(new MyCallbacks());

  // Start the service
  pService->start();

  // Start advertising
  pServer->getAdvertising()->start();
  Serial.println("BLE Advertising started - OK");
  Serial.println("=================================");
  Serial.println("Device Name: BLE Mouse & Keyboard");
  Serial.println("Status: Waiting for connection...");
  Serial.println("=================================");
  Serial.println("\nAvailable commands:");
  Serial.println("Mode switching:");
  Serial.println("  mode:mouse     - Switch to mouse mode");
  Serial.println("  mode:keyboard  - Switch to keyboard mode");
  Serial.println("\nMouse commands:");
  Serial.println("  click          - Click left mouse button");
  Serial.println("  up:N           - Move up N pixels");
  Serial.println("  down:N         - Move down N pixels");
  Serial.println("  left:N         - Move left N pixels");
  Serial.println("  right:N        - Move right N pixels");
  Serial.println("  scroll:N       - Scroll by N units");
  Serial.println("\nKeyboard commands:");
  Serial.println("  type:TEXT      - Type text string");
  Serial.println("\nCurrent mode: MOUSE");
  Serial.println("=================================\n");
}

void loop() {
  // Handle BLE disconnection/reconnection
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    Serial.println("Start advertising");
    oldDeviceConnected = deviceConnected;
  }

  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
    Serial.println("Client connected");
  }

  delay(10);
}

#endif /* ARDUINO_USB_MODE */
