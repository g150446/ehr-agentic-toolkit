/*
  BLE UART + USB HID Bridge

  Receives commands via BLE UART (Nordic UART Service) from Mac and translates
  them to USB HID mouse/keyboard actions on the connected Windows PC.

  Uses USBHIDAbsoluteMouse for precise pixel-accurate cursor positioning.
  USBHIDAbsoluteMouse and USBHIDMouse (relative) share a static initializer in
  the base class, so only one can be active — we use absolute only.

  Mouse Commands:
  - "moveto:X,Y"  : Move to absolute HID position (X,Y in 0-32767)
  - "move:DX,DY"  : Move relative to current tracked position (DX,DY in HID units)
  - "click"       : Left click at current position
  - "scroll:N"    : Scroll wheel N units (positive=down, negative=up)

  Keyboard Commands:
  - "mode:mouse"    : (no-op, kept for protocol compatibility)
  - "mode:keyboard" : (no-op, kept for protocol compatibility)
  - "type:TEXT"   : Type text string
  - "key:enter"   : Press Enter
  - "key:tab"     : Press Tab
  - "key:backspace": Press Backspace
  - "key:esc"     : Press Escape
  - "key:delete"  : Press Delete
  - "key:zenkaku" : Press 半角/全角 IME toggle key (HID 0x35, JP keyboard layout)
  - "key:lbracket" / "key:rbracket" / "key:lparen" / "key:rparen"
  - "key:percent" / "key:colon" / "key:newline"

  OTA Update:
  - Connect to WiFi defined in wifi_config.h
  - Use Arduino IDE "Upload via Network" or arduino-cli with --protocol network
  - Hostname: ble-hid-bridge

  Hardware:
  - M5AtomS3U (ESP32-S3, Native USB)

  Arduino IDE settings:
  - Tools > USB Mode       > "USB-OTG (TinyUSB)"
  - Tools > USB CDC On Boot> "Enabled"
*/

#if !defined(ARDUINO_USB_MODE)
#error This ESP32 SoC has no Native USB interface
#elif ARDUINO_USB_MODE == 1
#error Wrong USB Mode! Set Tools > USB Mode > "USB-OTG (TinyUSB)" in Arduino IDE
#else

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include "USB.h"
#include "USBHIDMouse.h"      // defines both USBHIDAbsoluteMouse and USBHIDMouse
#include "USBHIDKeyboard.h"
#include "wifi_config.h"      // #define WIFI_SSID / WIFI_PASSWORD  (gitignored)

// Use absolute mouse only — sharing the base-class static initializer with
// USBHIDMouse means only one can be registered; absolute is what we need.
USBHIDAbsoluteMouse AbsMouse;
USBHIDKeyboard      Keyboard;

// Tracked absolute position (0-32767 HID range).
// moveto:X,Y sets this; move:DX,DY adds to it.
// Initialized to screen center so relative moves work before first moveto.
static int16_t g_abs_x = 16383;
static int16_t g_abs_y = 16383;

BLEServer           *pServer = NULL;
BLECharacteristic   *pTxCharacteristic;
bool deviceConnected    = false;
bool oldDeviceConnected = false;

#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println("BLE client connected");
  }
  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println("BLE client disconnected");
  }
};

// Scroll in int8_t chunks (scroll wheel is still relative)
void mouseScroll(int amount) {
  while (amount != 0) {
    int8_t chunk = (amount > 127) ? 127 : (amount < -128) ? -128 : (int8_t)amount;
    AbsMouse.move(g_abs_x, g_abs_y, chunk);
    amount -= chunk;
    delay(5);
  }
}

void tapShiftedAscii(uint8_t key) {
  Keyboard.press(KEY_LEFT_SHIFT);
  delay(5);
  Keyboard.press(key);
  delay(5);
  Keyboard.releaseAll();
  delay(30);
}

bool pressNamedKey(const String &keyName) {
  if (keyName == "enter" || keyName == "return" || keyName == "newline") {
    Keyboard.write(KEY_RETURN);
    Serial.println("-> key: Enter");
    return true;
  }
  if (keyName == "tab") {
    Keyboard.write(KEY_TAB);
    Serial.println("-> key: Tab");
    return true;
  }
  if (keyName == "backspace") {
    Keyboard.write(KEY_BACKSPACE);
    Serial.println("-> key: Backspace");
    return true;
  }
  if (keyName == "delete") {
    Keyboard.write(KEY_DELETE);
    Serial.println("-> key: Delete");
    return true;
  }
  if (keyName == "esc") {
    Keyboard.write(KEY_ESC);
    Serial.println("-> key: Esc");
    return true;
  }
  if (keyName == "space") {
    Keyboard.write(' ');
    Serial.println("-> key: Space");
    return true;
  }
  if (keyName == "lbracket" || keyName == "left_bracket") {
    Keyboard.write('[');
    Serial.println("-> key: [");
    return true;
  }
  if (keyName == "rbracket" || keyName == "right_bracket") {
    Keyboard.write(']');
    Serial.println("-> key: ]");
    return true;
  }
  if (keyName == "lparen" || keyName == "left_paren") {
    tapShiftedAscii('9');
    Serial.println("-> key: (");
    return true;
  }
  if (keyName == "rparen" || keyName == "right_paren") {
    tapShiftedAscii('0');
    Serial.println("-> key: )");
    return true;
  }
  if (keyName == "percent") {
    tapShiftedAscii('5');
    Serial.println("-> key: %");
    return true;
  }
  if (keyName == "colon") {
    tapShiftedAscii(';');
    Serial.println("-> key: :");
    return true;
  }
  if (keyName == "zenkaku") {
    // ASCII 0x60 (backtick) maps to HID keycode 0x35 via en_US layout table.
    // On Windows with Japanese 106/109 keyboard layout, HID 0x35 = 半角/全角 key,
    // which toggles the IME between hiragana and alphanumeric input mode.
    Keyboard.write('`');
    Serial.println("-> key: zenkaku (半角/全角)");
    return true;
  }
  return false;
}

void processCommand(String command) {
  command.trim();
  Serial.print("CMD: ");
  Serial.println(command);

  if (command == "click") {
    AbsMouse.click(MOUSE_LEFT);
    Serial.println("-> click");

  } else if (command == "rclick") {
    AbsMouse.click(MOUSE_RIGHT);
    Serial.println("-> rclick");

  } else if (command.startsWith("moveto:")) {
    // moveto:X,Y  — absolute HID coordinates (0-32767)
    String coords = command.substring(7);
    int comma = coords.indexOf(',');
    if (comma > 0) {
      int16_t ax = (int16_t)coords.substring(0, comma).toInt();
      int16_t ay = (int16_t)coords.substring(comma + 1).toInt();
      g_abs_x = constrain(ax, 0, 32767);
      g_abs_y = constrain(ay, 0, 32767);
      AbsMouse.move(g_abs_x, g_abs_y);
      Serial.print("-> moveto "); Serial.print(g_abs_x); Serial.print(","); Serial.println(g_abs_y);
    }

  } else if (command.startsWith("move:")) {
    // move:DX,DY  — relative HID delta added to current tracked position
    String coords = command.substring(5);
    int comma = coords.indexOf(',');
    if (comma >= 0) {
      int dx = coords.substring(0, comma).toInt();
      int dy = coords.substring(comma + 1).toInt();
      g_abs_x = (int16_t)constrain((int)g_abs_x + dx, 0, 32767);
      g_abs_y = (int16_t)constrain((int)g_abs_y + dy, 0, 32767);
      AbsMouse.move(g_abs_x, g_abs_y);
      Serial.print("-> move dx="); Serial.print(dx);
      Serial.print(" dy="); Serial.print(dy);
      Serial.print(" -> ("); Serial.print(g_abs_x); Serial.print(","); Serial.print(g_abs_y); Serial.println(")");
    }

  } else if (command.startsWith("scroll:")) {
    int value = command.substring(7).toInt();
    mouseScroll(value);
    Serial.print("-> scroll "); Serial.println(value);

  } else if (command.startsWith("mode:")) {
    // Protocol compatibility — no action needed
    Serial.print("-> mode: "); Serial.println(command.substring(5));

  } else if (command.startsWith("type:")) {
    String text = command.substring(5);
    for (int i = 0; i < (int)text.length(); i++) {
      if (text[i] == '\\' && i + 1 < (int)text.length()) {
        if (text[i + 1] == 'n') {
          pressNamedKey("newline");
          i++;
          continue;
        }
        if (text[i + 1] == 't') {
          pressNamedKey("tab");
          i++;
          continue;
        }
      }
      Keyboard.write((uint8_t)text[i]);
      delay(30);
    }
    Serial.print("-> type: "); Serial.println(text);

  } else if (command.startsWith("key:")) {
    String keyName = command.substring(4);
    if (!pressNamedKey(keyName)) {
      Serial.print("-> unknown key: "); Serial.println(keyName);
    }

  } else {
    Serial.println("-> unknown command");
  }
}

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String rxValue = pCharacteristic->getValue();
    if (rxValue.length() > 0) {
      processCommand(rxValue);
    }
  }
};

void setupWiFiOTA() {
  Serial.print("Connecting to WiFi: "); Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi not connected — OTA unavailable");
    return;
  }

  Serial.print("\nWiFi connected, IP: ");
  Serial.println(WiFi.localIP());

  ArduinoOTA.setHostname("ble-hid-bridge");

  ArduinoOTA.onStart([]() {
    Serial.println("OTA update starting...");
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nOTA update complete. Rebooting...");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("OTA: %u%%\r", (progress * 100) / total);
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("OTA Error[%u]: ", error);
    if      (error == OTA_AUTH_ERROR)    Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR)   Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR)     Serial.println("End Failed");
  });

  ArduinoOTA.begin();
  Serial.println("OTA ready — hostname: ble-hid-bridge");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== BLE UART + USB HID Bridge ===");

  USB.manufacturerName("ESP32");
  USB.productName("BLE HID Bridge");
  USB.serialNumber("00000002");
  USB.VID(0x303A);
  USB.PID(0x4005);

  AbsMouse.begin();
  Keyboard.begin();
  USB.begin();
  Serial.println("USB HID initialized");

  setupWiFiOTA();

  BLEDevice::init("BLE Mouse & Keyboard");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

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

  pService->start();
  pServer->getAdvertising()->start();

  Serial.println("BLE advertising started");
  Serial.println("Device: BLE Mouse & Keyboard");
  Serial.println("================================\n");
}

void loop() {
  ArduinoOTA.handle();

  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    Serial.println("Restarted BLE advertising");
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
  delay(10);
}

#endif /* ARDUINO_USB_MODE */
