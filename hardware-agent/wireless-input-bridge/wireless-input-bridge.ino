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
  - "key:lbracket" / "key:rbracket" (JIS-aware: outputs '[' / ']')
  - "key:lbrace" / "key:rbrace" (JIS-aware: outputs '{' / '}')
  - "key:lparen" / "key:rparen" / "key:percent" / "key:colon" / "key:newline"
  - "key:up" / "key:down" / "key:left" / "key:right" (arrow keys)
  - "key:shift_right" : Shift+Right (IME segment extension)
  - "key:alt_tab" : Alt+Tab (window switch)
  - "key:win" : Windows key (single press)
  - "key:win_up" : Win+Up Arrow (snap window to maximize)

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

// Forward declaration — defined later in the file.
void bleLog(const String &msg);

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println("BLE client connected");
    bleLog("[LOG] BLE client connected");
  }
  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println("BLE client disconnected");
    bleLog("[LOG] BLE client disconnected");
  }
};

// Send a log message via BLE TX characteristic (Notify).
// If no BLE client is connected the message is silently dropped.
void bleLog(const String &msg) {
  if (deviceConnected && pTxCharacteristic != NULL) {
    // BLE Notify MTU is typically 20 bytes; chunk longer messages.
    const size_t chunkSize = 20;
    size_t len = msg.length();
    for (size_t i = 0; i < len; i += chunkSize) {
      String chunk = msg.substring(i, min(i + chunkSize, len));
      pTxCharacteristic->setValue(chunk.c_str());
      pTxCharacteristic->notify();
      delay(5);
    }
  }
}

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
    // HID 0x30 (US ']') → JIS keyboard layout maps this to '['.
    // Keyboard.write('[') sends HID 0x2F which JIS maps to '@' (wrong).
    Keyboard.write(']');
    Serial.println("-> key: [");
    return true;
  }
  if (keyName == "rbracket" || keyName == "right_bracket") {
    // HID 0x31 (US '\') → JIS keyboard layout maps this to ']'.
    // Keyboard.write(']') sends HID 0x30 which JIS maps to '[' (wrong).
    Keyboard.write('\\');
    Serial.println("-> key: ]");
    return true;
  }
  if (keyName == "lbrace" || keyName == "left_brace") {
    // Shift + HID 0x30 → JIS '{' (shift of '[')
    tapShiftedAscii(']');
    Serial.println("-> key: {");
    return true;
  }
  if (keyName == "rbrace" || keyName == "right_brace") {
    // Shift + HID 0x31 → JIS '}' (shift of ']')
    tapShiftedAscii('\\');
    Serial.println("-> key: }");
    return true;
  }
  if (keyName == "lparen" || keyName == "left_paren") {
    // JIS keyboard: Shift+8 = '(' (US layout has Shift+9 = '(', but JIS differs)
    tapShiftedAscii('8');
    Serial.println("-> key: (");
    return true;
  }
  if (keyName == "rparen" || keyName == "right_paren") {
    // JIS keyboard: Shift+9 = ')' (US layout has Shift+0 = ')', but JIS differs)
    tapShiftedAscii('9');
    Serial.println("-> key: )");
    return true;
  }
  if (keyName == "percent") {
    tapShiftedAscii('5');
    Serial.println("-> key: %");
    return true;
  }
  if (keyName == "colon") {
    // JIS keyboard: ':' is at HID 0x34 (US apostrophe/quote position), unshifted.
    // tapShiftedAscii(';') sends Shift+0x33 which gives '+' on JIS (not ':').
    Keyboard.write('\'');  // HID 0x34 → JIS ':' (unshifted colon key)
    Serial.println("-> key: :");
    return true;
  }
  if (keyName == "plus") {
    // JIS keyboard: '+' is Shift+HID 0x33 (the ';:' key shifted).
    // Keyboard.write('+') sends Shift+0x2E (US '=') which gives '~' on JIS.
    tapShiftedAscii(';');  // Shift+HID 0x33 → JIS '+'
    Serial.println("-> key: +");
    return true;
  }
  if (keyName == "zenkaku") {
    // ASCII 0x60 (backtick) maps to HID keycode 0.x35 via en_US layout table.
    // On Windows with Japanese 106/109 keyboard layout, HID 0x35 = 半角/全角 key,
    // which toggles the IME between hiragana and alphanumeric input mode.
    Keyboard.write('`');
    Serial.println("-> key: zenkaku (半角/全角)");
    return true;
  }
  if (keyName == "f7") {
    Keyboard.write(KEY_F7);
    Serial.println("-> key: F7 (全角カタカナ変換)");
    return true;
  }
  if (keyName == "f8") {
    Keyboard.write(KEY_F8);
    Serial.println("-> key: F8 (半角カタカナ変換)");
    return true;
  }
  if (keyName == "f6") {
    Keyboard.write(KEY_F6);
    Serial.println("-> key: F6 (全角ひらがな変換)");
    return true;
  }
  if (keyName == "up") {
    Keyboard.write(KEY_UP_ARROW);
    Serial.println("-> key: Up Arrow");
    return true;
  }
  if (keyName == "down") {
    Keyboard.write(KEY_DOWN_ARROW);
    Serial.println("-> key: Down Arrow");
    return true;
  }
  if (keyName == "left") {
    Keyboard.write(KEY_LEFT_ARROW);
    Serial.println("-> key: Left Arrow");
    return true;
  }
  if (keyName == "right") {
    Keyboard.write(KEY_RIGHT_ARROW);
    Serial.println("-> key: Right Arrow");
    return true;
  }
  if (keyName == "end") {
    Keyboard.write(KEY_END);
    Serial.println("-> key: End");
    return true;
  }
  if (keyName == "home") {
    Keyboard.write(KEY_HOME);
    Serial.println("-> key: Home");
    return true;
  }
  if (keyName == "shift_right") {
    Keyboard.press(KEY_LEFT_SHIFT);
    delay(5);
    Keyboard.press(KEY_RIGHT_ARROW);
    delay(5);
    Keyboard.releaseAll();
    Serial.println("-> key: Shift+Right");
    return true;
  }
  if (keyName == "ctrl_a" || keyName == "select_all") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press('a');
    delay(50);
    Keyboard.releaseAll();
    Serial.println("-> key: Ctrl+A");
    return true;
  }
  if (keyName == "ctrl_z" || keyName == "undo") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press('z');
    delay(50);
    Keyboard.releaseAll();
    Serial.println("-> key: Ctrl+Z");
    return true;
  }
  if (keyName == "ctrl_l") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press('l');
    delay(50);
    Keyboard.releaseAll();
    Serial.println("-> key: Ctrl+L");
    return true;
  }
  if (keyName == "ctrl_x" || keyName == "cut") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press('x');
    delay(50);
    Keyboard.releaseAll();
    Serial.println("-> key: Ctrl+X");
    return true;
  }
  if (keyName == "ctrl_v" || keyName == "paste") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press('v');
    delay(50);
    Keyboard.releaseAll();
    Serial.println("-> key: Ctrl+V");
    return true;
  }
  if (keyName == "ctrl_c" || keyName == "copy") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press('c');
    delay(50);
    Keyboard.releaseAll();
    Serial.println("-> key: Ctrl+C");
    return true;
  }
  if (keyName == "ctrl_end") {
    Keyboard.press(KEY_LEFT_CTRL);
    Keyboard.press(KEY_END);
    delay(50);
    Keyboard.releaseAll();
    Serial.println("-> key: Ctrl+End");
    return true;
  }
  if (keyName == "alt_tab") {
    Keyboard.press(KEY_LEFT_ALT);
    delay(50);
    Keyboard.press(KEY_TAB);
    delay(200);
    Keyboard.releaseAll();
    Serial.println("-> key: Alt+Tab");
    return true;
  }
  if (keyName == "win") {
    Keyboard.press(KEY_LEFT_GUI);
    delay(100);
    Keyboard.releaseAll();
    Serial.println("-> key: Win (Windows key)");
    return true;
  }
  if (keyName == "win_up") {
    Keyboard.press(KEY_LEFT_GUI);
    delay(5);
    Keyboard.press(KEY_UP_ARROW);
    delay(5);
    Keyboard.releaseAll();
    Serial.println("-> key: Win+Up");
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

  } else if (command == "mdown") {
    AbsMouse.press(MOUSE_LEFT);
    Serial.println("-> mdown");

  } else if (command == "mup") {
    AbsMouse.release(MOUSE_LEFT);
    Serial.println("-> mup");

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
  String msg = "Connecting to WiFi: " + String(WIFI_SSID);
  Serial.println(msg);
  bleLog("[LOG] " + msg);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi not connected — OTA unavailable");
    bleLog("[LOG] WiFi not connected — OTA unavailable");
    return;
  }

  String ip = WiFi.localIP().toString();
  Serial.print("\nWiFi connected, IP: "); Serial.println(ip);
  bleLog("[LOG] WiFi connected, IP: " + ip);

  ArduinoOTA.setHostname("ble-hid-bridge");

  ArduinoOTA.onStart([]() {
    Serial.println("OTA update starting...");
    bleLog("[LOG] OTA update starting...");
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nOTA update complete. Rebooting...");
    bleLog("[LOG] OTA update complete. Rebooting...");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("OTA: %u%%\r", (progress * 100) / total);
  });
  ArduinoOTA.onError([](ota_error_t error) {
    String errMsg = "OTA Error[" + String(error) + "]: ";
    if      (error == OTA_AUTH_ERROR)    errMsg += "Auth Failed";
    else if (error == OTA_BEGIN_ERROR)   errMsg += "Begin Failed";
    else if (error == OTA_CONNECT_ERROR) errMsg += "Connect Failed";
    else if (error == OTA_RECEIVE_ERROR) errMsg += "Receive Failed";
    else if (error == OTA_END_ERROR)     errMsg += "End Failed";
    else                                 errMsg += "Unknown";
    Serial.println(errMsg);
    bleLog("[ERR] " + errMsg);
  });

  ArduinoOTA.begin();
  Serial.println("OTA ready — hostname: ble-hid-bridge");
  bleLog("[LOG] OTA ready — hostname: ble-hid-bridge");
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
  bleLog("[LOG] BLE advertising started");
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
