/*
 * ===========================================================================
 *  harness-node  —  M5Core2 BLE phrase sender with B-button mode switching
 * ===========================================================================
 *
 *  Two BLE roles, switched by physical button B (only one active at a time,
 *  so the HID server and the UART server never coexist):
 *
 *    TYPE   (default):  BLE HID Keyboard (HOGP).  Pairs with the target host.
 *                       A long-press  -> types the active phrase.
 *
 *    CONFIG:            BLE UART (Nordic UART Service).  Pairs with the Mac.
 *                       Receives `setphrase:` to override the phrase at
 *                       runtime.  Stored in NVS; survives reboot.
 *                       Falls back to DEFAULT_PHRASE (config.h) if unset.
 *
 *  Hardware:  M5Core2 (ESP32, classic)  +  M5Unified
 *  BLE stack: bundled NimBLE (Arduino-ESP32 core 3.x)  — no external libs.
 *
 *  Build (see README.md for full commands):
 *    arduino-cli compile --fqbn esp32:esp32:m5stack-core2 harness-node
 *
 *  Button map:
 *    A  long-press  -> send active phrase   (TYPE mode only)
 *    B  click       -> toggle TYPE <-> CONFIG (re-initializes BLE, ~1s)
 *    C              -> unused
 * ===========================================================================
 */

#include <M5Unified.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLEHIDDevice.h>
#include <BLE2902.h>
#include <Preferences.h>
#include "config.h"

// ---------------------------------------------------------------------------
//  Nordic UART Service UUIDs (same as wireless-input-bridge)
// ---------------------------------------------------------------------------
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// HID Keyboard appearance code (GATT appearance values).
#define APPEARANCE_HID_KEYBOARD 0x03C1

// ---------------------------------------------------------------------------
//  State
// ---------------------------------------------------------------------------
enum BleMode { MODE_TYPE, MODE_CONFIG };
BleMode currentMode = MODE_TYPE;

Preferences prefs;
String activePhrase;

// BLE objects (only the ones for the active mode are non-null).
BLEServer         *server   = nullptr;
BLECharacteristic *txChar   = nullptr;   // CONFIG mode notify
BLEHIDDevice      *hid      = nullptr;   // TYPE mode
BLECharacteristic *inputKb  = nullptr;   // TYPE mode keyboard input report
bool bleConnected = false;

// Forward declarations.
void startMode(BleMode m, bool initial);
void initTypeMode();
void initConfigMode();
void processUartCommand(String cmd);
void bleLog(const String &msg);
void typePhrase(const String &s);
void drawAll();
void drawButtons();

// ===========================================================================
//  BLE connection tracking (shared by both modes)
// ===========================================================================
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) {
    bleConnected = true;
    Serial.println("[BLE] connected");
  }
  void onDisconnect(BLEServer *s) {
    bleConnected = false;
    Serial.println("[BLE] disconnected; re-advertising");
    s->startAdvertising();
  }
};

// ===========================================================================
//  CONFIG mode: Nordic UART Service command handling
// ===========================================================================
class UartCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) {
    String v = c->getValue();
    if (v.length() > 0) processUartCommand(v);
  }
};

void processUartCommand(String cmd) {
  cmd.trim();
  Serial.print("CMD: "); Serial.println(cmd);

  if (cmd.startsWith("setphrase:")) {
    String p = cmd.substring(10);
    prefs.begin(NVS_NAMESPACE, false);
    prefs.putString(NVS_KEY_PHRASE, p);
    prefs.end();
    activePhrase = p;
    bleLog("[OK] set (" + String(p.length()) + " chars)");
    Serial.println("phrase set: " + p);
  } else if (cmd == "clearphrase") {
    prefs.begin(NVS_NAMESPACE, false);
    prefs.remove(NVS_KEY_PHRASE);
    prefs.end();
    activePhrase = DEFAULT_PHRASE;
    bleLog("[OK] cleared -> default");
    Serial.println("phrase cleared");
  } else if (cmd == "get") {
    bleLog("[PHRASE] " + activePhrase);
  } else if (cmd == "help" || cmd == "?") {
    bleLog("[HELP] setphrase:<txt> | clearphrase | get");
  } else {
    bleLog("[ERR] unknown: " + cmd);
  }
}

// Notify a log/reply message to the connected Mac (chunked for BLE MTU).
void bleLog(const String &msg) {
  if (currentMode != MODE_CONFIG || !bleConnected || !txChar) return;
  const size_t chunk = 20;
  size_t len = msg.length();
  for (size_t i = 0; i < len; i += chunk) {
    String c = msg.substring(i, min(i + chunk, len));
    txChar->setValue(c.c_str());
    txChar->notify();
    delay(5);
  }
}

// ===========================================================================
//  TYPE mode: BLE HID keyboard
// ===========================================================================
//  Keyboard report descriptor (Report ID 1):
//    byte 0 : modifier bitmask (Ctrl/Shift/Alt/GUI x2)
//    byte 1 : reserved (0)
//    bytes 2-7 : up to 6 simultaneous keycodes
//  Per HOGP spec the Report characteristic VALUE does NOT include the report
//  id byte — the Report Reference descriptor (set by inputReport(1)) already
//  identifies it. Prepending the id shifts the report by one byte and the host
//  sees Ctrl+<keycode> (= invisible control chars), so nothing gets typed.
const uint8_t hidReportMap[] = {
  0x05, 0x01,              // Usage Page (Generic Desktop)
  0x09, 0x06,              // Usage (Keyboard)
  0xA1, 0x01,              // Collection (Application)
  0x85, 0x01,              //   Report ID (1)
  0x05, 0x07,              //   Usage Page (Keyboard/Keypad)
  0x19, 0xE0,              //   Usage Minimum (0xE0)
  0x29, 0xE7,              //   Usage Maximum (0xE7)
  0x15, 0x00,              //   Logical Minimum (0)
  0x25, 0x01,              //   Logical Maximum (1)
  0x75, 0x01,              //   Report Size (1)
  0x95, 0x08,              //   Report Count (8)
  0x81, 0x02,              //   Input (Data,Var,Abs) — modifiers
  0x95, 0x01,              //   Report Count (1)
  0x75, 0x08,              //   Report Size (8)
  0x81, 0x01,              //   Input (Cnst,Arr,Abs) — reserved
  0x95, 0x06,              //   Report Count (6)
  0x75, 0x08,              //   Report Size (8)
  0x15, 0x00,              //   Logical Minimum (0)
  0x26, 0x97, 0x00,        //   Logical Maximum (151)
  0x05, 0x07,              //   Usage Page (Keyboard)
  0x19, 0x00,              //   Usage Minimum (0)
  0x2A, 0x97, 0x00,        //   Usage Maximum (151)
  0x81, 0x00,              //   Input (Data,Arr,Abs) — key array
  0xC0                     // End Collection
};

struct __attribute__((packed)) KeyReport {
  uint8_t modifiers;
  uint8_t reserved;
  uint8_t keys[6];
};

// Map one ASCII char (US layout) to a HID keycode + modifier. false if unsupported.
bool asciiToKey(char c, uint8_t &key, uint8_t &mod) {
  mod = 0;
  if (c == '\n' || c == '\r') { key = 0x28; return true; }   // Enter
  if (c == '\t')              { key = 0x2B; return true; }   // Tab
  if (c == '\b')              { key = 0x2A; return true; }   // Backspace
  if (c == ' ')               { key = 0x2C; return true; }   // Space
  if (c >= 'a' && c <= 'z')   { key = 0x04 + (c - 'a'); return true; }
  if (c >= 'A' && c <= 'Z')   { key = 0x04 + (c - 'A'); mod = 0x02; return true; }  // Shift
  if (c >= '1' && c <= '9')   { key = 0x1E + (c - '1'); return true; }
  if (c == '0')               { key = 0x27; return true; }
  switch (c) {
    case '!': key = 0x1E; mod = 0x02; return true;
    case '@': key = 0x1F; mod = 0x02; return true;
    case '#': key = 0x20; mod = 0x02; return true;
    case '$': key = 0x21; mod = 0x02; return true;
    case '%': key = 0x22; mod = 0x02; return true;
    case '^': key = 0x23; mod = 0x02; return true;
    case '&': key = 0x24; mod = 0x02; return true;
    case '*': key = 0x25; mod = 0x02; return true;
    case '(': key = 0x26; mod = 0x02; return true;
    case ')': key = 0x27; mod = 0x02; return true;
    case '-': key = 0x2D; return true;
    case '_': key = 0x2D; mod = 0x02; return true;
    case '=': key = 0x2E; return true;
    case '+': key = 0x2E; mod = 0x02; return true;
    case '[': key = 0x2F; return true;
    case '{': key = 0x2F; mod = 0x02; return true;
    case ']': key = 0x30; return true;
    case '}': key = 0x30; mod = 0x02; return true;
    case '\\': key = 0x31; return true;
    case '|': key = 0x31; mod = 0x02; return true;
    case ';': key = 0x33; return true;
    case ':': key = 0x33; mod = 0x02; return true;
    case '\'': key = 0x34; return true;
    case '"': key = 0x34; mod = 0x02; return true;
    case '`': key = 0x35; return true;
    case '~': key = 0x35; mod = 0x02; return true;
    case ',': key = 0x36; return true;
    case '<': key = 0x36; mod = 0x02; return true;
    case '.': key = 0x37; return true;
    case '>': key = 0x37; mod = 0x02; return true;
    case '/': key = 0x38; return true;
    case '?': key = 0x38; mod = 0x02; return true;
  }
  return false;
}

// Press one key (with optional modifier) then release it.
void sendKey(uint8_t keycode, uint8_t modifier) {
  if (!inputKb) return;
  KeyReport r;
  r.modifiers = modifier;
  r.reserved = 0;
  memset(r.keys, 0, 6);
  r.keys[0] = keycode;
  inputKb->setValue((uint8_t *)&r, sizeof(r));   // 8 bytes, no report-id prefix
  inputKb->notify();
  delay(KEY_PRESS_MS);

  // release
  r.modifiers = 0;
  r.keys[0] = 0;
  inputKb->setValue((uint8_t *)&r, sizeof(r));
  inputKb->notify();
  delay(KEY_RELEASE_MS);
}

void typePhrase(const String &s) {
  if (currentMode != MODE_TYPE || !bleConnected || !inputKb) {
    Serial.println("[type] not ready (wrong mode / not connected)");
    return;
  }
  Serial.println("[type] sending: " + s);
  for (size_t i = 0; i < s.length(); i++) {
    uint8_t kc, mod;
    if (asciiToKey(s[i], kc, mod)) {
      sendKey(kc, mod);
    } else {
      Serial.printf("[type] skip unsupported char 0x%02X\n", (uint8_t)s[i]);
    }
    delay(TYPE_CHAR_DELAY_MS);
  }
  Serial.println("[type] done");
}

// ===========================================================================
//  Mode initialization
// ===========================================================================
static void resetBlePointers() {
  server = nullptr;
  txChar = nullptr;
  hid = nullptr;
  inputKb = nullptr;
  bleConnected = false;
}

void initTypeMode() {
  Serial.println("== init TYPE mode ==");
  BLEDevice::init(BLE_NAME_TYPE);

  // Just-works bonding so the host accepts keyboard input.
  BLESecurity *pSec = new BLESecurity();
  pSec->setCapability(ESP_IO_CAP_NONE);
  pSec->setAuthenticationMode(true, false, true);  // bond=true, MITM=false, SC=true

  server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  hid = new BLEHIDDevice(server);
  // NOTE: the getter manufacturer() creates the characteristic; the setter-only
  // form would dereference an uncreated characteristic and panic.
  hid->manufacturer()->setValue(String(BLE_NAME_TYPE));
  hid->pnp(0x02, 0x1209, 0x0001, 0x0100);   // generic USB-IF VID/PID placeholder
  hid->hidInfo(0x00, 0x01);
  hid->reportMap((uint8_t *)hidReportMap, sizeof(hidReportMap));
  inputKb = hid->inputReport(1);
  hid->setBatteryLevel(100);
  hid->startServices();

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->setAppearance(APPEARANCE_HID_KEYBOARD);
  adv->addServiceUUID(hid->hidService()->getUUID());
  adv->setScanResponse(true);
  adv->setMinPreferred(0x06);
  adv->setMaxPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("TYPE advertising started");
}

void initConfigMode() {
  Serial.println("== init CONFIG mode ==");
  BLEDevice::init(BLE_NAME_CONFIG);

  server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *svc = server->createService(SERVICE_UUID);

  txChar = svc->createCharacteristic(CHARACTERISTIC_UUID_TX,
                                     BLECharacteristic::PROPERTY_NOTIFY);
  txChar->addDescriptor(new BLE2902());

  BLECharacteristic *rx = svc->createCharacteristic(CHARACTERISTIC_UUID_RX,
                                                    BLECharacteristic::PROPERTY_WRITE);
  rx->setCallbacks(new UartCallbacks());

  svc->start();

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(BLEUUID(SERVICE_UUID));
  adv->setScanResponse(true);
  adv->setMinPreferred(0x06);
  adv->setMaxPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("CONFIG advertising started");
}

// Tear down (if needed) and bring up the requested mode.
void startMode(BleMode m, bool initial) {
  if (!initial) {
    Serial.println("== switching mode: deinit ==");
    // delete HID wrapper while its server is still alive, then release stack.
    if (hid) { delete hid; hid = nullptr; }
    BLEDevice::deinit(true);
    delay(250);
    resetBlePointers();
  }
  if (m == MODE_TYPE) initTypeMode();
  else                initConfigMode();
  currentMode = m;
  Serial.println(m == MODE_TYPE ? ">> TYPE mode active" : ">> CONFIG mode active");
}

// ===========================================================================
//  Phrase persistence
// ===========================================================================
String loadPhrase() {
  prefs.begin(NVS_NAMESPACE, true);
  String p = prefs.isKey(NVS_KEY_PHRASE)
               ? prefs.getString(NVS_KEY_PHRASE, DEFAULT_PHRASE)
               : DEFAULT_PHRASE;
  prefs.end();
  return p;
}

// ===========================================================================
//  Display
// ===========================================================================
static const int BTN_Y = 200;
static const int BTN_H = 40;
static bool needRedraw     = true;
static bool btnPrev[3]     = { false, false, false };

void drawAll() {
  auto &d = M5.Display;
  d.startWrite();
  d.fillScreen(TFT_BLACK);

  d.setColor(TFT_WHITE);
  d.setTextSize(2);
  d.setCursor(8, 4);
  d.print("Harness Node");

  // mode badge
  const char *ms = (currentMode == MODE_TYPE) ? "TYPE" : "CONFIG";
  uint16_t   mc  = (currentMode == MODE_TYPE) ? TFT_GREEN : TFT_ORANGE;
  d.setColor(mc);
  d.fillRoundRect(8, 26, 110, 24, 4);
  d.setColor(TFT_BLACK);
  d.setTextSize(2);
  d.setCursor(20, 30);
  d.print(ms);

  // BLE status
  d.setColor(bleConnected ? TFT_GREEN : TFT_DARKGREY);
  d.setTextSize(2);
  d.setCursor(150, 30);
  d.print(bleConnected ? "BLE:ON" : "BLE:--");

  // phrase
  d.setColor(TFT_CYAN);
  d.setTextSize(1);
  d.setCursor(8, 66);
  d.print("ACTIVE PHRASE:");
  d.setColor(TFT_WHITE);
  d.setTextSize(2);
  d.setCursor(8, 82);
  String shown = activePhrase;
  if (shown.length() > 18) shown = shown.substring(0, 18) + "..";
  d.print(shown);

  // hints
  d.setColor(TFT_DARKGREY);
  d.setTextSize(1);
  d.setCursor(8, 116);
  if (currentMode == MODE_TYPE)
    d.print("A: long-press = type phrase");
  else
    d.print("A: disabled (in CONFIG)");
  d.setCursor(8, 130);
  d.print("B: switch mode    C: --");
  d.endWrite();

  drawButtons();
}

void drawButtons() {
  auto &d = M5.Display;
  const int w = d.width() / 3;
  const char *labels[3] = { "A", "B", "C" };
  const uint16_t cols[3] = { TFT_RED, TFT_BLUE, TFT_DARKGREY };
  bool pressed[3] = {
    M5.BtnA.isPressed(), M5.BtnB.isPressed(), M5.BtnC.isPressed()
  };
  d.startWrite();
  for (int i = 0; i < 3; i++) {
    int x = w * i;
    if (pressed[i]) {
      d.setColor(cols[i]);
      d.fillRect(x + 2, BTN_Y + 2, w - 4, BTN_H - 4);
      d.setColor(TFT_BLACK);
    } else {
      d.setColor(TFT_BLACK);
      d.fillRect(x + 2, BTN_Y + 2, w - 4, BTN_H - 4);
      d.setColor(cols[i]);
      d.drawRect(x + 2, BTN_Y + 2, w - 4, BTN_H - 4);
    }
    d.setTextSize(3);
    int lw = d.textWidth(labels[i]);
    int lh = d.fontHeight();
    d.setCursor(x + (w - lw) / 2, BTN_Y + (BTN_H - lh) / 2);
    d.print(labels[i]);
  }
  d.endWrite();
}

// ===========================================================================
//  Setup / Loop
// ===========================================================================
void setup() {
  M5.begin();
  if (M5.Display.width() < M5.Display.height()) {
    M5.Display.setRotation(M5.Display.getRotation() ^ 1);
  }
  M5.BtnA.setHoldThresh(HOLD_THRESHOLD_MS);

  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== harness-node ===");

  activePhrase = loadPhrase();
  Serial.println("active phrase: " + activePhrase);

  startMode(MODE_TYPE, true);
}

void loop() {
  M5.update();

  // B click -> toggle mode
  if (M5.BtnB.wasClicked()) {
    startMode(currentMode == MODE_TYPE ? MODE_CONFIG : MODE_TYPE, false);
    needRedraw = true;
  }

  // A long-press -> type phrase (TYPE mode only)
  if (currentMode == MODE_TYPE && M5.BtnA.wasHold()) {
    typePhrase(activePhrase);
  }

  // Full redraw on demand, otherwise just refresh the button strip when it changes.
  if (needRedraw) {
    drawAll();
    needRedraw = false;
    btnPrev[0] = M5.BtnA.isPressed();
    btnPrev[1] = M5.BtnB.isPressed();
    btnPrev[2] = M5.BtnC.isPressed();
  } else {
    bool p[3] = {
      M5.BtnA.isPressed(), M5.BtnB.isPressed(), M5.BtnC.isPressed()
    };
    if (p[0] != btnPrev[0] || p[1] != btnPrev[1] || p[2] != btnPrev[2]) {
      drawButtons();
      btnPrev[0] = p[0]; btnPrev[1] = p[1]; btnPrev[2] = p[2];
    }
  }

  M5.delay(10);
}
