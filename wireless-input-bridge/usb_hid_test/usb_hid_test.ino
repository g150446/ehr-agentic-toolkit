/*
  USB HID Mouse 最小テストスケッチ（BLEなし）

  BLEコードを完全に除いたUSB HIDマウスのみのテスト。
  これでWindowsに認識されれば、BLEがUSBを妨害していることが原因。
  認識されなければ、USB HID自体がこのボード+Windowsで機能しない。

  Arduino IDE 設定:
    Tools → USB CDC On Boot → Disabled
    Tools → USB Mode       → USB-OTG (TinyUSB)
    Board: M5AtomS3U (or ESP32-S3)
*/

#include "USB.h"
#include "USBHIDMouse.h"

USBHIDMouse Mouse;

void setup() {
  Mouse.begin();
  USB.begin();
}

void loop() {
  delay(2000);
  Mouse.move(100, 0, 0);   // 右へ100px
  delay(2000);
  Mouse.move(-100, 0, 0);  // 左へ100px（元に戻す）
}
