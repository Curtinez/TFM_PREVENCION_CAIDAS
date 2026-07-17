

#include <M5Unified.h>
#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth no está habilitado. En Arduino IDE: Tools > Partition Scheme, elige uno que incluya BT, y asegúrate de tener el core ESP32 con BT habilitado.
#endif

// Nombre con el que aparecerá el dispositivo al emparejar por Bluetooth
static const char* BT_DEVICE_NAME = "M5StickC-IMU";

BluetoothSerial SerialBT;

// Frecuencia de muestreo objetivo (Hz). El M5StickC Plus (MPU6886)
// puede dar hasta ~200 Hz de forma fiable; 100 Hz es un buen balance
// entre resolución temporal y volumen de datos por Bluetooth.
static const uint32_t SAMPLE_INTERVAL_MS = 10; // 100 Hz

// Fuerza de calibración (0 = desactivada, 1-255 = intensidad)
static constexpr const uint8_t calib_value = 64;
static uint8_t calib_countdown = 0;

void startCalibration()
{
  calib_countdown = 10; // 10 segundos de calibración
  M5.Imu.setCalibration(calib_value, calib_value, 0); // accel + gyro, sin magnetómetro
  M5.Display.fillScreen(TFT_BLUE);
  M5.Display.setCursor(0, 0);
  M5.Display.println("Calibrando...");
  M5.Display.println("Gira el M5 en");
  M5.Display.println("distintos angulos");
}

void stopCalibration()
{
  calib_countdown = 0;
  M5.Imu.setCalibration(0, 0, 0);
  M5.Imu.saveOffsetToNVS();
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.println("Listo.");
  M5.Display.println("Btn A: recalibrar");
}
void setup()
{
  auto cfg = M5.config();
  M5.begin(cfg);

  Serial.begin(115200);

  // Inicia Bluetooth Serial (Bluetooth clásico SPP)
  SerialBT.begin(BT_DEVICE_NAME);
  M5.Display.setRotation(1);
  M5.Display.setTextSize(2);

  if (M5.Imu.getType() == m5::imu_none)
  {
    M5.Display.println("IMU no detectado!");
    for (;;) { delay(1000); }
  }

  // Intenta cargar calibración previa; si no existe, calibra ahora
  if (!M5.Imu.loadOffsetFromNVS())
  {
    startCalibration();
  }
  else
  {
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0, 0);
    M5.Display.println("Listo.");
    M5.Display.println("Btn A: recalibrar");
  }
}

void loop()
{
  M5.update();

  // Botón A del M5StickC Plus: relanzar calibración manualmente
  if (M5.BtnA.wasClicked())
  {
    startCalibration();
  }

  static uint32_t last_sample_ms = 0;
  uint32_t now = millis();

  // Muestreo a intervalo fijo (no dependemos solo de M5.Imu.update())
  if (now - last_sample_ms >= SAMPLE_INTERVAL_MS)
  {
    last_sample_ms = now;

    if (M5.Imu.update())
    {
      auto data = M5.Imu.getImuData();

      // Envía por Bluetooth: timestamp,ax,ay,az,gx,gy,gz
      SerialBT.printf("%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f\n",
        now,
        data.accel.x, data.accel.y, data.accel.z,
        data.gyro.x, data.gyro.y, data.gyro.z);
    }
  }

  // Manejo de la cuenta atrás de calibración (una vez por segundo)
  static uint32_t last_calib_check = 0;
  if (calib_countdown && (now - last_calib_check >= 1000))
  {
    last_calib_check = now;
    calib_countdown--;
    M5.Display.fillRect(0, 60, 240, 30, TFT_BLUE);
    M5.Display.setCursor(0, 60);
    M5.Display.printf("Restan: %d s", calib_countdown);
    if (calib_countdown == 0)
    {
      stopCalibration();
    }
  }

  // Muestra en pantalla el estado de conexión Bluetooth (opcional, ligero)
  static bool was_connected = false;
  bool is_connected = SerialBT.hasClient();
  if (is_connected != was_connected)
  {
    was_connected = is_connected;
    M5.Display.fillRect(0, 100, 240, 30, TFT_BLACK);
    M5.Display.setCursor(0, 100);
    M5.Display.println(is_connected ? "BT: conectado" : "BT: esperando...");
  }
}