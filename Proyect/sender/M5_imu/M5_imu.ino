

#include <M5Unified.h>
#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth no está habilitado. En Arduino IDE: Tools > Partition Scheme, elige uno que incluya BT, y asegúrate de tener el core ESP32 con BT habilitado.
#endif

static const char* BT_DEVICE_NAME = "M5StickC-IMU";
BluetoothSerial SerialBT;

static const uint32_t SAMPLE_INTERVAL_MS = 10; // ~100 Hz

static constexpr const uint8_t calib_value = 64;
static uint8_t calib_countdown = 0;

uint32_t muestras_enviadas = 0;
uint32_t muestras_ultimo_segundo = 0;
uint32_t tasa_actual_hz = 0;

void dibujarPantallaListo()
{
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setCursor(0, 0);
  M5.Display.println("Listo.");
}

void startCalibration()
{
  calib_countdown = 10;
  M5.Imu.setCalibration(calib_value, calib_value, 0);
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
  dibujarPantallaListo();
}

void setup()
{
  auto cfg = M5.config();
  M5.begin(cfg);

  Serial.begin(115200);
  SerialBT.begin(BT_DEVICE_NAME);
  M5.Display.setRotation(1);
  M5.Display.setTextSize(2);

  if (M5.Imu.getType() == m5::imu_none)
  {
    M5.Display.println("IMU no detectado!");
    for (;;) { delay(1000); }
  }

  if (!M5.Imu.loadOffsetFromNVS())
  {
    startCalibration();
  }
  else
  {
    dibujarPantallaListo();
  }
}

void loop()
{
  M5.update();

  if (M5.BtnA.wasClicked())
  {
    startCalibration();
  }

  static uint32_t last_sample_ms = 0;
  uint32_t now = millis();

  if (now - last_sample_ms >= SAMPLE_INTERVAL_MS)
  {
    last_sample_ms = now;

    if (M5.Imu.update())
    {
      auto data = M5.Imu.getImuData();

      SerialBT.printf("%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f\n",
        now,
        data.accel.x, data.accel.y, data.accel.z,
        data.gyro.x, data.gyro.y, data.gyro.z);

      muestras_enviadas++;
      muestras_ultimo_segundo++;
    }
  }

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

  // Estado en pantalla (BT, muestras enviadas, tasa real): una vez por segundo
  static uint32_t last_estado_ms = 0;
  if (!calib_countdown && (now - last_estado_ms >= 1000))
  {
    last_estado_ms = now;
    tasa_actual_hz = muestras_ultimo_segundo;
    muestras_ultimo_segundo = 0;

    bool conectado = SerialBT.hasClient();
    M5.Display.fillRect(0, 20, 240, 100, TFT_BLACK);
    M5.Display.setCursor(0, 20);
    M5.Display.println(conectado ? "BT: conectado" : "BT: esperando...");
    M5.Display.printf("Muestras: %lu\n", muestras_enviadas);
    M5.Display.printf("Tasa: %lu Hz\n", tasa_actual_hz);
    M5.Display.println("Btn A: recalibrar");
  }
}
