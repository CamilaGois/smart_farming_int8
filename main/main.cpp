#include <stdio.h>
#include "esp_log.h"

static const char *TAG = "SMART_FARMING";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Smart Farming INT8 - ESP32-S3");
    ESP_LOGI(TAG, "Sistema iniciado com sucesso.");
}
