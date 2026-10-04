#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dht.h"


#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "esp_log.h"

#include "esp_timer.h"
#include "esp_rom_sys.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "model_config.h"

static const char *TAG = "SMART_FARMING";

#define DHT_PIN GPIO_NUM_4

static bool esperar_nivel(int nivel, int timeout_us)
{
    int64_t inicio = esp_timer_get_time();

    while (gpio_get_level(DHT_PIN) != nivel) {
        if ((esp_timer_get_time() - inicio) > timeout_us) {
            return false;
        }
    }

    return true;
}

static bool ler_dht22(float *temperatura, float *umidade)
{
    uint8_t dados[5] = {0};

    gpio_set_direction(DHT_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(DHT_PIN, 0);

    vTaskDelay(pdMS_TO_TICKS(2));

    gpio_set_level(DHT_PIN, 1);
    esp_rom_delay_us(30);

    gpio_set_direction(DHT_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(DHT_PIN, GPIO_PULLUP_ONLY);

    if (!esperar_nivel(0, 120)) return false;
    if (!esperar_nivel(1, 120)) return false;
    if (!esperar_nivel(0, 120)) return false;

    for (int i = 0; i < 40; i++) {

        if (!esperar_nivel(1, 100)) return false;

        int64_t inicio_alto = esp_timer_get_time();

        if (!esperar_nivel(0, 120)) return false;

        int64_t duracao = esp_timer_get_time() - inicio_alto;

        dados[i / 8] <<= 1;

        if (duracao > 40) {
            dados[i / 8] |= 1;
        }
    }

    uint8_t checksum =
        (dados[0] + dados[1] + dados[2] + dados[3]) & 0xFF;

    if (checksum != dados[4]) {
        return false;
    }

    uint16_t raw_umidade =
        ((uint16_t)dados[0] << 8) | dados[1];

    uint16_t raw_temperatura =
        ((uint16_t)(dados[2] & 0x7F) << 8) | dados[3];

    *umidade = raw_umidade / 10.0f;
    *temperatura = raw_temperatura / 10.0f;

    if (dados[2] & 0x80) {
        *temperatura = -*temperatura;
    }

    return true;
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Smart Farming INT8 iniciado.");
    ESP_LOGI(TAG, "ESP32-S3 + DHT22 + TensorFlow Lite Micro");

    const tflite::Model *modelo = tflite::GetModel(g_model);

    if (modelo->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Versao do modelo TFLite incompatível.");
        return;
    }

    tflite::MicroMutableOpResolver<2> resolver;

    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    constexpr int kTensorArenaSize = 32 * 1024;
    static uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter interpreter(
        modelo,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Erro ao alocar tensores.");
        return;
    }

    TfLiteTensor *entrada = interpreter.input(0);
    TfLiteTensor *saida = interpreter.output(0);

    ESP_LOGI(TAG, "Modelo INT8 carregado: %u bytes", g_model_len);

    vTaskDelay(pdMS_TO_TICKS(2000));

    while (true) {

        float temperatura = 0.0f;
        float umidade = 0.0f;

esp_err_t res = dht_read_float_data(
    DHT_TYPE_AM2301,
    DHT_PIN,
    &umidade,
    &temperatura
);

if (res != ESP_OK) {
    ESP_LOGW(TAG, "Falha na leitura do DHT22.");
} else {
    // normalização + inferência INT8
}

            float entrada_float[2];

            entrada_float[0] =
                (temperatura - FEATURE_MEAN[0]) / FEATURE_STD[0];

            entrada_float[1] =
                (umidade - FEATURE_MEAN[1]) / FEATURE_STD[1];

            for (int i = 0; i < 2; i++) {

                int valor =
                    (int)roundf(
                        entrada_float[i] / INPUT_SCALE
                    ) + INPUT_ZERO_POINT;

                if (valor > 127) valor = 127;
                if (valor < -128) valor = -128;

                entrada->data.int8[i] = (int8_t)valor;
            }

            if (interpreter.Invoke() != kTfLiteOk) {

                ESP_LOGE(TAG, "Erro durante a inferencia.");

            } else {

                int classe = 0;

                for (int i = 1; i < NUM_CLASSES; i++) {
                    if (saida->data.int8[i] >
                        saida->data.int8[classe]) {
                        classe = i;
                    }
                }

                ESP_LOGI(
                    TAG,
                    "Temperatura: %.1f C | Umidade: %.1f %%",
                    temperatura,
                    umidade
                );

                ESP_LOGI(
                    TAG,
                    "Classe INT8: %s",
                    CLASS_NAMES[classe]
                );

                printf("Saidas INT8: ");

                for (int i = 0; i < NUM_CLASSES; i++) {

                    float valor =
                        (saida->data.int8[i] -
                         OUTPUT_ZERO_POINT) *
                         OUTPUT_SCALE;

                    printf(
                        "%s=%.3f ",
                        CLASS_NAMES[i],
                        valor
                    );
                }

                printf("\n\n");
            }
        }

        vTaskDelay(pdMS_TO_TICKS(2500));
    }
