idf_component_register(
    SRCS
        "main.cpp"
        "model_data.cc"
    INCLUDE_DIRS "."
    REQUIRES
        esp-tflite-micro
        driver
        esp_timer
)