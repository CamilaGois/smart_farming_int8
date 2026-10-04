#ifndef MODEL_CONFIG_H_
#define MODEL_CONFIG_H_

static constexpr float FEATURE_MEAN[2] = {24.693574905f, 65.089775085f};
static constexpr float FEATURE_STD[2] = {5.353726387f, 14.494213104f};
static constexpr float INPUT_SCALE = 0.014532667f;
static constexpr int INPUT_ZERO_POINT = -3;
static constexpr float OUTPUT_SCALE = 0.003906250f;
static constexpr int OUTPUT_ZERO_POINT = -128;
static constexpr int NUM_CLASSES = 4;
static const char* CLASS_NAMES[NUM_CLASSES] = {"AMENO", "INTERMEDIARIO", "QUENTE_SECO", "QUENTE_UMIDO"};

#endif