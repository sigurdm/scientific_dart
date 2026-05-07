#define MINIAUDIO_IMPLEMENTATION
#include "../lib/src/miniaudio.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    ma_device device;
    float* ringBuffer;
    int bufferSize;
    int writeIndex;
} TunerContext;

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    TunerContext* pContext = (TunerContext*)pDevice->pUserData;
    if (pInput == NULL) return;

    float* fInput = (float*)pInput;
    for (ma_uint32 i = 0; i < frameCount; i++) {
        pContext->ringBuffer[pContext->writeIndex] = fInput[i];
        pContext->writeIndex = (pContext->writeIndex + 1) % pContext->bufferSize;
    }

    (void)pOutput;
}

TunerContext* tuner_init(int sampleRate, int ringBufferSize) {
    TunerContext* pContext = (TunerContext*)malloc(sizeof(TunerContext));
    pContext->bufferSize = ringBufferSize;
    pContext->ringBuffer = (float*)calloc(ringBufferSize, sizeof(float));
    pContext->writeIndex = 0;

    ma_device_config deviceConfig;
    deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate       = sampleRate;
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData         = pContext;

    if (ma_device_init(NULL, &deviceConfig, &pContext->device) != MA_SUCCESS) {
        free(pContext->ringBuffer);
        free(pContext);
        return NULL;
    }

    if (ma_device_start(&pContext->device) != MA_SUCCESS) {
        ma_device_uninit(&pContext->device);
        free(pContext->ringBuffer);
        free(pContext);
        return NULL;
    }

    return pContext;
}

void tuner_get_samples(TunerContext* pContext, float* output, int count) {
    int readIndex = (pContext->writeIndex - count + pContext->bufferSize) % pContext->bufferSize;
    for (int i = 0; i < count; i++) {
        output[i] = pContext->ringBuffer[(readIndex + i) % pContext->bufferSize];
    }
}

void tuner_close(TunerContext* pContext) {
    if (pContext == NULL) return;
    ma_device_stop(&pContext->device);
    ma_device_uninit(&pContext->device);
    free(pContext->ringBuffer);
    free(pContext);
}
