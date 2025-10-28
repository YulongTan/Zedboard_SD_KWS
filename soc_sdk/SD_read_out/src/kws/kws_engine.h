#ifndef KWS_ENGINE_H_
#define KWS_ENGINE_H_

#include <stddef.h>

#include "xil_types.h"
#include "xstatus.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_SD_MOUNT_POINT      "0:/"
#define KWS_DEFAULT_WEIGHT_PATH "0:/kws/kws_weights.bin"

#define KWS_TARGET_SAMPLE_RATE   16000U
#define KWS_INPUT_ROWS           40U
#define KWS_INPUT_COLS           98U
#define KWS_INPUT_DEPTH          1U
#define KWS_CONV1_OUT_CH         32U
#define KWS_CONV2_OUT_CH         64U
#define KWS_CONV3_OUT_CH         128U
#define KWS_FC1_OUT_UNITS        256U

XStatus KwsEngine_Initialize(const char *weight_file_path);
void    KwsEngine_Shutdown(void);
int     KwsEngine_IsReady(void);
XStatus KwsEngine_ProcessRecording(const int32_t *stereo_buffer,
                                   size_t frames_per_channel,
                                   u32 *out_class_index,
                                   float *out_confidence);
const float *KwsEngine_GetLogits(u32 *num_classes);

#ifdef __cplusplus
}
#endif

#endif /* KWS_ENGINE_H_ */
