#include "kws_engine.h"

#include "xil_printf.h"
#include "ff.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

#define KWS_WEIGHT_MAGIC   0x4B575331u /* 'KWS1' */
#define KWS_WEIGHT_VERSION 0x00010000u

#define KWS_DECIMATION_FACTOR  (KWS_SOURCE_SAMPLE_RATE / KWS_TARGET_SAMPLE_RATE)
#define KWS_REQUIRED_SOURCE_FRAMES (KWS_TARGET_SAMPLE_RATE * KWS_DECIMATION_FACTOR)

#define KWS_WINDOW_SIZE 400U
#define KWS_HOP_LENGTH  160U
#define KWS_FFT_SIZE    KWS_WINDOW_SIZE
#define KWS_FFT_BINS    (KWS_FFT_SIZE / 2U + 1U)
#define KWS_NUM_MELS    KWS_INPUT_ROWS
#define KWS_NUM_FRAMES  KWS_INPUT_COLS

#define POOL_OUT_DIM(v) (((v) >= 2U) ? (((v) - 2U) / 2U + 1U) : 0U)

#define KWS_POOL1_ROWS POOL_OUT_DIM(KWS_INPUT_ROWS)
#define KWS_POOL1_COLS POOL_OUT_DIM(KWS_INPUT_COLS)
#define KWS_POOL2_ROWS POOL_OUT_DIM(KWS_POOL1_ROWS)
#define KWS_POOL2_COLS POOL_OUT_DIM(KWS_POOL1_COLS)
#define KWS_POOL3_ROWS POOL_OUT_DIM(KWS_POOL2_ROWS)
#define KWS_POOL3_COLS POOL_OUT_DIM(KWS_POOL2_COLS)

#define KWS_GAP_ROWS 5U
#define KWS_GAP_COLS 5U

typedef struct {
    u32 magic;
    u32 version;
    u32 num_classes;
    u32 reserved;
} __attribute__((packed)) KwsWeightHeader;

typedef struct {
    u32 num_classes;
    float *conv1_weights;
    float *conv1_bias;
    float *conv1_bn_scale;
    float *conv1_bn_bias;

    float *conv2_weights;
    float *conv2_bn_scale;
    float *conv2_bn_bias;

    float *conv3_weights;
    float *conv3_bn_scale;
    float *conv3_bn_bias;

    float *fc1_weights;
    float *fc1_bn_scale;
    float *fc1_bn_bias;

    float *fc_out_weights;
    float *fc_out_bias;
} KwsModel;

typedef struct {
    float *input_tensor;
    float *mono_buffer;
    float *fft_power;

    float *conv1_out;
    float *pool1_out;
    float *conv2_out;
    float *pool2_out;
    float *conv3_out;
    float *pool3_out;
    float *gap_out;
    float *flat;
    float *fc1_out;
    float *logits;
} KwsScratch;

typedef enum {
    KWS_ACT_NONE = 0,
    KWS_ACT_RELU,
    KWS_ACT_SIGN
} KwsActivation;

typedef struct {
    int initialized;
    float hann[KWS_WINDOW_SIZE];
    float cos_table[KWS_FFT_BINS * KWS_WINDOW_SIZE];
    float sin_table[KWS_FFT_BINS * KWS_WINDOW_SIZE];
    float mel_filter[KWS_NUM_MELS * KWS_FFT_BINS];
} KwsFeatureTables;

static FATFS gFatFs;
static int gFatFsMounted;
static KwsModel gModel;
static KwsScratch gScratch;
static KwsFeatureTables gFeatureTables;
static int gEngineReady;
static int gHasResult;

static void free_model(void);
static void free_scratch(void);
static XStatus mount_sd_if_needed(void);
static XStatus load_weights(const char *path);
static XStatus allocate_scratch(void);
static void init_feature_tables(void);
static XStatus extract_logmel(const int32_t *source_buffer,
                              size_t frames_per_channel,
                              float *out_tensor);
static void conv2d_forward(const float *input,
                           u32 in_channels,
                           u32 in_rows,
                           u32 in_cols,
                           const float *weights,
                           const float *bias,
                           const float *bn_scale,
                           const float *bn_bias,
                           u32 out_channels,
                           float *output,
                           KwsActivation activation);
static void maxpool2d(const float *input,
                      u32 channels,
                      u32 in_rows,
                      u32 in_cols,
                      float *output);
static void adaptive_avg_pool(const float *input,
                              u32 channels,
                              u32 in_rows,
                              u32 in_cols,
                              u32 out_rows,
                              u32 out_cols,
                              float *output);
static void dense_forward(const float *input,
                          u32 in_features,
                          const float *weights,
                          const float *bias,
                          const float *bn_scale,
                          const float *bn_bias,
                          u32 out_features,
                          float *output,
                          KwsActivation activation);
static void run_network(const float *input_tensor, float *logits);

XStatus KwsEngine_Initialize(const char *weight_file_path)
{
    const char *path = (weight_file_path != NULL) ? weight_file_path : KWS_DEFAULT_WEIGHT_PATH;

    if (gEngineReady) {
        return XST_SUCCESS;
    }

    if (mount_sd_if_needed() != XST_SUCCESS) {
        xil_printf("KWS: failed to mount SD card\r\n");
        return XST_FAILURE;
    }

    if (load_weights(path) != XST_SUCCESS) {
        xil_printf("KWS: failed to load weights from %s\r\n", path);
        free_model();
        return XST_FAILURE;
    }

    if (allocate_scratch() != XST_SUCCESS) {
        xil_printf("KWS: scratch allocation failed\r\n");
        free_model();
        free_scratch();
        return XST_FAILURE;
    }

    init_feature_tables();
    gEngineReady = 1;
    gHasResult = 0;
    xil_printf("KWS: engine ready, %lu classes\r\n", (unsigned long)gModel.num_classes);
    return XST_SUCCESS;
}

void KwsEngine_Shutdown(void)
{
    gEngineReady = 0;
    gHasResult = 0;
    free_model();
    free_scratch();
}

int KwsEngine_IsReady(void)
{
    return gEngineReady;
}

XStatus KwsEngine_MountSd(void)
{
    return mount_sd_if_needed();
}

XStatus KwsEngine_ProcessRecording(const int32_t *source_buffer,
                                   size_t frames_per_channel,
                                   u32 *out_class_index,
                                   float *out_confidence)
{
    if (!gEngineReady) {
        return XST_FAILURE;
    }

    if (source_buffer == NULL) {
        return XST_FAILURE;
    }

    if (frames_per_channel < KWS_REQUIRED_SOURCE_FRAMES) {
        xil_printf("KWS: insufficient frames (%lu < %lu)\r\n", (unsigned long)frames_per_channel,
                   (unsigned long)KWS_REQUIRED_SOURCE_FRAMES);
        return XST_FAILURE;
    }

    if (extract_logmel(source_buffer, frames_per_channel, gScratch.input_tensor) != XST_SUCCESS) {
        xil_printf("KWS: feature extraction failed\r\n");
        return XST_FAILURE;
    }

    run_network(gScratch.input_tensor, gScratch.logits);

    float max_logit = gScratch.logits[0];
    u32 max_index = 0U;
    for (u32 i = 1U; i < gModel.num_classes; ++i) {
        if (gScratch.logits[i] > max_logit) {
            max_logit = gScratch.logits[i];
            max_index = i;
        }
    }

    float sum = 0.0f;
    for (u32 i = 0U; i < gModel.num_classes; ++i) {
        sum += expf(gScratch.logits[i] - max_logit);
    }
    float confidence = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    gHasResult = 1;

    if (out_class_index != NULL) {
        *out_class_index = max_index;
    }
    if (out_confidence != NULL) {
        *out_confidence = confidence;
    }
    return XST_SUCCESS;
}

const float *KwsEngine_GetLogits(u32 *num_classes)
{
    if (!gHasResult) {
        return NULL;
    }
    if (num_classes != NULL) {
        *num_classes = gModel.num_classes;
    }
    return gScratch.logits;
}

static void free_model(void)
{
    free(gModel.conv1_weights);
    free(gModel.conv1_bias);
    free(gModel.conv1_bn_scale);
    free(gModel.conv1_bn_bias);
    free(gModel.conv2_weights);
    free(gModel.conv2_bn_scale);
    free(gModel.conv2_bn_bias);
    free(gModel.conv3_weights);
    free(gModel.conv3_bn_scale);
    free(gModel.conv3_bn_bias);
    free(gModel.fc1_weights);
    free(gModel.fc1_bn_scale);
    free(gModel.fc1_bn_bias);
    free(gModel.fc_out_weights);
    free(gModel.fc_out_bias);
    memset(&gModel, 0, sizeof(gModel));
}

static void free_scratch(void)
{
    free(gScratch.input_tensor);
    free(gScratch.mono_buffer);
    free(gScratch.fft_power);
    free(gScratch.conv1_out);
    free(gScratch.pool1_out);
    free(gScratch.conv2_out);
    free(gScratch.pool2_out);
    free(gScratch.conv3_out);
    free(gScratch.pool3_out);
    free(gScratch.gap_out);
    free(gScratch.flat);
    free(gScratch.fc1_out);
    free(gScratch.logits);
    memset(&gScratch, 0, sizeof(gScratch));
}

static XStatus mount_sd_if_needed(void)
{
    if (gFatFsMounted) {
        return XST_SUCCESS;
    }

    FRESULT res = f_mount(&gFatFs, KWS_SD_MOUNT_POINT, 1);
    if (res != FR_OK) {
        xil_printf("KWS: f_mount failed (%d)\r\n", (int)res);
        return XST_FAILURE;
    }

    gFatFsMounted = 1;
    return XST_SUCCESS;
}

static FRESULT read_exact(FIL *fil, void *dst, UINT bytes)
{
    UINT br = 0U;
    FRESULT res = f_read(fil, dst, bytes, &br);
    if (res != FR_OK) {
        return res;
    }
    if (br != bytes) {
        return FR_INT_ERR;
    }
    return FR_OK;
}

static XStatus load_weights(const char *path)
{
    FIL fil;
    FRESULT res = f_open(&fil, path, FA_READ);
    if (res != FR_OK) {
        xil_printf("KWS: f_open(%s) failed (%d)\r\n", path, (int)res);
        return XST_FAILURE;
    }

    KwsWeightHeader header;
    res = read_exact(&fil, &header, sizeof(header));
    if (res != FR_OK) {
        xil_printf("KWS: unable to read weight header (%d)\r\n", (int)res);
        f_close(&fil);
        return XST_FAILURE;
    }

    if (header.magic != KWS_WEIGHT_MAGIC) {
        xil_printf("KWS: invalid magic 0x%08lx\r\n", (unsigned long)header.magic);
        f_close(&fil);
        return XST_FAILURE;
    }
    if (header.version != KWS_WEIGHT_VERSION) {
        xil_printf("KWS: unsupported version 0x%08lx\r\n", (unsigned long)header.version);
        f_close(&fil);
        return XST_FAILURE;
    }
    if (header.num_classes == 0U) {
        xil_printf("KWS: num_classes is zero\r\n");
        f_close(&fil);
        return XST_FAILURE;
    }
    gModel.num_classes = header.num_classes;

    const size_t conv1_params = (size_t)KWS_CONV1_OUT_CH * KWS_INPUT_DEPTH * 3U * 3U;
    const size_t conv2_params = (size_t)KWS_CONV2_OUT_CH * KWS_CONV1_OUT_CH * 3U * 3U;
    const size_t conv3_params = (size_t)KWS_CONV3_OUT_CH * KWS_CONV2_OUT_CH * 3U * 3U;
    const size_t fc1_params   = (size_t)KWS_FC1_OUT_UNITS * KWS_CONV3_OUT_CH * KWS_GAP_ROWS * KWS_GAP_COLS;
    const size_t fc_out_params = (size_t)gModel.num_classes * KWS_FC1_OUT_UNITS;

    gModel.conv1_weights  = (float *)malloc(conv1_params * sizeof(float));
    gModel.conv1_bias     = (float *)calloc(KWS_CONV1_OUT_CH, sizeof(float));
    gModel.conv1_bn_scale = (float *)malloc(KWS_CONV1_OUT_CH * sizeof(float));
    gModel.conv1_bn_bias  = (float *)malloc(KWS_CONV1_OUT_CH * sizeof(float));

    gModel.conv2_weights  = (float *)malloc(conv2_params * sizeof(float));
    gModel.conv2_bn_scale = (float *)malloc(KWS_CONV2_OUT_CH * sizeof(float));
    gModel.conv2_bn_bias  = (float *)malloc(KWS_CONV2_OUT_CH * sizeof(float));

    gModel.conv3_weights  = (float *)malloc(conv3_params * sizeof(float));
    gModel.conv3_bn_scale = (float *)malloc(KWS_CONV3_OUT_CH * sizeof(float));
    gModel.conv3_bn_bias  = (float *)malloc(KWS_CONV3_OUT_CH * sizeof(float));

    gModel.fc1_weights    = (float *)malloc(fc1_params * sizeof(float));
    gModel.fc1_bn_scale   = (float *)malloc(KWS_FC1_OUT_UNITS * sizeof(float));
    gModel.fc1_bn_bias    = (float *)malloc(KWS_FC1_OUT_UNITS * sizeof(float));

    gModel.fc_out_weights = (float *)malloc(fc_out_params * sizeof(float));
    gModel.fc_out_bias    = (float *)malloc(gModel.num_classes * sizeof(float));

    if (!gModel.conv1_weights || !gModel.conv1_bias || !gModel.conv1_bn_scale || !gModel.conv1_bn_bias ||
        !gModel.conv2_weights || !gModel.conv2_bn_scale || !gModel.conv2_bn_bias ||
        !gModel.conv3_weights || !gModel.conv3_bn_scale || !gModel.conv3_bn_bias ||
        !gModel.fc1_weights || !gModel.fc1_bn_scale || !gModel.fc1_bn_bias ||
        !gModel.fc_out_weights || !gModel.fc_out_bias) {
        xil_printf("KWS: weight allocation failure\r\n");
        f_close(&fil);
        return XST_FAILURE;
    }

    res = read_exact(&fil, gModel.conv1_weights, conv1_params * sizeof(float));
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv1_bias, KWS_CONV1_OUT_CH * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv1_bn_scale, KWS_CONV1_OUT_CH * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv1_bn_bias, KWS_CONV1_OUT_CH * sizeof(float));
    }

    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv2_weights, conv2_params * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv2_bn_scale, KWS_CONV2_OUT_CH * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv2_bn_bias, KWS_CONV2_OUT_CH * sizeof(float));
    }

    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv3_weights, conv3_params * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv3_bn_scale, KWS_CONV3_OUT_CH * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.conv3_bn_bias, KWS_CONV3_OUT_CH * sizeof(float));
    }

    if (res == FR_OK) {
        res = read_exact(&fil, gModel.fc1_weights, fc1_params * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.fc1_bn_scale, KWS_FC1_OUT_UNITS * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.fc1_bn_bias, KWS_FC1_OUT_UNITS * sizeof(float));
    }

    if (res == FR_OK) {
        res = read_exact(&fil, gModel.fc_out_weights, fc_out_params * sizeof(float));
    }
    if (res == FR_OK) {
        res = read_exact(&fil, gModel.fc_out_bias, gModel.num_classes * sizeof(float));
    }

    f_close(&fil);

    if (res != FR_OK) {
        xil_printf("KWS: failed to read weights (%d)\r\n", (int)res);
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static XStatus allocate_scratch(void)
{
    const size_t input_tensor_size = (size_t)KWS_INPUT_ROWS * KWS_INPUT_COLS;
    const size_t mono_size = KWS_TARGET_SAMPLE_RATE;
    const size_t fft_power_size = KWS_FFT_BINS;

    const size_t conv1_size = (size_t)KWS_CONV1_OUT_CH * KWS_INPUT_ROWS * KWS_INPUT_COLS;
    const size_t pool1_size = (size_t)KWS_CONV1_OUT_CH * KWS_POOL1_ROWS * KWS_POOL1_COLS;
    const size_t conv2_size = (size_t)KWS_CONV2_OUT_CH * KWS_POOL1_ROWS * KWS_POOL1_COLS;
    const size_t pool2_size = (size_t)KWS_CONV2_OUT_CH * KWS_POOL2_ROWS * KWS_POOL2_COLS;
    const size_t conv3_size = (size_t)KWS_CONV3_OUT_CH * KWS_POOL2_ROWS * KWS_POOL2_COLS;
    const size_t pool3_size = (size_t)KWS_CONV3_OUT_CH * KWS_POOL3_ROWS * KWS_POOL3_COLS;
    const size_t gap_size   = (size_t)KWS_CONV3_OUT_CH * KWS_GAP_ROWS * KWS_GAP_COLS;
    const size_t flat_size  = (size_t)KWS_CONV3_OUT_CH * KWS_GAP_ROWS * KWS_GAP_COLS;

    gScratch.input_tensor = (float *)malloc(input_tensor_size * sizeof(float));
    gScratch.mono_buffer  = (float *)malloc(mono_size * sizeof(float));
    gScratch.fft_power    = (float *)malloc(fft_power_size * sizeof(float));
    gScratch.conv1_out    = (float *)malloc(conv1_size * sizeof(float));
    gScratch.pool1_out    = (float *)malloc(pool1_size * sizeof(float));
    gScratch.conv2_out    = (float *)malloc(conv2_size * sizeof(float));
    gScratch.pool2_out    = (float *)malloc(pool2_size * sizeof(float));
    gScratch.conv3_out    = (float *)malloc(conv3_size * sizeof(float));
    gScratch.pool3_out    = (float *)malloc(pool3_size * sizeof(float));
    gScratch.gap_out      = (float *)malloc(gap_size * sizeof(float));
    gScratch.flat         = (float *)malloc(flat_size * sizeof(float));
    gScratch.fc1_out      = (float *)malloc(KWS_FC1_OUT_UNITS * sizeof(float));
    gScratch.logits       = (float *)malloc(gModel.num_classes * sizeof(float));

    if (!gScratch.input_tensor || !gScratch.mono_buffer || !gScratch.fft_power ||
        !gScratch.conv1_out || !gScratch.pool1_out || !gScratch.conv2_out ||
        !gScratch.pool2_out || !gScratch.conv3_out || !gScratch.pool3_out ||
        !gScratch.gap_out || !gScratch.flat || !gScratch.fc1_out || !gScratch.logits) {
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static void init_feature_tables(void)
{
    if (gFeatureTables.initialized) {
        return;
    }

    for (u32 n = 0U; n < KWS_WINDOW_SIZE; ++n) {
        gFeatureTables.hann[n] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)n) / (float)(KWS_WINDOW_SIZE - 1U));
    }

    for (u32 k = 0U; k < KWS_FFT_BINS; ++k) {
        for (u32 n = 0U; n < KWS_WINDOW_SIZE; ++n) {
            float angle = 2.0f * (float)M_PI * (float)k * (float)n / (float)KWS_FFT_SIZE;
            gFeatureTables.cos_table[k * KWS_WINDOW_SIZE + n] = cosf(angle);
            gFeatureTables.sin_table[k * KWS_WINDOW_SIZE + n] = sinf(angle);
        }
    }

    float mel_min = 2595.0f * log10f(1.0f + 0.0f / 700.0f);
    float mel_max = 2595.0f * log10f(1.0f + ((float)KWS_TARGET_SAMPLE_RATE / 2.0f) / 700.0f);
    float mel_step = (mel_max - mel_min) / (float)(KWS_NUM_MELS + 1U);

    float mel_points[KWS_NUM_MELS + 2U];
    u32 bin_points[KWS_NUM_MELS + 2U];

    for (u32 m = 0U; m < KWS_NUM_MELS + 2U; ++m) {
        mel_points[m] = mel_min + mel_step * (float)m;
        float hz = 700.0f * (powf(10.0f, mel_points[m] / 2595.0f) - 1.0f);
        float bin = ((float)(KWS_FFT_SIZE + 1U)) * hz / (float)KWS_TARGET_SAMPLE_RATE;
        if (bin < 0.0f) {
            bin = 0.0f;
        }
        bin_points[m] = (u32)bin;
        if (bin_points[m] > KWS_FFT_BINS - 1U) {
            bin_points[m] = KWS_FFT_BINS - 1U;
        }
    }

    memset(gFeatureTables.mel_filter, 0, sizeof(gFeatureTables.mel_filter));
    for (u32 m = 1U; m <= KWS_NUM_MELS; ++m) {
        u32 start = bin_points[m - 1U];
        u32 center = bin_points[m];
        u32 end = bin_points[m + 1U];
        if (center <= start) {
            center = start + 1U;
        }
        if (end <= center) {
            end = center + 1U;
        }
        for (u32 k = start; k < center; ++k) {
            float denom = (float)(center - start);
            gFeatureTables.mel_filter[(m - 1U) * KWS_FFT_BINS + k] = (denom > 0.0f) ? ((float)(k - start) / denom) : 0.0f;
        }
        for (u32 k = center; k < end; ++k) {
            float denom = (float)(end - center);
            gFeatureTables.mel_filter[(m - 1U) * KWS_FFT_BINS + k] = (denom > 0.0f) ? ((float)(end - k) / denom) : 0.0f;
        }
    }

    gFeatureTables.initialized = 1;
}

static XStatus extract_logmel(const int32_t *source_buffer,
                              size_t frames_per_channel,
                              float *out_tensor)
{
    const size_t decimation = KWS_DECIMATION_FACTOR;
    const size_t mono_samples = KWS_TARGET_SAMPLE_RATE;
    const size_t channels = KWS_SOURCE_CHANNELS;

    if (channels == 0U) {
        return XST_FAILURE;
    }

    for (size_t i = 0U; i < mono_samples; ++i) {
        size_t start = i * decimation;
        double acc = 0.0;
        for (size_t j = 0U; j < decimation; ++j) {
            size_t frame_idx = start + j;
            if (frame_idx >= frames_per_channel) {
                return XST_FAILURE;
            }
            size_t base = frame_idx * channels;
            double frame_sum = 0.0;
            for (size_t ch = 0U; ch < channels; ++ch) {
                frame_sum += (double)source_buffer[base + ch];
            }
            acc += frame_sum / (double)channels;
        }
        double norm = acc / (double)decimation;
        gScratch.mono_buffer[i] = (float)(norm / 2147483648.0);
    }

    for (u32 frame = 0U; frame < KWS_NUM_FRAMES; ++frame) {
        u32 frame_offset = frame * KWS_HOP_LENGTH;
        for (u32 k = 0U; k < KWS_FFT_BINS; ++k) {
            double real = 0.0;
            double imag = 0.0;
            for (u32 n = 0U; n < KWS_WINDOW_SIZE; ++n) {
                u32 idx = frame_offset + n;
                if (idx >= mono_samples) {
                    break;
                }
                double sample = (double)gScratch.mono_buffer[idx] * (double)gFeatureTables.hann[n];
                real += sample * (double)gFeatureTables.cos_table[k * KWS_WINDOW_SIZE + n];
                imag -= sample * (double)gFeatureTables.sin_table[k * KWS_WINDOW_SIZE + n];
            }
            gScratch.fft_power[k] = (float)(real * real + imag * imag);
        }
        for (u32 mel = 0U; mel < KWS_NUM_MELS; ++mel) {
            const float *filter = &gFeatureTables.mel_filter[mel * KWS_FFT_BINS];
            float acc = 0.0f;
            for (u32 k = 0U; k < KWS_FFT_BINS; ++k) {
                acc += filter[k] * gScratch.fft_power[k];
            }
            if (acc < 1e-6f) {
                acc = 1e-6f;
            }
            out_tensor[mel * KWS_NUM_FRAMES + frame] = logf(acc);
        }
    }

    return XST_SUCCESS;
}

static void conv2d_forward(const float *input,
                           u32 in_channels,
                           u32 in_rows,
                           u32 in_cols,
                           const float *weights,
                           const float *bias,
                           const float *bn_scale,
                           const float *bn_bias,
                           u32 out_channels,
                           float *output,
                           KwsActivation activation)
{
    const u32 kernel = 3U;
    const int pad = 1;

    for (u32 oc = 0U; oc < out_channels; ++oc) {
        for (u32 oy = 0U; oy < in_rows; ++oy) {
            for (u32 ox = 0U; ox < in_cols; ++ox) {
                double acc = 0.0;
                for (u32 ic = 0U; ic < in_channels; ++ic) {
                    for (u32 ky = 0U; ky < kernel; ++ky) {
                        int iy = (int)oy + (int)ky - pad;
                        if (iy < 0 || iy >= (int)in_rows) {
                            continue;
                        }
                        for (u32 kx = 0U; kx < kernel; ++kx) {
                            int ix = (int)ox + (int)kx - pad;
                            if (ix < 0 || ix >= (int)in_cols) {
                                continue;
                            }
                            size_t in_index = ((size_t)ic * in_rows + (size_t)iy) * in_cols + (size_t)ix;
                            size_t w_index = ((((size_t)oc * in_channels) + ic) * kernel + ky) * kernel + kx;
                            acc += (double)input[in_index] * (double)weights[w_index];
                        }
                    }
                }
                if (bias != NULL) {
                    acc += bias[oc];
                }
                float bn = (float)(acc * (double)bn_scale[oc] + (double)bn_bias[oc]);
                if (activation == KWS_ACT_RELU) {
                    bn = (bn > 0.0f) ? bn : 0.0f;
                } else if (activation == KWS_ACT_SIGN) {
                    bn = (bn >= 0.0f) ? 1.0f : -1.0f;
                }
                output[((size_t)oc * in_rows + oy) * in_cols + ox] = bn;
            }
        }
    }
}

static void maxpool2d(const float *input,
                      u32 channels,
                      u32 in_rows,
                      u32 in_cols,
                      float *output)
{
    const u32 kernel = 2U;
    const u32 stride = 2U;
    u32 out_rows = POOL_OUT_DIM(in_rows);
    u32 out_cols = POOL_OUT_DIM(in_cols);

    for (u32 c = 0U; c < channels; ++c) {
        for (u32 oy = 0U; oy < out_rows; ++oy) {
            for (u32 ox = 0U; ox < out_cols; ++ox) {
                float max_val = -FLT_MAX;
                for (u32 ky = 0U; ky < kernel; ++ky) {
                    u32 iy = oy * stride + ky;
                    if (iy >= in_rows) {
                        continue;
                    }
                    for (u32 kx = 0U; kx < kernel; ++kx) {
                        u32 ix = ox * stride + kx;
                        if (ix >= in_cols) {
                            continue;
                        }
                        float val = input[((size_t)c * in_rows + iy) * in_cols + ix];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[((size_t)c * out_rows + oy) * out_cols + ox] = max_val;
            }
        }
    }
}

static void adaptive_avg_pool(const float *input,
                              u32 channels,
                              u32 in_rows,
                              u32 in_cols,
                              u32 out_rows,
                              u32 out_cols,
                              float *output)
{
    for (u32 c = 0U; c < channels; ++c) {
        for (u32 oy = 0U; oy < out_rows; ++oy) {
            u32 y_start = (oy * in_rows) / out_rows;
            u32 y_end = ((oy + 1U) * in_rows + out_rows - 1U) / out_rows;
            if (y_end > in_rows) {
                y_end = in_rows;
            }
            for (u32 ox = 0U; ox < out_cols; ++ox) {
                u32 x_start = (ox * in_cols) / out_cols;
                u32 x_end = ((ox + 1U) * in_cols + out_cols - 1U) / out_cols;
                if (x_end > in_cols) {
                    x_end = in_cols;
                }
                float acc = 0.0f;
                u32 count = 0U;
                for (u32 iy = y_start; iy < y_end; ++iy) {
                    for (u32 ix = x_start; ix < x_end; ++ix) {
                        acc += input[((size_t)c * in_rows + iy) * in_cols + ix];
                        ++count;
                    }
                }
                output[((size_t)c * out_rows + oy) * out_cols + ox] = (count > 0U) ? (acc / (float)count) : 0.0f;
            }
        }
    }
}

static void dense_forward(const float *input,
                          u32 in_features,
                          const float *weights,
                          const float *bias,
                          const float *bn_scale,
                          const float *bn_bias,
                          u32 out_features,
                          float *output,
                          KwsActivation activation)
{
    for (u32 o = 0U; o < out_features; ++o) {
        double acc = 0.0;
        for (u32 i = 0U; i < in_features; ++i) {
            acc += (double)input[i] * (double)weights[o * in_features + i];
        }
        if (bias != NULL) {
            acc += bias[o];
        }
        float bn = (float)(acc * (double)bn_scale[o] + (double)bn_bias[o]);
        if (activation == KWS_ACT_RELU) {
            bn = (bn > 0.0f) ? bn : 0.0f;
        } else if (activation == KWS_ACT_SIGN) {
            bn = (bn >= 0.0f) ? 1.0f : -1.0f;
        }
        output[o] = bn;
    }
}

static void run_network(const float *input_tensor, float *logits)
{
    conv2d_forward(input_tensor,
                   KWS_INPUT_DEPTH,
                   KWS_INPUT_ROWS,
                   KWS_INPUT_COLS,
                   gModel.conv1_weights,
                   gModel.conv1_bias,
                   gModel.conv1_bn_scale,
                   gModel.conv1_bn_bias,
                   KWS_CONV1_OUT_CH,
                   gScratch.conv1_out,
                   KWS_ACT_RELU);

    maxpool2d(gScratch.conv1_out,
              KWS_CONV1_OUT_CH,
              KWS_INPUT_ROWS,
              KWS_INPUT_COLS,
              gScratch.pool1_out);

    conv2d_forward(gScratch.pool1_out,
                   KWS_CONV1_OUT_CH,
                   KWS_POOL1_ROWS,
                   KWS_POOL1_COLS,
                   gModel.conv2_weights,
                   NULL,
                   gModel.conv2_bn_scale,
                   gModel.conv2_bn_bias,
                   KWS_CONV2_OUT_CH,
                   gScratch.conv2_out,
                   KWS_ACT_SIGN);

    maxpool2d(gScratch.conv2_out,
              KWS_CONV2_OUT_CH,
              KWS_POOL1_ROWS,
              KWS_POOL1_COLS,
              gScratch.pool2_out);

    conv2d_forward(gScratch.pool2_out,
                   KWS_CONV2_OUT_CH,
                   KWS_POOL2_ROWS,
                   KWS_POOL2_COLS,
                   gModel.conv3_weights,
                   NULL,
                   gModel.conv3_bn_scale,
                   gModel.conv3_bn_bias,
                   KWS_CONV3_OUT_CH,
                   gScratch.conv3_out,
                   KWS_ACT_SIGN);

    maxpool2d(gScratch.conv3_out,
              KWS_CONV3_OUT_CH,
              KWS_POOL2_ROWS,
              KWS_POOL2_COLS,
              gScratch.pool3_out);

    adaptive_avg_pool(gScratch.pool3_out,
                      KWS_CONV3_OUT_CH,
                      KWS_POOL3_ROWS,
                      KWS_POOL3_COLS,
                      KWS_GAP_ROWS,
                      KWS_GAP_COLS,
                      gScratch.gap_out);

    memcpy(gScratch.flat,
           gScratch.gap_out,
           (size_t)KWS_CONV3_OUT_CH * KWS_GAP_ROWS * KWS_GAP_COLS * sizeof(float));

    dense_forward(gScratch.flat,
                  KWS_CONV3_OUT_CH * KWS_GAP_ROWS * KWS_GAP_COLS,
                  gModel.fc1_weights,
                  NULL,
                  gModel.fc1_bn_scale,
                  gModel.fc1_bn_bias,
                  KWS_FC1_OUT_UNITS,
                  gScratch.fc1_out,
                  KWS_ACT_SIGN);

    for (u32 o = 0U; o < gModel.num_classes; ++o) {
        double acc = 0.0;
        for (u32 i = 0U; i < KWS_FC1_OUT_UNITS; ++i) {
            acc += (double)gScratch.fc1_out[i] * (double)gModel.fc_out_weights[o * KWS_FC1_OUT_UNITS + i];
        }
        acc += gModel.fc_out_bias[o];
        logits[o] = (float)acc;
    }
}
