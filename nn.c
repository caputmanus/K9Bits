/* Сеть: 144x256 -> пулинг 8x8 -> 576 -> dense(NN_HID)+ReLU -> dense(64) -> знак = бит */
#include "nn.h"
#include "weights.h"   /* генерится train.py: NN_HID, nn_w1, nn_b1, nn_w2, nn_b2 */

#define BLK 8
#define GW  (NN_IMG_W / BLK)
#define GH  (NN_IMG_H / BLK)
#define NIN (GW * GH)

uint64_t nn_predict(const uint8_t *img)
{
    uint8_t p[NIN];
    float h[NN_HID];
    uint64_t out = 0;
    int x, y, i, j, k;

    for (y = 0; y < GH; y++) {
        for (x = 0; x < GW; x++) {
            uint32_t s = 0;
            const uint8_t *r = img + (y * BLK) * NN_IMG_W + x * BLK;
            for (j = 0; j < BLK; j++, r += NN_IMG_W)
                for (i = 0; i < BLK; i++) s += r[i];
            p[y * GW + x] = (uint8_t)(s / (BLK * BLK));
        }
    }

    for (j = 0; j < NN_HID; j++) {
        const float *w = nn_w1 + j * NIN;
        float a = 0.0f;
        for (i = 0; i < NIN; i++) a += w[i] * (float)p[i];
        a = a * (1.0f / 255.0f) + nn_b1[j];
        h[j] = a > 0.0f ? a : 0.0f;
    }

    for (k = 0; k < 64; k++) {
        const float *w = nn_w2 + k * NN_HID;
        float a = nn_b2[k];
        for (j = 0; j < NN_HID; j++) a += w[j] * h[j];
        if (a > 0.0f) out |= (uint64_t)1 << k;
    }
    return out;
}
