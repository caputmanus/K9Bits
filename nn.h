#ifndef NN_PREDICT_H
#define NN_PREDICT_H
#include <stdint.h>

#define NN_IMG_W 144
#define NN_IMG_H 256

#ifdef __cplusplus
extern "C" {
#endif

/* img: NN_IMG_W*NN_IMG_H байт яркости 0..255, построчно (row-major).
   Возвращает 64-битное число. Без malloc, без libc, потокобезопасно. */
uint64_t nn_predict(const uint8_t *img);

#ifdef __cplusplus
}
#endif
#endif
