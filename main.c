/* Linux-тест: ./nn_test image.raw  (36864 байта) -> число */
#include <stdio.h>
#include <inttypes.h>
#include "nn.h"

int main(int argc, char **argv)
{
    static uint8_t img[NN_IMG_W * NN_IMG_H];
    FILE *f = argc > 1 ? fopen(argv[1], "rb") : stdin;
    if (!f || fread(img, 1, sizeof img, f) != sizeof img) {
        fprintf(stderr, "need %d raw bytes\n", (int)sizeof img);
        return 1;
    }
    printf("%" PRIu64 "\n", nn_predict(img));
    return 0;
}
