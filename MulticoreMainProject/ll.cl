#define TILESIZE 16

float gelu(float x) {
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
}

__kernel void linear_layer(
    __global float* output,
    __global float* weight,
    __global float* input,
    __global float* bias,
    int rowA,
    int colA,
    int colB,
    int doGelu
) {
    int tilei = get_group_id(0);
    int tilej = get_group_id(1);

    int li = get_local_id(0);
    int lj = get_local_id(1);

    int gi = tilei * TILESIZE + li;
    int gj = tilej * TILESIZE + lj;
    //if (gi >= rowA|| gj >= colB) return;
    __local float inputTile[TILESIZE][TILESIZE];
    __local float weightTile[TILESIZE][TILESIZE];

    float result = 0;
    // 타일 수 = 전체 크기를 타일 크기로 나눈 값의 올림.
    int tileCount = (colA + TILESIZE - 1) / TILESIZE;
    for (int tile = 0; tile < tileCount; tile++) {
        int tileStart = tile * TILESIZE;

        // Load input tile: input[gi][tileStart + lj]
        int inputCol = tileStart + lj;
        if (gi < rowA && inputCol < colA) {
            inputTile[li][lj] = input[gi * colA + inputCol];
        }
        else {
            inputTile[li][lj] = 0.0f;
        }

        // Load weight tile: weight[gj][tileStart + li]
        // CRITICAL: We're computing output[gi][gj] = sum_k input[gi][k] * weight[gj][k]
        // So we need weight[gj][tileStart + li], but stored as weightTile[li][lj]
        int weightCol = tileStart + li;
        if (gj < colB && weightCol < colA) {
            weightTile[li][lj] = weight[gj * colA + weightCol];
        }
        else {
            weightTile[li][lj] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product
        for (int k = 0; k < TILESIZE; k++) {
            result += inputTile[li][k] * weightTile[k][lj];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (gi < rowA && gj < colB) {
        result += bias[gj];
        if (doGelu == 1)
            output[gi * colB + gj] = gelu(result);
        else
            output[gi * colB + gj] = result;
    }

    /*int i = get_global_id(0);
    int j = get_global_id(1);


    float result = bias[j];

    for (int k = 0; k < colA; k++) {
        result += input[i * colA + k] * weight[j * colA + k];
    }

    if (doGelu == 1)
        output[i * colB + j] = gelu(result);
    else
        output[i * colB + j] = result;*/
}