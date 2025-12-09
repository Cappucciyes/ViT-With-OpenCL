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
    int tileCount= (colA + TILESIZE - 1) / TILESIZE;
    for (int tile = 0; tile < tileCount; tile++) {
        int tileStart = tile * TILESIZE;
        
        int inputCol= tileStart+ lj;
        if (inputCol < colA && gi < rowA)
            inputTile[li][lj] = input[gi * colA + inputCol];
        else
            inputTile[li][lj] = 0;
        
        int weightCol= tileStart + lj;
        if (weightCol < colB && gj < colB) 
            weightTile[li][lj] = weight[gj * colA+ weightCol];
        else
            weightTile[li][lj] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILESIZE; k++) {
            result += inputTile[li][k] * weightTile[k][lj];
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result += bias[gj];
    if (gi < rowA && gj < colB) {
        if (doGelu == 1)
			output[gi * colB + gj] = gelu(result);
		else
			output[gi * colB + gj] = result;
    }
    
}
