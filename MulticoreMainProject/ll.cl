#define TILESIZE 16

float gelu(float x) {
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
}

__kernel void linear_layer(
    __global float* output,
    __global float* weight,
    __global float* input,
    __global float* bias,
    int outCount,
    int featureCount,
    int doGelu
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int li = get_local_id(0);
    int lj = get_local_id(1);
    
    if (i >= get_global_size(0) || j >= outCount) return;
    
	__local float inputTile[TILESIZE][TILESIZE];
    __local float weightTile[TILESIZE][TILESIZE];

    float result = 0;
    // 타일 수 = 전체 크기를 타일 크기로 나눈 값의 올림.
    int tileCount= (featureCount + TILESIZE - 1) / TILESIZE;

    for (int tile = 0; tile < tileCount; tile++) {
        int tileStart = tile * TILESIZE;
        
        int inputCol= tileStart+ lj;
        if (inputCol < featureCount && li < TILESIZE)
            inputTile[li][lj] = input[i * featureCount + inputCol];
        else
            inputTile[li][lj] = 0;
        
        int weightCol= tileStart + li;
        if (weightCol < featureCount && lj < TILESIZE) 
            weightTile[lj][li] = weight[j * featureCount + weightCol];
        else
            weightTile[lj][li] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        int tileWidth = min(TILESIZE, featureCount - tileStart);
        for (int k = 0; k < tileWidth; k++) {
            result += inputTile[li][k] * weightTile[lj][k];
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result += bias[j];
    if (doGelu == 1)
		output[i * outCount + j] = gelu(result);
    else
		output[i * outCount + j] = result;
}
