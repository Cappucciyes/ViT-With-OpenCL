__kernel void QKV(
    __global float* input,
    __global float* weight,
    __global float* outputBundle,
    __global float* bias,
    __global int * offset,
    int rowSize,
    int middleSize,
    int colSize 
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int outputOffset = k * rowSize * colSize;
    int currentOffset = offset[k];

    float result = bias[j + currentOffset];

    for (int k = 0; k < middleSize; k ++) {
        result += input[i*middleSize + k] * weight[(j + currentOffset) * middleSize + k];
    }
    
	outputBundle[outputOffset + i * colSize + j] = result;
}