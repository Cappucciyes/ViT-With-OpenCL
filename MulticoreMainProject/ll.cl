float gelu(float x) {
    return 0.5f * x * (1.0f + erf(x / sqrt(2.0f)));
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


    float result = bias[j];

    for (int k = 0; k < featureCount; k ++) {
        result += input[i*featureCount + k] * weight[j * featureCount + k];
    }

    if (doGelu == 1)
		output[i * outCount + j] = gelu(result);
    else
		output[i * outCount + j] = result;
}
