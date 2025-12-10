#define eps 1e-6

__kernel void layerNorm(
	__global float* input,
	__global float* weight,
	__global float* bias,
	__global float* output,
	int tokens,
	int embed_dim
) {
	int t = get_global_id(0);
	int d = get_global_id(1);
	int localSize = get_local_size(1);
	int j = get_local_id(1);

	__local float localSum[256];
	__local float localSumSq[256];
	__local float mean;
	__local float inv_std;

	float sum = 0;
    float sum_sq = 0;
    
    for (int i = j; i < embed_dim; i += localSize) {
        float val = input[t * embed_dim + i];
        sum += val;
        sum_sq += val * val;
    }
    
    localSum[j] = sum;
	localSumSq[j] = sum_sq;	
	barrier(CLK_LOCAL_MEM_FENCE);

	// reduction
	for (int p = localSize / 2; p >= 1; p >>= 1) {
		if (j < p) {
			localSum[j] = localSum[j] + localSum[j + p];
			localSumSq[j] = localSumSq[j] + localSumSq[j + p];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (j == 0) {
		mean = localSum[0] / embed_dim;
		float var = (localSumSq[0] / embed_dim) - (mean * mean);
		inv_std = 1.0 / sqrt(var + eps);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int index = t * embed_dim + d;
	float normalized = (input[index] - mean) * inv_std;
	output[index] = normalized * weight[d] + bias[d];
}

__kernel void encoderResidual(
	__global float* input,
	__global float* toAdd,
	__global float* output
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int colSize = get_global_size(1);

	output[i * colSize + j] =input[i * colSize + j] +  toAdd[i * colSize + j];
}
