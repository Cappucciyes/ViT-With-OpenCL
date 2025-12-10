__kernel void softMax(
	__global float * input,	
	__global float * output,	
	int length
) {
	int localIndex = get_global_id(0);
	__local float score_local[1024];
	__local float localSum[1024];
    __local float maxReduce[1024];

	if (localIndex < length) {
		score_local[localIndex] = input[localIndex];
		maxReduce[localIndex] = score_local[localIndex];
	} else {
		score_local[localIndex] = 0;
		maxReduce[localIndex] = -FLT_MAX;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = 1024 / 2; p >= 1; p >>= 1) { 
		if (localIndex < p) {
			maxReduce[localIndex] = max(maxReduce[localIndex], maxReduce[localIndex + p]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	float maxVal = maxReduce[0];

	// Compute exp(score - max)
	float sum = 0;
	if (localIndex < length ) {
		score_local[localIndex] = exp(score_local[localIndex] - maxVal);
	} else {
		score_local[localIndex] = 0.0f;
	}
	localSum[localIndex] = score_local[localIndex];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Sum reduction
	for (int p = 1024 / 2; p >= 1; p >>= 1) {
		if (localIndex < p) 
			localSum[localIndex] += localSum[localIndex + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	float acc = localSum[0];
	if (localIndex < length)
		score_local[localIndex] /= acc;
	barrier(CLK_LOCAL_MEM_FENCE);

	output[localIndex] = score_local[localIndex];
}
