__kernel void QKV(
    __global float* input,
    __global float* weight,
    __global float* outputBundle,
    __global float* bias,
    int rowSize,
    int embed_dim
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    if (i >= rowSize) return;
    int outputOffset = k * rowSize * embed_dim;
    int currentOffset = k * embed_dim;

    int totalWeightCols = 3 * embed_dim;

    float result = bias[j + currentOffset];
    for (int m = 0; m < embed_dim; m ++) {
        result += input[i*embed_dim + m] * weight[(j + currentOffset) * embed_dim + m];
    }
    
	outputBundle[outputOffset + i * embed_dim + j] = result;
}


__kernel void QKV_TO_SCOREV(
    __global float* QKV,
    __global float* output,
    int tokens,
    int embed_dim,
    int head_dim,
    int num_heads
) {
    int localIndex = get_local_id(0);
    int t = get_global_id(1); 
    int h = get_global_id(2);
    
    int kOffset = tokens * embed_dim;
    int vOffset = tokens * embed_dim * 2;
    int head_offset = h * head_dim;
    
    __local float score_local[256];
    __local float maxReduce[256];
    __local float toReduce[256];

    
    // Compute Q·K scores for this token t and head h
    if (localIndex < tokens) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += QKV[t * embed_dim + head_offset + d] * 
                   QKV[kOffset + localIndex * embed_dim + head_offset + d];
        }
        score_local[localIndex] = dot / sqrt((float)head_dim);
        maxReduce[localIndex] = score_local[localIndex];
    } else {
        score_local[localIndex] = 0;
        maxReduce[localIndex] = -FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Find max for softmax stability
    for (int p = 128; p >= 1; p >>= 1) { 
        if (localIndex < p) {
            maxReduce[localIndex] = max(maxReduce[localIndex], maxReduce[localIndex + p]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float maxVal = maxReduce[0];
    
    // Compute exp(score - max)
    if (localIndex < tokens) {
        score_local[localIndex] = exp(score_local[localIndex] - maxVal);
    } else {
        score_local[localIndex] = 0.0f;
    }
    toReduce[localIndex] = score_local[localIndex];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Sum reduction
    for (int p = 128; p >= 1; p >>= 1) {
        if (localIndex < p) 
            toReduce[localIndex] += toReduce[localIndex + p];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float acc = toReduce[0];
    if (localIndex < tokens)
        score_local[localIndex] /= acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute attention output: score · V
    if (localIndex < head_dim) {
        float sum = 0.0f;
        for (int i = 0; i < tokens; i++) {
            sum += score_local[i] * QKV[vOffset + i * embed_dim + head_offset + localIndex];
        }
        output[t * embed_dim + head_offset + localIndex] = sum;
    }
}