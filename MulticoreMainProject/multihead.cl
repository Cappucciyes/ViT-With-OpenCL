#define TILESIZE 8

__kernel void QKV(
    __global float* input,
    __global float* weight,
    __global float* outputBundle,
    __global float* bias,
    int rowSize,
    int embed_dim
) {
    int tilei = get_group_id(0);
    int tilej = get_group_id(1);
    int k = get_global_id(2);

    int li = get_local_id(0);
    int lj = get_local_id(1);

    int gi = tilei * TILESIZE + li;
    int gj = tilej * TILESIZE + lj;
    //if (gi >= rowSize|| gj >= embed_dim) return;
    int outputOffset = k * rowSize * embed_dim;
    int currentOffset = k * embed_dim;
    __local float inputTile[TILESIZE][TILESIZE];
    __local float weightTile[TILESIZE][TILESIZE];

    float result = 0;
    // 타일 수 = 전체 크기를 타일 크기로 나눈 값의 올림.
    int tileCount = (embed_dim + TILESIZE - 1) / TILESIZE;
    for (int tile = 0; tile < tileCount; tile++) {
        int tileStart = tile * TILESIZE;

        // Load input tile: input[gi][tileStart + lj]
        int inputCol = tileStart + lj;
        if (gi < rowSize && inputCol < embed_dim) {
            inputTile[li][lj] = input[gi * embed_dim + inputCol];
        }
        else {
            inputTile[li][lj] = 0.0f;
        }

        int weightCol = tileStart + li;
        if (gj < embed_dim && weightCol < embed_dim) {
            weightTile[li][lj] = weight[(gj + currentOffset) * embed_dim + weightCol];
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

    if (gi < rowSize && gj < embed_dim) {
        result += bias[currentOffset + gj];
        outputBundle[outputOffset + gi * embed_dim + gj] = result;
    }
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
    if (t >= tokens || h >= num_heads) return;
    
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