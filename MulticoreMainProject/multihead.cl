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

__kernel void scaledDot(
    __global float* Q,
    __global float* K,
    __global float* score,
    int tokens,
    int embed_dim,
    int head_dim
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int currentOffset = tokens * tokens * k;
    int qkOffset = k * head_dim;

    float result = 0;
    for (int m = 0; m < head_dim; m++) {
        result += Q[qkOffset + i * embed_dim + m] * K[qkOffset  + j * embed_dim + m];
    }

    score[currentOffset + i * tokens + j] = result / sqrt((float)head_dim);
    //if (i==0 && j==0 && k==0) printf("score[0,0,0]=%f\n", score[currentOffset + i * tokens + j]);
    //if (i==tokens-1 && j==tokens-1 && k==get_global_size(2) - 1) printf("score[x,y,z]=%f\n", score[currentOffset + i * tokens + j]);
}

__kernel void softMax(
    __global float *score,
    int head_dim,
    int tokens)
{
    int i = get_global_id(0);
    int j = get_local_id(1);
    int k = get_global_id(2);
    int currentOffset = tokens * tokens * k + i * tokens;

    __local float cache[197];

    cache[j] = score[currentOffset + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    float max_val = cache[j];
    for (int m = 0; m < tokens; m++)
    {
        if (cache[m] > max_val)
            max_val = cache[m];
    }

    cache[j] = exp(cache[j] - max_val);
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum_exp = 0.0f;
    for (int m = 0; m < tokens; m++)
        sum_exp += cache[m];

    score[currentOffset + j] = cache[j] / sum_exp;
    // if (i == 0 && j == 0 && k == 0) printf("softmax[0,0,0]=%f\n", score[currentOffset + j]);
    // if (i == tokens - 1 && j == tokens - 1 && k == get_global_size(2) - 1) printf("softmax[x,y,z]=%f\n", score[currentOffset + j]);
}


// global memory version
// __kernel void softMax(__global float *score, int head_dim, int tokens)
// {
//     int i = get_global_id(0);
//     int j = get_global_id(1);
//     int k = get_global_id(2);
//     int currentOffset = tokens * tokens * k;
//     float max_val = score[currentOffset + i * tokens];
//     for (int m = 1; m < tokens; m++)
//     {
//         if (score[currentOffset + i * tokens + m] > max_val)
//             max_val = score[currentOffset + i * tokens + m];
//     }
//     barrier(CLK_GLOBAL_MEM_FENCE);
//     score[currentOffset + i * tokens + j] = exp(score[currentOffset + i * tokens + j] - max_val);
//     float sum_exp = 0.0f;
//     barrier(CLK_GLOBAL_MEM_FENCE);
//     for (int m = 0; m < tokens; m++)
//     {
//         sum_exp += score[currentOffset + i * tokens + m];
//     }
//     score[currentOffset + i * tokens + j] /= sum_exp;
//     // if (i==0 && j==0 && k==0) printf("softmax[0,0,0]=%f\n", score[currentOffset + i * tokens + j]);
//     // if (i==tokens-1 && j==tokens-1 && k==get_global_size(2) - 1) printf("softmax[x,y,z]=%f\n", score[currentOffset + i * tokens + j]);
// }

__kernel void scoreV(
    __global float* score,
    __global float* v,
    __global float* output,
    int tokens,
    int embed_dim,
    const int head_dim
    ) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int head_offset = k * head_dim;
    int out_index = i * embed_dim + head_offset + j ;

    int score_base = k * tokens * tokens + i * tokens;

    float result = 0;
    for (int m = 0; m < tokens; m++) {
        result += score[score_base + m] * v[m * embed_dim + head_offset + j];
    }

    output[out_index] = result;
}       