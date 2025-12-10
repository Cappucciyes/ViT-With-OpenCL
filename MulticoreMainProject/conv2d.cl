__kernel void conv2d_kernel(
    __global float* input,
    __global float* output,
    __global float* weight,
    __global float* bias,
    int img_size,
    int patch_size,
    int in_chans,
    int embed_dim
) {
    int oc = get_global_id(0);  // output channel
    int oh = get_global_id(1);  // output height
    int ow = get_global_id(2);  // output width

    int output_size = img_size / patch_size;

    if (oc >= embed_dim || oh >= output_size || ow >= output_size)
        return;

    float sum = bias[oc];

    for (int ic = 0; ic < in_chans; ++ic) {
        for (int kh = 0; kh < patch_size; ++kh) {
            for (int kw = 0; kw < patch_size; ++kw) {
                int ih = oh * patch_size + kh;
                int iw = ow * patch_size + kw;
                int input_idx = (ic * img_size + ih) * img_size + iw;
                int kernel_idx = ((oc * in_chans + ic) * patch_size + kh) * patch_size + kw;

                sum += input[input_idx] * weight[kernel_idx];
            }
        }
    }

    output[(oc * output_size + oh) * output_size + ow] = sum;
}


__kernel void postprocess(
    __global const float* conv_out,      // [embed_dim, output_size, output_size]
    __global const float* class_token,   // [embed_dim]
    __global const float* pos_embed,     // [(num_patches+1)*embed_dim]
    __global float* final_tokens,         // [(num_patches+1)*embed_dim]
    int img_size,
    int patch_size,
    int embed_dim
)
{
    const int output_size = img_size / patch_size;
    const int num_patches = output_size * output_size;
    const int total_tokens = num_patches + 1;

    int gid = get_global_id(0);
    int total = total_tokens * embed_dim;
    if (gid >= total) return;

    int token_idx = gid / embed_dim;
    int oc = gid % embed_dim;

    float val;

    if (token_idx == 0) {
        // class token
        val = class_token[oc];
    }
    else {
        // flatten + transpose from conv_out
        int patch_idx = token_idx - 1;
        int oh = patch_idx / output_size;
        int ow = patch_idx % output_size;

        int conv_idx = (oc * output_size + oh) * output_size + ow;
        val = conv_out[conv_idx];
    }

    // positional embedding add
    val += pos_embed[gid];

    final_tokens[gid] = val;
}
