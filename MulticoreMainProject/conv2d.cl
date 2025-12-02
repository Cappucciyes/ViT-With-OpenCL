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