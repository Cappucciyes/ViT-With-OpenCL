#include "Network.h"

#ifndef _ViT_seq_H
#define _ViT_seq_H
//void ViT_seq(ImageData *image, Network *networks, float **prb);
void linear_layer_seq(float* input, float* output, int tokens, int in_features, int out_features, Network weight, Network bias);
void multihead_attn_seq(float* input, float* output, Network in_weight, Network in_bias, Network out_weight, Network out_bias);
#endif #pragma once
