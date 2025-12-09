#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <stdbool.h>
#include "Network.h"
#include "ViT_seq.h"
#include "kernelHandler.h"
#define img_size 224
#define patch_size 16
#define in_chans 3
#define num_classes 1000
#define embed_dim 768
#define depth 12
#define num_heads 12
#define mlp_ratio 4.0
#define dropout 0.0
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6

// Global Variable
cl_platform_id PLATFORM;
cl_device_id DEVICE;
cl_context CONTEXT;

cl_program MULTIHEAD_PROGRAM;
cl_kernel QKV_KERNEL;
cl_kernel QKV_TO_SCOREV_KERNEL;
cl_command_queue MULTIHEAD_QUEUE;

cl_program LL_PROGRAM;
cl_kernel LL_KERNEL;
cl_command_queue LL_QUEUE;

cl_program CONV2D_PROGRAM;
cl_kernel CONV2D_KERNEL;
cl_command_queue CONV2D_QUEUE;

cl_program LAYERNORM_PROGRAM;
cl_kernel LAYERNORM_KERNEL;
cl_command_queue LAYERNORM_QUEUE;

cl_ulong CONV_WRITE_TIME;
cl_ulong CONV_EXEC_TIME;
cl_ulong CONV_READ_TIME;
int CONV_EXEC_COUNT;

cl_ulong QKV_WRITE_TIME;
cl_ulong QKV_EXEC_TIME;
cl_ulong QKV_TO_SCORE_EXEC_TIME;
cl_ulong QKV_FINAL_LL_EXEC_TIME;
cl_ulong QKV_READ_TIME;
int QKV_EXEC_COUNT;

cl_ulong LL_WRITE_TIME;
cl_ulong LL_EXEC_TIME;
cl_ulong LL_READ_TIME;;
int LL_EXEC_COUNT;

cl_ulong LAYERNORM_WRITE_TIME;
cl_ulong LAYERNORM_EXEC_TIME;
cl_ulong LAYERNORM_READ_TIME;
int LAYERNORM_EXEC_COUNT;

int encoderCount;
////////////////////////////////////// utils function //////////////////////////////////////
void printSpec();
void profileEvents(cl_event* events, int eventCount, cl_ulong* timeGlobalVariable);

// testing 
void printEventProfile();
void test_linear_layer();
bool findNaN(float* a, int tokens, int embedings);
void test_linear_layer_big();
////////////////////////////////////// ViT function //////////////////////////////////////

void Conv2d(float* input, float* output, Network weight, Network bias)
{
    int output_size = img_size / patch_size;
    cl_int err;
    cl_event writeEvent[3];
    cl_event execEvent;
    cl_event readEvent;

    cl_mem inputBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY, sizeof(float) * img_size * img_size * in_chans, NULL, &err);
    CHECK_ERROR(err);
    cl_mem outputBuf = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY, sizeof(float) * embed_dim * output_size * output_size, NULL, &err);
    CHECK_ERROR(err);
    cl_mem weightBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY, sizeof(float) * weight.size, NULL, &err);
    CHECK_ERROR(err);
    cl_mem biasBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY, sizeof(float) * bias.size, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(CONV2D_QUEUE, inputBuf, CL_FALSE, 0, sizeof(float) * img_size * img_size * in_chans, input, 0, NULL, writeEvent);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(CONV2D_QUEUE, weightBuf, CL_FALSE, 0, sizeof(float) * weight.size, weight.data, 0, NULL, writeEvent + 1);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(CONV2D_QUEUE, biasBuf, CL_FALSE, 0, sizeof(float) * bias.size, bias.data, 0, NULL, writeEvent + 2);
    CHECK_ERROR(err);

    err = clSetKernelArg(CONV2D_KERNEL, 0, sizeof(cl_mem), &inputBuf);
    CHECK_ERROR(err);
    err = clSetKernelArg(CONV2D_KERNEL, 1, sizeof(cl_mem), &outputBuf);
    CHECK_ERROR(err);
    err = clSetKernelArg(CONV2D_KERNEL, 2, sizeof(cl_mem), &weightBuf);
    CHECK_ERROR(err);
    err = clSetKernelArg(CONV2D_KERNEL, 3, sizeof(cl_mem), &biasBuf);
    CHECK_ERROR(err);
    cl_int imgSize = img_size;
    err = clSetKernelArg(CONV2D_KERNEL, 4, sizeof(cl_int), &imgSize);
    CHECK_ERROR(err);
    cl_int patchSize = patch_size;
    err = clSetKernelArg(CONV2D_KERNEL, 5, sizeof(cl_int), &patchSize);
    CHECK_ERROR(err);
    cl_int inChans = in_chans;
    err = clSetKernelArg(CONV2D_KERNEL, 6, sizeof(cl_int), &inChans);
    CHECK_ERROR(err);
    cl_int embedDim = embed_dim;
    err = clSetKernelArg(CONV2D_KERNEL, 7, sizeof(cl_int), &embedDim);
    CHECK_ERROR(err);

    size_t global_size[3] = { embed_dim, output_size, output_size };
    size_t local_size[3] = { 1, 1, 1 };

    err = clEnqueueNDRangeKernel(CONV2D_QUEUE, CONV2D_KERNEL, 3, NULL,
        global_size, local_size, 3, writeEvent, &execEvent);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(CONV2D_QUEUE, outputBuf, CL_FALSE, 0,
        sizeof(float) * embed_dim * output_size * output_size,
        output, 1, &execEvent, &readEvent);
    CHECK_ERROR(err);
    
    clFinish(CONV2D_QUEUE);
    printf("asdf\n");
    profileEvents(writeEvent, 3, &CONV_WRITE_TIME);
    printf("asdfasdf\n");
    profileEvents(&execEvent, 1, &CONV_EXEC_TIME);
    profileEvents(&readEvent, 1, &CONV_READ_TIME);
    CONV_EXEC_COUNT += 1;

    clReleaseMemObject(inputBuf);
    clReleaseMemObject(outputBuf);
    clReleaseMemObject(weightBuf);
    clReleaseMemObject(biasBuf);
}

void flatten_transpose(float *input, float *output)
{
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 각 공간 위치(oh, ow)를 하나의 패치로 취급하여 patch index 계산
    for (int oh = 0; oh < output_size; oh++)
    {
        for (int ow = 0; ow < output_size; ow++)
        {
            int patch_idx = oh * output_size + ow;
            for (int oc = 0; oc < embed_dim; oc++)
            {
                // 기존 입력은 (oc, oh, ow)
                int idx_input = (oc * output_size + oh) * output_size + ow;
                // 원하는 출력은 (patch_idx, oc)
                int idx_output = patch_idx * embed_dim + oc;
                output[idx_output] = input[idx_input];
                // printf("%f ",output[idx_output]);
            }
        }
    }
}

void class_token(float *patch_tokens, float *final_tokens, Network cls_tk)
{
    // 이미지의 패치 수 계산: output_size = img_size / patch_size, num_patches = output_size^2
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 1. 첫 번째 토큰에 class token 복사 (networks[0].data에 저장됨, embed_dim 길이)
    for (int j = 0; j < embed_dim; j++)
    {
        final_tokens[j] = cls_tk.data[j];
    }

    // 2. 이후 patch_tokens를 이어붙임
    // final_tokens의 인덱스 embed_dim부터, patch_tokens 전체(embed_dim * num_patches) 복사
    memcpy(final_tokens + embed_dim, patch_tokens, sizeof(float) * embed_dim * num_patches);
}

void pos_emb(float *input, float *output, Network pos_emb)
{
    // output_size: 한 변의 패치 수, num_patches: 전체 패치 수, total_tokens: class token + patch tokens
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;
    int total_tokens = num_patches + 1;
    int total_elements = total_tokens * embed_dim;
    for (int i = 0; i < total_elements; i++)
    {
        output[i] = input[i] + pos_emb.data[i];
    }
}

void layer_norm(float *input, float *output, Network weight, Network bias)
{
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    cl_int err;
    cl_event writeEvent[3];
    cl_event execEvent;
    cl_event readEvent;
    cl_mem inputBuf= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens * embed_dim), NULL, &err);
	CHECK_ERROR(err);
	cl_mem weightBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
	CHECK_ERROR(err);
	cl_mem biasBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
	CHECK_ERROR(err);
    cl_mem outputbuf= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens * embed_dim), NULL, &err);
	CHECK_ERROR(err);
    
	err = clEnqueueWriteBuffer(LAYERNORM_QUEUE, inputBuf, CL_FALSE, 0, sizeof(float) * (tokens * embed_dim), input, 0, NULL, writeEvent);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(LAYERNORM_QUEUE, weightBuf, CL_FALSE, 0, sizeof(float) * embed_dim, weight.data , 0, NULL, writeEvent + 1);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(LAYERNORM_QUEUE, biasBuf, CL_FALSE, 0, sizeof(float) * embed_dim, bias.data, 0, NULL, writeEvent + 2);
	CHECK_ERROR(err);
    
    err = clSetKernelArg(LAYERNORM_KERNEL, 0, sizeof(cl_mem), &inputBuf);
	CHECK_ERROR(err);
    err = clSetKernelArg(LAYERNORM_KERNEL, 1, sizeof(cl_mem), &weightBuf);
	CHECK_ERROR(err);
	err = clSetKernelArg(LAYERNORM_KERNEL, 2, sizeof(cl_mem), &biasBuf);
	CHECK_ERROR(err);
	err = clSetKernelArg(LAYERNORM_KERNEL, 3, sizeof(cl_mem), &outputbuf);
	CHECK_ERROR(err);
	cl_int clTokens = tokens;
	err = clSetKernelArg(LAYERNORM_KERNEL, 4, sizeof(cl_int), &clTokens);
	CHECK_ERROR(err);
	cl_int clEmbedDim= embed_dim;
	err = clSetKernelArg(LAYERNORM_KERNEL, 5, sizeof(cl_int), &clEmbedDim);
	CHECK_ERROR(err);
	
    size_t globalSize[2] = {tokens, embed_dim};
    size_t localSize[2] = {1, 256};
    err = clEnqueueNDRangeKernel(
		LAYERNORM_QUEUE,
		LAYERNORM_KERNEL,
		2,
		NULL,
		globalSize,
		localSize,
		3,
		writeEvent,
		&execEvent);
	CHECK_ERROR(err);
    err = clEnqueueReadBuffer(LAYERNORM_QUEUE, outputbuf, CL_FALSE, 0, sizeof(float) * tokens * embed_dim, output, 1, &execEvent, &readEvent);
    CHECK_ERROR(err);
    clFinish(LAYERNORM_QUEUE);

    profileEvents(writeEvent, 3, &LAYERNORM_WRITE_TIME);
	profileEvents(&execEvent, 1, &LAYERNORM_EXEC_TIME);
	profileEvents(&readEvent, 1, &LAYERNORM_READ_TIME);
	LAYERNORM_EXEC_COUNT += 1;

    clReleaseMemObject(inputBuf);
    clReleaseMemObject(weightBuf);
    clReleaseMemObject(biasBuf);
    clReleaseMemObject(outputbuf);
    //printf("Layer Normalization: %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);

}

void multihead_attn(float *input, float *output,
                    Network in_weight, Network in_bias, Network out_weight, Network out_bias)
{
    int head_dim = embed_dim / num_heads, tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    /*Allocate Q, K, V : tokens * dim*/
    int Q_dim = 0, K_dim = embed_dim, V_dim = embed_dim * 2;
    cl_int err;
    cl_event writeEvent[5];
    cl_event execEvent[3];
    cl_event readEvent;

	//clock_t startTime = clock();
    cl_mem inputBuf= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens * embed_dim), NULL, &err);
	CHECK_ERROR(err);
    cl_mem qkvBuf= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens * embed_dim * 3), NULL, &err);
	CHECK_ERROR(err);
	cl_mem weightBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (embed_dim* embed_dim* 3), NULL, &err);
	CHECK_ERROR(err);
	cl_mem biasBuf= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (embed_dim * 3), NULL, &err);
	CHECK_ERROR(err);
    
    err = clEnqueueWriteBuffer(MULTIHEAD_QUEUE, inputBuf, CL_FALSE, 0, sizeof(float) * (tokens * embed_dim), input, 0, NULL, writeEvent);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(MULTIHEAD_QUEUE, weightBuf, CL_FALSE, 0, sizeof(float) * (embed_dim* embed_dim * 3), in_weight.data, 0, NULL, &writeEvent[1]);
	CHECK_ERROR(err); 
	err = clEnqueueWriteBuffer(MULTIHEAD_QUEUE, biasBuf, CL_FALSE, 0, sizeof(float) * (embed_dim * 3), in_bias.data, 0, NULL, &writeEvent[2]);
	CHECK_ERROR(err);

   	err = clSetKernelArg(QKV_KERNEL, 0, sizeof(cl_mem), &inputBuf);
	CHECK_ERROR(err);
    err = clSetKernelArg(QKV_KERNEL, 1, sizeof(cl_mem), &weightBuf);
    CHECK_ERROR(err);
    err = clSetKernelArg(QKV_KERNEL, 2, sizeof(cl_mem), &qkvBuf);
    CHECK_ERROR(err);
    err = clSetKernelArg(QKV_KERNEL, 3, sizeof(cl_mem), &biasBuf);
    CHECK_ERROR(err);
    cl_int rowSize= tokens;
    err = clSetKernelArg(QKV_KERNEL, 4, sizeof(cl_int), &rowSize);
    CHECK_ERROR(err);
    cl_int middleSize = embed_dim; 
    err = clSetKernelArg(QKV_KERNEL, 5, sizeof(cl_int), &middleSize);
	CHECK_ERROR(err);

    int tileSize = 16;
    size_t global_size[] = {
        ((tokens + tileSize - 1) / tileSize) * tileSize,        // Round up to multiple of 16
        ((embed_dim + tileSize - 1) / tileSize) * tileSize,
        3
    };
    size_t local_size[] = { tileSize, tileSize, 1 };
    //    size_t global_size[] = {
    //    256,        // Round up to multiple of 16
    //    embed_dim,
    //    3
    //};
    //size_t local_size[] = { 256, 1, 1 };
    err = clEnqueueNDRangeKernel(
        MULTIHEAD_QUEUE,
		QKV_KERNEL,
		3,
		NULL,
		global_size,
		local_size,
		3,
		writeEvent,
		execEvent); 
    CHECK_ERROR(err);
    /*clEnqueueReadBuffer(MULTIHEAD_QUEUE, qkvBuf, CL_TRUE, 0,
        sizeof(float) * tokens * embed_dim * 3, qkv_host, 1, execEvent, NULL);

    printf("=== Checking QKV outputs ===\n");
    printf("Q[0][0:5]: ");
    for (int i = 0; i < 5; i++) printf("%f ", qkv_host[i]);
    printf("\nK[0][0:5]: ");
    for (int i = 0; i < 5; i++) printf("%f ", qkv_host[tokens * embed_dim + i]);
    printf("\nV[0][0:5]: ");
    for (int i = 0; i < 5; i++) printf("%f ", qkv_host[2 * tokens * embed_dim + i]);
    printf("\n");
    free(qkv_host);*/
    // --- 
    int print_tokens = tokens < 5 ? tokens : 5;
    int print_dims = embed_dim < 10 ? embed_dim : 10;

    /*Attn 결과를 저장할 버퍼*/
    float *attn_output = (float *)malloc(sizeof(float) * tokens * embed_dim);
	cl_mem attnBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
	CHECK_ERROR(err);
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 0, sizeof(cl_mem), &qkvBuf);
	CHECK_ERROR(err);
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 1, sizeof(cl_mem), &attnBuf);
	CHECK_ERROR(err);
	cl_int clToken = tokens;
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 2, sizeof(cl_int), &clToken);
	CHECK_ERROR(err);
	cl_int embedDim = embed_dim;
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 3, sizeof(cl_int), &embedDim);
	CHECK_ERROR(err);
	cl_int headDim = head_dim;
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 4, sizeof(cl_int), &headDim);
	cl_int numHeads = num_heads;
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 5, sizeof(cl_int), &numHeads);
	CHECK_ERROR(err);
	size_t global_QKV_TO_SCOREV_size[3] = {256 , tokens, num_heads};
	size_t local_QKV_TO_SCOREV_size[3] = {256, 1, 1};
	err = clEnqueueNDRangeKernel(
        MULTIHEAD_QUEUE, 
        QKV_TO_SCOREV_KERNEL,
        3, 
        NULL, 
        global_QKV_TO_SCOREV_size, 
        local_QKV_TO_SCOREV_size, 
        1, 
        execEvent, &execEvent[1]);
    CHECK_ERROR(err);

    /*float* attn_host = (float*)malloc(sizeof(float) * tokens * embed_dim);
    clEnqueueReadBuffer(MULTIHEAD_QUEUE, attnBuf, CL_TRUE, 0,
        sizeof(float) * tokens * embed_dim, attn_host, 1, &execEvent[1], NULL);

    printf("\n=== Checking attn outputs ===\n");
    printf("attn_host [0][0:5]: ");
    for (int i = 0; i < 5; i++) printf("%f ", attn_host[i]);
    free(attn_host);
    printf("\n");*/
    // 최종 선형 프로젝션
    cl_mem outputBuf= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
	CHECK_ERROR(err);
    cl_mem outWeightBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    cl_mem outBiasBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(MULTIHEAD_QUEUE, outWeightBuf, CL_FALSE, 0, sizeof(float) * (embed_dim * embed_dim), out_weight.data, 1, &execEvent[1], &writeEvent[3]);
	CHECK_ERROR(err); 
	err = clEnqueueWriteBuffer(MULTIHEAD_QUEUE, outBiasBuf, CL_FALSE, 0, sizeof(float) * (embed_dim), out_bias.data, 1, &execEvent[1], &writeEvent[4]);
	CHECK_ERROR(err);

    err = clSetKernelArg(LL_KERNEL, 0, sizeof(cl_mem), &outputBuf);
	CHECK_ERROR(err);
	err = clSetKernelArg(LL_KERNEL, 1, sizeof(cl_mem), &outWeightBuf);
	CHECK_ERROR(err);
	err = clSetKernelArg(LL_KERNEL, 2, sizeof(cl_mem), &attnBuf);
	CHECK_ERROR(err);
	err = clSetKernelArg(LL_KERNEL, 3, sizeof(cl_mem), &outBiasBuf);
	CHECK_ERROR(err);

    cl_int rowA= tokens;
	cl_int colA= embed_dim;
	cl_int colB= embed_dim; 
	err = clSetKernelArg(LL_KERNEL, 4, sizeof(cl_int), &rowA);
	CHECK_ERROR(err); 
	err = clSetKernelArg(LL_KERNEL, 5, sizeof(cl_int), &colA);
	CHECK_ERROR(err);
	err = clSetKernelArg(LL_KERNEL, 6, sizeof(cl_int), &colB);
	CHECK_ERROR(err);

    cl_int doGelu = 0;
    err = clSetKernelArg(LL_KERNEL, 7, sizeof(cl_int), &doGelu);
	CHECK_ERROR(err);

    tileSize = 16;
    size_t global_LL_size[] = {
        ((tokens + tileSize - 1) / tileSize) * tileSize,        // Round up to multiple of 16
        ((embed_dim + tileSize - 1) / tileSize) * tileSize
    };
    size_t local_LL_size[] = { tileSize, tileSize };
    //size_t global_LL_size[] = {
    //    tokens,        // Round up to multiple of 16
    //    embed_dim
    //};
    //size_t local_LL_size[] = { 1, 1 };
	err = clEnqueueNDRangeKernel(
		MULTIHEAD_QUEUE, 
        LL_KERNEL,
		2, 
		NULL, 
		global_LL_size, 
		local_LL_size, 
		2, 
		&writeEvent[3], &execEvent[2]);
	CHECK_ERROR(err); 

	err = clEnqueueReadBuffer(MULTIHEAD_QUEUE, outputBuf, CL_FALSE, 0, sizeof(float) * tokens * embed_dim, output, 1, &execEvent[2], &readEvent);
	CHECK_ERROR(err);

    clFinish(MULTIHEAD_QUEUE);
    //printf("\n=== Checking final outputs ===\n");
    //printf("outputs [0][0:5]: ");
    //for (int i = 0; i < 5; i++) printf("%f ", output[i]);
    //printf("\n");
    profileEvents(writeEvent, 3, &QKV_WRITE_TIME);
    profileEvents(writeEvent + 3, 2, &QKV_WRITE_TIME);
    profileEvents(execEvent, 1, &QKV_EXEC_TIME);
    profileEvents(execEvent + 1, 1, &QKV_TO_SCORE_EXEC_TIME);
    profileEvents(execEvent + 2, 1, &QKV_FINAL_LL_EXEC_TIME);
    profileEvents(&readEvent, 1, &QKV_READ_TIME);

    err = clReleaseMemObject(inputBuf);
    err = clReleaseMemObject(weightBuf);
    err = clReleaseMemObject(biasBuf);
    err = clReleaseMemObject(outWeightBuf);
    err = clReleaseMemObject(outBiasBuf);
    clReleaseMemObject(attnBuf);
    clReleaseMemObject(qkvBuf);
    free(attn_output);
    QKV_EXEC_COUNT += 1;
    //printf("QKV total time: %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);
}


void linear_layer(float* input, float* output, int tokens, int in_features, int out_features, Network weight, Network bias, bool doGelu) {

    //clock_t startTime = clock();
    cl_int err;
    cl_event writeEvent[3];
    cl_event execEvent;
    cl_event readEvent;
	cl_mem outBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens * out_features), NULL, &err);
    CHECK_ERROR(err);
    cl_mem weightBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (in_features * out_features ), NULL, &err);
    CHECK_ERROR(err);
    cl_mem inputBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens * in_features), NULL, &err);
    CHECK_ERROR(err);
    cl_mem biasBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * out_features, NULL, &err);
    CHECK_ERROR(err);
    
    err = clEnqueueWriteBuffer(LL_QUEUE, weightBuf, CL_FALSE, 0, sizeof(float) * (in_features * out_features), weight.data , 0, NULL, writeEvent);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(LL_QUEUE, inputBuf, CL_FALSE, 0, sizeof(float) * (tokens * in_features), input, 0, NULL, writeEvent + 1);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(LL_QUEUE, biasBuf, CL_FALSE, 0, sizeof(float) * out_features, bias.data, 0, NULL, writeEvent + 2);
    CHECK_ERROR(err);

	err = clSetKernelArg(LL_KERNEL, 0, sizeof(cl_mem), &outBuf);
	CHECK_ERROR(err);
    err = clSetKernelArg(LL_KERNEL, 1, sizeof(cl_mem), &weightBuf);
    CHECK_ERROR(err);
    err = clSetKernelArg(LL_KERNEL, 2, sizeof(cl_mem), &inputBuf);
    CHECK_ERROR(err);

    err = clSetKernelArg(LL_KERNEL, 3, sizeof(cl_mem), &biasBuf);
    CHECK_ERROR(err);

    cl_int rowA = tokens;
    cl_int colA= in_features;
    cl_int colB= out_features;
    err = clSetKernelArg(LL_KERNEL, 4, sizeof(cl_int), &rowA);
    CHECK_ERROR(err);
    err = clSetKernelArg(LL_KERNEL, 5, sizeof(cl_int), &colA);
    CHECK_ERROR(err);
    err = clSetKernelArg(LL_KERNEL, 6, sizeof(cl_int), &colB);
	CHECK_ERROR(err);
    cl_int activateGelu = doGelu ? 1 : 0;
    err = clSetKernelArg(LL_KERNEL, 7, sizeof(cl_int), &activateGelu);
	CHECK_ERROR(err); 

    int tileSize = 16;
	size_t global_size[] = {
        ((tokens + tileSize - 1) / tileSize) * tileSize,        // Round up to multiple of 16
        ((out_features + tileSize - 1) / tileSize) * tileSize
    };
    size_t local_size[] = { tileSize, tileSize };  // Must match TILE_SIZE
    //size_t global_size[] = {
    //    tokens,        // Round up to multiple of 16
    //    out_features
    //};
    //size_t local_size[] = {1, 1};  // Must match TILE_SIZE
	err = clEnqueueNDRangeKernel(
		LL_QUEUE,
		LL_KERNEL,
		2,
		NULL,
		global_size,
		local_size,
		3,
		writeEvent,
		&execEvent);
    CHECK_ERROR(err);
    
    err = clEnqueueReadBuffer(LL_QUEUE, outBuf, CL_FALSE, 0, sizeof(float) * (tokens * out_features), output, 1, &execEvent, &readEvent);
	CHECK_ERROR(err); 
    clFinish(LL_QUEUE); 

    profileEvents(writeEvent, 3, &LL_WRITE_TIME);
    profileEvents(&execEvent, 1, &LL_EXEC_TIME);
    profileEvents(&readEvent, 1, &LL_READ_TIME);
    LL_EXEC_COUNT += 1;
	err = clReleaseMemObject(outBuf);
	CHECK_ERROR(err); 
    err = clReleaseMemObject(biasBuf);
	CHECK_ERROR(err); 
    err = clReleaseMemObject(weightBuf);
	CHECK_ERROR(err); 
    err = clReleaseMemObject(inputBuf);
	CHECK_ERROR(err);
    //printf("ll : %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);
}

void mlp_block(float *input, float *output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias)
{
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; // 197
    int Embed_dim = embed_dim;                                            // 768
    int hidden_dim = ((int)(embed_dim * mlp_ratio));                      // 3072

    float *fc1_out = (float *)malloc(sizeof(float) * tokens * hidden_dim);
	if (fc1_out== NULL) printf("malloc failed in line %d\n", __LINE__);
    
    linear_layer(input, fc1_out, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias, true); 
    linear_layer(fc1_out, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias, false);
    if (findNaN(output, tokens, embed_dim)) printf("ll asdf is nan\n");
    free(fc1_out);
}

////////////////////////////////////// Encoder Architecture //////////////////////////////////////
void Encoder(float *input, float *output,
             Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, 
             Network attn_out_w, Network attn_out_b, Network ln2_w, Network ln2_b, 
             Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b)
{
    encoderCount += 1;

    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    float *ln1_out = (float *)malloc(sizeof(float) * tokens * embed_dim);
	if (ln1_out== NULL) printf("malloc failed in line %d\n", __LINE__);
    float *attn_out = (float *)malloc(sizeof(float) * tokens * embed_dim);
	if (attn_out== NULL) printf("malloc failed in line %d\n", __LINE__);
    float *residual = (float *)malloc(sizeof(float) * tokens * embed_dim);
	if (residual== NULL) printf("malloc failed in line %d\n", __LINE__);
    float *ln2_out = (float *)malloc(sizeof(float) * tokens * embed_dim);
	if (ln2_out== NULL) printf("malloc failed in line %d\n", __LINE__);
    float *mlp_out = (float *)malloc(sizeof(float) * tokens * embed_dim);
	if (mlp_out== NULL) printf("malloc failed in line %d\n", __LINE__);

    /*LN1*/
    layer_norm(input, ln1_out, ln1_w, ln1_b);
    //if (findNaN(ln1_out, tokens, embed_dim)) printf("ln 1 is nan on %d, encoder:%d\n", __LINE__, encoderCount);

    /*Attn*/
    multihead_attn(ln1_out, attn_out, attn_w, attn_b, attn_out_w, attn_out_b);
    //if (findNaN(ln1_out, tokens, embed_dim)) printf("multihead is nan on %d, encoder:%d\n", __LINE__, encoderCount);

    /*Residual1*/
    for (int i = 0; i < tokens * embed_dim; i++)
    {
        residual[i] = input[i] + attn_out[i];
    }

    /*LN2*/
    layer_norm(residual, ln2_out, ln2_w, ln2_b);
    //if (findNaN(ln1_out, tokens, embed_dim)) printf("ln 2 is nan on %d, encoder:%d\n", __LINE__, encoderCount);

    /*MLP*/
    mlp_block(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);

    /*Residual2*/
    for (int i = 0; i < tokens * embed_dim; i++)
    {
        output[i] = residual[i] + mlp_out[i];
    }
    free(ln1_out);
    free(attn_out);
    free(residual);
    free(ln2_out);
    free(mlp_out);
}

void Softmax(float *logits, float *probabilities, int length)
{
    // 수치 안정성을 위한 최대값 계산
    float max_val = logits[0];
    for (int i = 1; i < length; i++)
    {
        if (logits[i] > max_val)
        {
            max_val = logits[i];
        }
    }

    // 각 원소에 대해 exp(logit - max_val)을 계산하고 합산
    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++)
    {
        probabilities[i] = expf(logits[i] - max_val);
        sum_exp += probabilities[i];
    }

    // 확률값으로 정규화
    for (int i = 0; i < length; i++)
    {
        probabilities[i] /= sum_exp;
    }
}

////////////////////////////////////// layer별 size //////////////////////////////////////
const int size[] = {
    embed_dim * (img_size / patch_size) * (img_size / patch_size),      // conv2D
    embed_dim *(img_size / patch_size) * (img_size / patch_size),       // flatten and transpose
    embed_dim *((img_size / patch_size) * (img_size / patch_size) + 1), // class token
    embed_dim *((img_size / patch_size) * (img_size / patch_size) + 1)  // position embedding
};

const int enc_size = embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1);



////////////////////////////////////// Model Architecture //////////////////////////////////////
void ViT_opencl(ImageData *image, Network *networks, float **probabilities)
{  
    time_t setupTime = clock();
    cl_int err;
    // Platform ID
    err = clGetPlatformIDs(1, &PLATFORM, NULL);
    CHECK_ERROR(err);

    // Device ID
    err = clGetDeviceIDs(PLATFORM, CL_DEVICE_TYPE_GPU, 1, &DEVICE, NULL);
    CHECK_ERROR(err);

    // Create Context
    CONTEXT = clCreateContext(NULL, 1, &DEVICE, NULL, NULL, &err);
    CHECK_ERROR(err);

    int token_size = ((img_size / patch_size) * (img_size / patch_size) + 1);
    float *layer[4];
    float *enc_layer[12];
    float *enc_output;
    int hidden_dim = ((int)(embed_dim * mlp_ratio));
    // printf("%d %d = %d\n", token_size, hidden_dim, token_size * hidden_dim);

    for (int i = 0; i < 4; i++)
    {
        layer[i] = (float *)malloc(sizeof(float) * size[i]);
        if (layer[i] == NULL) printf("malloc failed in line %d\n", __LINE__);
    }
    for (int i = 0; i < 12; i++)
    {
        enc_layer[i] = (float *)malloc(sizeof(float) * enc_size);
        if (enc_layer[i] == NULL) printf("malloc failed in line %d\n", __LINE__);
    }
    enc_output = (float *)malloc(sizeof(float) * enc_size);
	if (enc_output == NULL) printf("malloc failed in line %d\n", __LINE__);
       

	// for multihead
	size_t kernel_source_size;
	char* kernel_source = get_source_code("multihead.cl", &kernel_source_size);
	MULTIHEAD_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(MULTIHEAD_PROGRAM, 1, &DEVICE, "", NULL, NULL);
	build_error(MULTIHEAD_PROGRAM, DEVICE, err);
	CHECK_ERROR(err);
	QKV_KERNEL = clCreateKernel(MULTIHEAD_PROGRAM, "QKV", &err);
	CHECK_ERROR(err);
    QKV_TO_SCOREV_KERNEL = clCreateKernel(MULTIHEAD_PROGRAM, "QKV_TO_SCOREV", &err);
    CHECK_ERROR(err);


	cl_queue_properties props[] = {
		CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
		//CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
		0
	};
	MULTIHEAD_QUEUE= clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err);
	CHECK_ERROR(err);    
    
    // for ll
	kernel_source = get_source_code("ll.cl", &kernel_source_size);
	LL_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(LL_PROGRAM, 1, &DEVICE, "", NULL, NULL);
	build_error(LL_PROGRAM, DEVICE, err);
	CHECK_ERROR(err);
	LL_KERNEL = clCreateKernel(LL_PROGRAM, "linear_layer", &err);
	CHECK_ERROR(err);
	LL_QUEUE = clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err);
    CHECK_ERROR(err);

    // for conv2d
    kernel_source = get_source_code("conv2d.cl", &kernel_source_size);
    CONV2D_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source,
        &kernel_source_size, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(CONV2D_PROGRAM, 1, &DEVICE, "", NULL, NULL);
    build_error(CONV2D_PROGRAM, DEVICE, err);
    CHECK_ERROR(err);
    CONV2D_KERNEL = clCreateKernel(CONV2D_PROGRAM, "conv2d_kernel", &err);
	CHECK_ERROR(err);
    CONV2D_QUEUE = clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err);
	CHECK_ERROR(err);
    
    // for layer_norm 
	kernel_source = get_source_code("layer_norm.cl", &kernel_source_size);
	LAYERNORM_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);
	CHECK_ERROR(clBuildProgram(LAYERNORM_PROGRAM, 1, &DEVICE, "", NULL, NULL));
	build_error(CONV2D_PROGRAM, DEVICE, err);
	CHECK_ERROR(err);
	LAYERNORM_KERNEL= clCreateKernel(LAYERNORM_PROGRAM, "layerNorm", &err);
	CHECK_ERROR(err);
	LAYERNORM_QUEUE= clCreateCommandQueueWithProperties(CONTEXT, DEVICE,props, &err);
	CHECK_ERROR(err);
    printf("setup time: %.6f sec\n\n" , (double)(clock() - setupTime) / CLK_TCK);
    //printSpec();

    //test_linear_layer();
    //test_linear_layer_big();
    for (int i = 0; i < image->n; i++)
    {
        double startTime = clock();
        /*patch embedding*/
        Conv2d(image[i].data, layer[0], networks[1], networks[2]);
        /*flatten and transpose*/
        flatten_transpose(layer[0], layer[1]);
        /*prepend class token*/
        class_token(layer[1], layer[2], networks[0]);
        /*position embedding*/
        pos_emb(layer[2], layer[3], networks[3]); 
        printf("pre-processing : %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);
        /*Encoder - 12 Layers*/
        Encoder(layer[3], enc_layer[0],
            networks[4], networks[5], networks[6], networks[7],
            networks[8], networks[9], networks[10], networks[11],
            networks[12], networks[13], networks[14], networks[15]);

        for (int j = 1; j < 12; j++) {
            Encoder(enc_layer[j - 1], enc_layer[j],
                networks[4 + j * 12], networks[5 + j * 12], networks[6 + j * 12], networks[7 + j * 12],
                networks[8 + j * 12], networks[9 + j * 12], networks[10 + j * 12], networks[11 + j * 12],
                networks[12 + j * 12], networks[13 + j * 12], networks[14 + j * 12], networks[15 + j * 12]);
        }

        for (int asdf = 0; asdf < 12; asdf += 1)
            if (findNaN(enc_layer[asdf], 197, embed_dim)) printf("%d has nan", asdf);
         
        layer_norm(enc_layer[11], enc_output, networks[148], networks[149]);

        /* Token 값 추출 */
        float *cls_token = (float *)malloc(sizeof(float) * embed_dim);
        if (cls_token == NULL) printf("malloc failed on %d\n", __LINE__);
        float *cls_output = (float *)malloc(sizeof(float) * num_classes);
        if (cls_output == NULL) printf("malloc failed\n");
        memcpy(cls_token, enc_output, sizeof(float) * embed_dim);

        linear_layer(cls_token, cls_output, 1, embed_dim, num_classes, networks[150], networks[151], false);
        /* 확률분포 추출 */
        Softmax(cls_output, probabilities[i], num_classes);

        printf("picture #%d: %.6f sec\n\n", i, (double)(clock() - startTime) / CLK_TCK);
    }
    printEventProfile();

    CHECK_ERROR(clReleaseKernel(LL_KERNEL));
    CHECK_ERROR(clReleaseKernel(QKV_KERNEL));
    CHECK_ERROR(clReleaseKernel(QKV_TO_SCOREV_KERNEL));
    CHECK_ERROR(clReleaseKernel(CONV2D_KERNEL));
    CHECK_ERROR(clReleaseKernel(LAYERNORM_KERNEL));
	CHECK_ERROR(clReleaseProgram(MULTIHEAD_PROGRAM));
    CHECK_ERROR(clReleaseProgram(LL_PROGRAM));
    CHECK_ERROR(clReleaseProgram(CONV2D_PROGRAM));
    CHECK_ERROR(clReleaseProgram(LAYERNORM_PROGRAM));
	CHECK_ERROR(clReleaseCommandQueue(MULTIHEAD_QUEUE));
    CHECK_ERROR(clReleaseCommandQueue(LL_QUEUE));
    CHECK_ERROR(clReleaseCommandQueue(CONV2D_QUEUE));
    CHECK_ERROR(clReleaseCommandQueue(LAYERNORM_QUEUE));

    err = clReleaseContext(CONTEXT);
    CHECK_ERROR(err);
}


void printSpec() {
    cl_uint num_platforms;
    cl_platform_id *platforms;
    cl_uint num_devices;
    cl_device_id *devices;
    char str[1024];
    cl_device_type device_type;
    size_t max_work_group_size;
    cl_uint max_clock_frequency;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    cl_ulong max_mem_alloc_size;
    cl_ulong max_compute_units;
    cl_command_queue_properties queue_properties;
    cl_uint p, d;
    cl_int err;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err);

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err);

    printf("Number of platforms: %u\n\n", num_platforms);
    for (p = 0; p < num_platforms; p++)
    {
        printf("platform: %u\n", p);

        err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, str, NULL);
        CHECK_ERROR(err);
        printf("- CL_PLATFORM_NAME\t:%s\n", str);

        err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, str, NULL);
        CHECK_ERROR(err);
        printf("- CL_PLATFORM_VENDOR\t:%s\n\n", str);

        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        CHECK_ERROR(err);
        printf("Number of devices:\t%u\n\n", num_devices);

        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        CHECK_ERROR(err);

        for (d = 0; d < num_devices; d++)
        {
            printf("device: %u\n", d);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_TYPE\t:");
            if (device_type & CL_DEVICE_TYPE_CPU)
                printf(" CL_DEVICE_TYPE_CPU");
            if (device_type & CL_DEVICE_TYPE_GPU)
                printf(" CL_DEVICE_TYPE_GPU");
            if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
                printf(" CL_DEVICE_TYPE_ACCELERATOR");
            if (device_type & CL_DEVICE_TYPE_DEFAULT)
                printf(" CL_DEVICE_TYPE_DEFAULT");
            if (device_type & CL_DEVICE_TYPE_CUSTOM)
                printf(" CL_DEVICE_TYPE_CUSTOM");
            printf("\n");

            err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, str, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_NAME\t: %s\n", str);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, 1024, str, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_VENDOR\t: %s\n", str);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, 1024, str, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_VERSION\t: %s\n", str);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_ulong), &max_clock_frequency, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_MAX_CLOCK_FREQUENCY : %luMHz\n", max_clock_frequency);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &max_compute_units, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_MAX_COMPUTE_UNITS : %lu\n", max_compute_units);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE : %lu\n", max_work_group_size);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_GLOBAL_MEM_SIZE : %lu\n", global_mem_size);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_LOCAL_MEM_SIZE : %lu\n", local_mem_size);

            err = clGetDeviceInfo(devices[d], CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_ulong), &queue_properties, NULL);
            CHECK_ERROR(err);
            printf("- CL_DEVICE_QUEUE_PROPERTIES :");
            if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                printf(" CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
            if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
                printf(" CL_QUEUE_PROFILING_ENABLE");
            printf("\n");
        }

        free(devices);
    }

    free(platforms);
    return 0;
}

void profileEvents(cl_event *events, int eventCount, cl_ulong* timeGlobalVariable) { 
    cl_ulong start, end, minStart = ULLONG_MAX, maxEnd = 0;
    for (int i = 0; i < eventCount; i++) {
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                               sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, 
                               sizeof(cl_ulong), &end, NULL);

        if (i == 1 && (minStart <= start && start <= maxEnd)) printf("second pipe! nice\n");
        if (start < minStart) minStart= start;
        if (end > maxEnd) maxEnd= end;
    }
    cl_ulong total = (maxEnd- minStart);

    *timeGlobalVariable += total;
}

void printEventProfile() {
    printf("\n========== PROFILING RESULTS ==========\n");
    
    if (LAYERNORM_EXEC_COUNT > 0) {
        printf("\nLayerNorm (%d executions):\n", LAYERNORM_EXEC_COUNT);
        printf("  Write: %.6f sec (avg: %.6f sec)\n",
               LAYERNORM_WRITE_TIME / 1000000000.0,
               LAYERNORM_WRITE_TIME / 1000000000.0 / LAYERNORM_EXEC_COUNT);
        printf("  Exec:  %.6f sec (avg: %.6f sec)\n",
               LAYERNORM_EXEC_TIME / 1000000000.0,
               LAYERNORM_EXEC_TIME / 1000000000.0 / LAYERNORM_EXEC_COUNT);
        printf("  Read:  %.6f sec (avg: %.6f sec)\n",
               LAYERNORM_READ_TIME / 1000000000.0,
               LAYERNORM_READ_TIME / 1000000000.0 / LAYERNORM_EXEC_COUNT);
        printf("  Total: %.6f sec (avg: %.6f sec)\n",
               (LAYERNORM_WRITE_TIME + LAYERNORM_EXEC_TIME + LAYERNORM_READ_TIME) / 1000000000.0,
               (LAYERNORM_WRITE_TIME + LAYERNORM_EXEC_TIME + LAYERNORM_READ_TIME) / 1000000000.0 / LAYERNORM_EXEC_COUNT);
    }
    
    if (QKV_EXEC_COUNT> 0) {
        printf("\nMultiHeadAttention (%d executions):\n", QKV_EXEC_COUNT);
        printf("  Write: %.6f sec (avg: %.6f sec)\n",
               QKV_WRITE_TIME / 1000000000.0,
               QKV_WRITE_TIME / 1000000000.0 / QKV_EXEC_COUNT);
        printf("  Exec(QKV):  %.6f sec (avg: %.6f sec)\n",
               QKV_EXEC_TIME / 1000000000.0,
               QKV_EXEC_TIME / 1000000000.0 / QKV_EXEC_COUNT);
        printf("  Exec(QKV_TO_SCOREV):  %.6f sec (avg: %.6f sec)\n",
			   QKV_TO_SCORE_EXEC_TIME / 1000000000.0,
			   QKV_TO_SCORE_EXEC_TIME / 1000000000.0 / QKV_EXEC_COUNT);
        printf("  Exec(final LL):  %.6f sec (avg: %.6f sec)\n",
			   QKV_FINAL_LL_EXEC_TIME / 1000000000.0,
			   QKV_FINAL_LL_EXEC_TIME / 1000000000.0 / QKV_EXEC_COUNT);
        printf("  Read:  %.6f sec (avg: %.6f sec)\n",
               QKV_READ_TIME / 1000000000.0,
               QKV_READ_TIME / 1000000000.0 / QKV_EXEC_COUNT);
    }
    
    if (LL_EXEC_COUNT > 0) {
        printf("\nLL (%d executions):\n", LL_EXEC_COUNT);
        printf("  Write: %.6f sec (avg: %.6f sec)\n",
               LL_WRITE_TIME / 1000000000.0,
               LL_WRITE_TIME / 1000000000.0 / LL_EXEC_COUNT);
        printf("  Exec:  %.6f sec (avg: %.6f sec)\n",
               LL_EXEC_TIME / 1000000000.0,
               LL_EXEC_TIME / 1000000000.0 / LL_EXEC_COUNT);
        printf("  Read:  %.6f sec (avg: %.6f sec)\n",
               LL_READ_TIME / 1000000000.0,
               LL_READ_TIME / 1000000000.0 / LL_EXEC_COUNT);
    }

	if (CONV_EXEC_COUNT> 0) {
		printf("\nCONV (%d executions):\n", CONV_EXEC_COUNT);
		printf("  Write: %.6f sec (avg: %.6f sec)\n",
			   CONV_WRITE_TIME / 1000000000.0,
			   CONV_WRITE_TIME / 1000000000.0 / CONV_EXEC_COUNT);
		printf("  Exec:  %.6f sec (avg: %.6f sec)\n",
			   CONV_EXEC_TIME / 1000000000.0,
			   CONV_EXEC_TIME / 1000000000.0 / CONV_EXEC_COUNT);
		printf("  Read:  %.6f sec (avg: %.6f sec)\n",
			   CONV_READ_TIME / 1000000000.0,
			   CONV_READ_TIME / 1000000000.0 / CONV_EXEC_COUNT);
	}
    cl_ulong allwrite= CONV_WRITE_TIME + LL_WRITE_TIME+ QKV_WRITE_TIME + LAYERNORM_WRITE_TIME;
    cl_ulong allExec= CONV_EXEC_TIME + LL_EXEC_TIME + LAYERNORM_EXEC_TIME + QKV_EXEC_TIME + QKV_TO_SCORE_EXEC_TIME + QKV_FINAL_LL_EXEC_TIME;
    cl_ulong allRead = CONV_READ_TIME + LL_READ_TIME+ QKV_READ_TIME + LAYERNORM_READ_TIME;

    printf("\nTotal Execution Time By Jobs:\n");
    printf("  Total write:  %.6f sec \n",
        allwrite / 1000000000.0);
	printf("  Total Exec:  %.6f sec \n",
		allExec / 1000000000.0);
   printf("  Total Read:  %.6f sec \n",
	   allRead / 1000000000.0);


    cl_ulong grandTotal = allwrite + allExec + allRead;

    if (grandTotal == 0) {
        printf("\nNo profiling data available for total ratio breakdown.\n");
    }
    else {
        double totalSec = grandTotal / 1000000000.0;

        printf("\nOverall Totals:\n");
        printf("  Grand Total Time: %.6f sec\n", totalSec);

        double write_ratio = (double)allwrite / (double)grandTotal * 100.0;
        double exec_ratio = (double)allExec / (double)grandTotal * 100.0;
        double read_ratio = (double)allRead / (double)grandTotal * 100.0;

        printf("  Write portion: %.2f%%\n", write_ratio);
        printf("  Exec portion:  %.2f%%\n", exec_ratio);
        printf("  Read portion:  %.2f%%\n", read_ratio);

        printf("\nBreakdown by job type (percentage of total):\n");

        // LayerNorm
        if (LAYERNORM_EXEC_COUNT > 0) {
            double w = LAYERNORM_WRITE_TIME / (double)grandTotal * 100.0;
            double e = LAYERNORM_EXEC_TIME / (double)grandTotal * 100.0;
            double r = LAYERNORM_READ_TIME / (double)grandTotal * 100.0;
            printf("  LayerNorm: write %.2f%%, exec %.2f%%, read %.2f%% (sum %.2f%%)\n",
                w, e, r, w + e + r);
        }

        // QKV (MultiHeadAttention)
        if (QKV_EXEC_COUNT > 0) {
            double w = QKV_WRITE_TIME / (double)grandTotal * 100.0;
            double e1 = QKV_EXEC_TIME / (double)grandTotal * 100.0;
            double e2 = QKV_TO_SCORE_EXEC_TIME / (double)grandTotal * 100.0;
            double e3 = QKV_FINAL_LL_EXEC_TIME / (double)grandTotal * 100.0;
            double r = QKV_READ_TIME / (double)grandTotal * 100.0;
            printf("  MultiHeadAttention: write %.2f%%, exec %.2f%% + %.2f%% + %.2f%%, read %.2f%% (sum %.2f%%)\n",
                w, e1, e2, e3, r, w + e1 + e2 + e3 + r);
        }

        // LL
        if (LL_EXEC_COUNT > 0) {
            double w = LL_WRITE_TIME / (double)grandTotal * 100.0;
            double e = LL_EXEC_TIME / (double)grandTotal * 100.0;
            double r = LL_READ_TIME / (double)grandTotal * 100.0;
            printf("  LL: write %.2f%%, exec %.2f%%, read %.2f%% (sum %.2f%%)\n",
                w, e, r, w + e + r);
        }

        // CONV
        if (CONV_EXEC_COUNT > 0) {
            double w = CONV_WRITE_TIME / (double)grandTotal * 100.0;
            double e = CONV_EXEC_TIME / (double)grandTotal * 100.0;
            double r = CONV_READ_TIME / (double)grandTotal * 100.0;
            printf("  CONV: write %.2f%%, exec %.2f%%, read %.2f%% (sum %.2f%%)\n",
                w, e, r, w + e + r);
        }
    }
    printf("\n=======================================\n");
}


void test_linear_layer(){
    // Simple 3×2 input, 2×4 weight
    float input[6] = {1, 2,  3, 4,  5, 6};       // [3, 2]
    float weight[8] = {1, 0, 0, 1,  0, 1, 1, 0}; // [4, 2]
    //float bias[4] = {1, 1, 1, 1};
    float bias[4] = {0, 0, 0, 0};
    float output[12];  // [3, 4]


    Network weight_net = { weight, 8 };
    Network bias_net = { bias, 4 };
    
    // Expected output:
    // [1, 2, 2, 1]
    // [3, 4, 4, 3]
    // [5, 6, 6, 5]
    
    // Run your kernel
    linear_layer(input, output, 3, 2, 4, weight_net, bias_net, false);
    
    // Check results
    printf("Output:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.1f ", output[i * 4 + j]);
        }
        printf("\n");
    }
}

bool findNaN(float* a, int tokens, int embedings) {
    int count = 0;
    for (int i = 0; i < tokens; i++) {
        for (int j = 0; j < embedings; j++)
            if (a[i * embedings + j] != a[i * embedings + j]) {
                printf("found NaN on %d %d\n", i, j);
                return true;
            }
    }

    return false;
}
void test_linear_layer_big(){
    float input[17 * 32];    
    float weight[32 * 32];
    float bias[32] = { 0 };
    float output[32 * 32];

    for (int i = 0; i < 17; i++) {
        for (int j = 0; j < 32; j++) {
            input[i * 32 + j] = 1;
        }
    }
    for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			weight[i * 32 + j] = i;
		}
	}

    Network weight_net = { weight, 32*32 };
    Network bias_net = { bias, 32 };
    
    // Expected output:
    
    // Run your kernel
    linear_layer(input, output, 17, 32, 32, weight_net, bias_net, false);
    
    // Check results
    if (findNaN(output, 17, 32)){
		printf("found Nan during testing");
        return;
    }

    printf("Output:\n");
    for (int i = 0; i < 17; i++) {
		for (int j = 0; j < 32; j++) {
            printf("%.1f ", output[i * 32 + j]);
		}
        printf("\n");
	}



    for (int i = 0; i < 17; i++) {
        for (int j = 0; j < 32; j++) {
            if (output[i * 32 + j] != 32 * j) {
                printf("wrong anwer(%f) on %d %d\n", output[i * 32 + j], i, j);
                return;
            }
        }
    }

    printf("all good!\n");
}