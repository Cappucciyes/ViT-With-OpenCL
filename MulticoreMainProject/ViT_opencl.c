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

typedef struct {
    cl_mem ln1_w, ln1_b;
    cl_mem attn_w, attn_b;
    cl_mem attn_out_w, attn_out_b;
    cl_mem ln2_w, ln2_b;
    cl_mem mlp1_w, mlp1_b;
    cl_mem mlp2_w, mlp2_b;
} EncoderWeights;

typedef struct {
    cl_mem conv_w, conv_b;
    cl_mem cls;
    cl_mem pos;
    cl_mem ln_w, ln_b;
    cl_mem mlp_w, mlp_b;
} LoadedOnceWeight;

typedef struct {
	cl_mem CONV_BUF;
	cl_mem LL_BUF;
	cl_mem QKV_BUF;
	cl_mem ATTN_BUF;

    cl_mem enc_layer1;
    cl_mem enc_layer2;

	cl_mem ln1Out;
	cl_mem mthOut;
	cl_mem residual;
	cl_mem ln2Out;
	cl_mem mlpOut;
	cl_mem softMaxInputBuf;
	cl_mem softMaxBuf;
} ImageBufferSet;


// Global Variable
cl_platform_id PLATFORM;
cl_device_id DEVICE;
cl_context CONTEXT;

cl_command_queue WRITE_COMMAND_QUEUE;
cl_command_queue EXEC_COMMAND_QUEUE;
cl_command_queue READ_COMMAND_QUEUE;

cl_program MULTIHEAD_PROGRAM;
cl_kernel QKV_KERNEL;
cl_kernel QKV_TO_SCOREV_KERNEL;

cl_program LL_PROGRAM;
cl_kernel LL_KERNEL;

cl_program CONV2D_PROGRAM;
cl_kernel CONV2D_KERNEL;
cl_kernel POST_PROCESS_KERNEL;

cl_program LAYERNORM_PROGRAM;
cl_kernel LAYERNORM_KERNEL;
cl_kernel RESIDUAL_KERNEL;

cl_program SOFTMAX_PROGRAM;
cl_kernel SOFTMAX_KERNEL;

cl_ulong CONV_EXEC_TIME;
int CONV_EXEC_COUNT;

cl_ulong QKV_EXEC_TIME;
cl_ulong QKV_TO_SCORE_EXEC_TIME;
cl_ulong QKV_FINAL_LL_EXEC_TIME;
int QKV_EXEC_COUNT;

cl_ulong LL_EXEC_TIME;
int LL_EXEC_COUNT;

cl_ulong LAYERNORM_EXEC_TIME;
int LAYERNORM_EXEC_COUNT;

cl_ulong OFF_WRITE_TIME;
int encoderCount;


cl_event ENCODER_FLAGS[12 * 5 * 100]; //(layernorm, multihead, layer norm, ll ,ll) * 12 * 100
cl_event START_CONV[101]; //100 + 마지막 하나 더 (imageBufset 정리하기 위해)
cl_event START_POST_CONV[100]; //
cl_event END_PATCHING[100]; //
cl_event DO_RESIDUAL[2400];
cl_event DO_SOFTMAX[100];
cl_event ENCODER_WEIGHT_WRITE_EVENT[12 * 12]; 
cl_event FINAL_FLAG[100]; //(layernorm, ll) * 12 

////////////////////////////////////// utils function //////////////////////////////////////
void profileEvents(cl_event* events, int eventCount, cl_ulong* timeGlobalVariable);

// testing 
void printEventProfile();
//void test_linear_layer();
bool findNaN(float* a, int tokens, int embedings);
//void test_linear_layer_big();
////////////////////////////data loading
EncoderWeights initEncoderWeight() {
    EncoderWeights weights;
	cl_int err;
    int hidden_dim = ((int)(embed_dim * mlp_ratio));

    weights.ln1_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.ln1_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.attn_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim * 3, NULL, &err);
    CHECK_ERROR(err);
    weights.attn_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * 3, NULL, &err);
    CHECK_ERROR(err);
    weights.attn_out_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.attn_out_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.ln2_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.ln2_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.mlp1_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * hidden_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.mlp1_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * hidden_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.mlp2_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * hidden_dim, NULL, &err);
    CHECK_ERROR(err);
    weights.mlp2_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    return weights;
}

void fillEncoderWeight(EncoderWeights weights, Network* networkList, int encoderIndex, cl_command_queue writingPipe, int flagNumbers, cl_event *flagEvent, cl_event* writeEvents) {
    cl_int err;
    int base = 4 + encoderIndex * 12;
    err = clEnqueueWriteBuffer(writingPipe, weights.ln1_w, CL_FALSE, 0,
        sizeof(float) * networkList[base].size,
        networkList[base].data, flagNumbers, flagEvent, &writeEvents[0]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.ln1_b, CL_FALSE, 0,
        sizeof(float) * networkList[base + 1].size,
        networkList[base + 1].data, flagNumbers, flagEvent, &writeEvents[1]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.attn_w, CL_FALSE, 0,
        sizeof(float) * networkList[base + 2].size,
        networkList[base + 2].data, flagNumbers, flagEvent, &writeEvents[2]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.attn_b, CL_FALSE, 0,
        sizeof(float) * networkList[base + 3].size,
        networkList[base + 3].data, flagNumbers, flagEvent, &writeEvents[3]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.attn_out_w, CL_FALSE, 0,
        sizeof(float) * networkList[base + 4].size,
        networkList[base + 4].data, flagNumbers, flagEvent, &writeEvents[4]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.attn_out_b, CL_FALSE, 0,
        sizeof(float) * networkList[base + 5].size,
        networkList[base + 5].data, flagNumbers, flagEvent, &writeEvents[5]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.ln2_w, CL_FALSE, 0,
        sizeof(float) * networkList[base + 6].size,
        networkList[base + 6].data, flagNumbers, flagEvent, &writeEvents[6]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.ln2_b, CL_FALSE, 0,
        sizeof(float) * networkList[base + 7].size,
        networkList[base + 7].data, flagNumbers, flagEvent, &writeEvents[7]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.mlp1_w, CL_FALSE, 0,
        sizeof(float) * networkList[base + 8].size,
        networkList[base + 8].data, flagNumbers, flagEvent, &writeEvents[8]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.mlp1_b, CL_FALSE, 0,
        sizeof(float) * networkList[base + 9].size,
        networkList[base + 9].data, flagNumbers, flagEvent, &writeEvents[9]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.mlp2_w, CL_FALSE, 0,
        sizeof(float) * networkList[base + 10].size,
        networkList[base + 10].data, flagNumbers, flagEvent, &writeEvents[10]);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(writingPipe, weights.mlp2_b, CL_FALSE, 0,
        sizeof(float) * networkList[base + 11].size,
        networkList[base + 11].data, flagNumbers, flagEvent, &writeEvents[11]);
    CHECK_ERROR(err);
}

void releaseEncoderWeights(EncoderWeights weights) {
    clReleaseMemObject(weights.ln1_w);
    clReleaseMemObject(weights.ln1_b);
    clReleaseMemObject(weights.attn_w);
    clReleaseMemObject(weights.attn_b);
    clReleaseMemObject(weights.attn_out_w);
    clReleaseMemObject(weights.attn_out_b);
    clReleaseMemObject(weights.ln2_w);
    clReleaseMemObject(weights.ln2_b);
    clReleaseMemObject(weights.mlp1_w);
    clReleaseMemObject(weights.mlp1_b);
    clReleaseMemObject(weights.mlp2_w);
    clReleaseMemObject(weights.mlp2_b);
}

ImageBufferSet initImageBufferSet() {
    ImageBufferSet bufferset;
    cl_int err; int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    bufferset.CONV_BUF= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * (img_size / patch_size) * (img_size / patch_size), NULL, &err);
    CHECK_ERROR(err);
    bufferset.QKV_BUF= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens* embed_dim * 3), NULL, &err);
    CHECK_ERROR(err);
    bufferset.ATTN_BUF= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens* embed_dim), NULL, &err);
    CHECK_ERROR(err);
    bufferset.LL_BUF= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * (tokens* 3072), NULL, &err);
    CHECK_ERROR(err);

    bufferset.enc_layer1= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
	CHECK_ERROR(err);
	bufferset.enc_layer2= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
	CHECK_ERROR(err);

    bufferset.ln1Out= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    bufferset.mthOut= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    bufferset.residual= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    bufferset.ln2Out= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    bufferset.mlpOut= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);

    bufferset.softMaxInputBuf= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * num_classes, NULL, &err);
	CHECK_ERROR(err);
    bufferset.softMaxBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * num_classes, NULL, &err);
	CHECK_ERROR(err);
    return bufferset;
}

void releaseImageBufferSet(ImageBufferSet bufferset) {
    CHECK_ERROR(clReleaseMemObject(bufferset.CONV_BUF));
    CHECK_ERROR(clReleaseMemObject(bufferset.QKV_BUF));
    CHECK_ERROR(clReleaseMemObject(bufferset.ATTN_BUF));
    CHECK_ERROR(clReleaseMemObject(bufferset.LL_BUF));
    CHECK_ERROR(clReleaseMemObject(bufferset.enc_layer1));
    CHECK_ERROR(clReleaseMemObject(bufferset.enc_layer2));
    CHECK_ERROR(clReleaseMemObject(bufferset.ln1Out));
    CHECK_ERROR(clReleaseMemObject(bufferset.mthOut));
    CHECK_ERROR(clReleaseMemObject(bufferset.residual));
    CHECK_ERROR(clReleaseMemObject(bufferset.ln2Out));
    CHECK_ERROR(clReleaseMemObject(bufferset.mlpOut));
    CHECK_ERROR(clReleaseMemObject(bufferset.softMaxInputBuf));
    CHECK_ERROR(clReleaseMemObject(bufferset.softMaxBuf));
}

LoadedOnceWeight initLoadedOnceWeight(Network* networkList) {
    LoadedOnceWeight weights;
    cl_int err;
    
    weights.conv_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[1].size, NULL, &err);
    CHECK_ERROR(err);
	weights.conv_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[2].size, NULL, &err);
	CHECK_ERROR(err);
	weights.cls= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[0].size, NULL, &err);
	CHECK_ERROR(err);
	weights.pos= clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[3].size, NULL, &err);
	CHECK_ERROR(err);

	weights.ln_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[148].size, NULL, &err);
	CHECK_ERROR(err);
	weights.ln_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[149].size, NULL, &err);
    CHECK_ERROR(err);
	weights.mlp_w = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[150].size, NULL, &err);
	CHECK_ERROR(err);
	weights.mlp_b = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, sizeof(float) * networkList[151].size, NULL, &err);
	CHECK_ERROR(err);

    return weights;
}

void fillLoadedOnceWeight(LoadedOnceWeight weights, Network* networkList) {
    cl_int err;
    
    err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.conv_w, CL_FALSE, 0, sizeof(float) * networkList[1].size, networkList[1].data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.conv_b, CL_FALSE, 0, sizeof(float) * networkList[2].size, networkList[2].data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.cls, CL_FALSE, 0, sizeof(float) * networkList[0].size, networkList[0].data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.pos, CL_FALSE, 0, sizeof(float) * networkList[3].size, networkList[3].data, 0, NULL, NULL);
	CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.ln_w, CL_FALSE, 0, sizeof(float) * networkList[148].size, networkList[148].data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.ln_b, CL_FALSE, 0, sizeof(float) * networkList[149].size, networkList[149].data, 0, NULL, NULL);
	CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.mlp_w, CL_FALSE, 0, sizeof(float) * networkList[150].size, networkList[150].data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, weights.mlp_b, CL_FALSE, 0, sizeof(float) * networkList[151].size, networkList[151].data, 0, NULL, NULL);
	CHECK_ERROR(err);

    clFinish(WRITE_COMMAND_QUEUE);
}

void releaseLoadedOncelWeights(LoadedOnceWeight weights) {
    clReleaseMemObject(weights.ln_w);
    clReleaseMemObject(weights.ln_b);
    clReleaseMemObject(weights.mlp_w);
    clReleaseMemObject(weights.mlp_b);
}

// This function is called by OpenCL when the event completes
void CL_CALLBACK cleanupCallback(cl_event event, cl_int event_status, void* user_data) {
	ImageBufferSet* toClean = (ImageBufferSet*)user_data;

    if (event_status != CL_COMPLETE) {
        printf("Warning: Event completed with status %d\n", event_status);
    }

    releaseImageBufferSet(*toClean);
	free(toClean);
}

// ==================== Helper to Register Cleanup ====================

void registerCleanup(cl_event completion_event, ImageBufferSet toCleanup) {
    // Register callback - OpenCL will call cleanupCallback when event completes
    ImageBufferSet* persistent = (ImageBufferSet*)malloc(sizeof(ImageBufferSet));
    if (persistent == NULL) {
        printf("Failed to allocate memory for cleanup callback\n");
        return;
    }
    *persistent = toCleanup;

    cl_int err = clSetEventCallback(completion_event, CL_COMPLETE, cleanupCallback, persistent);
    CHECK_ERROR(err);
}

////////////////////////////////////// ViT function //////////////////////////////////////

void Conv2d(float* input, cl_mem output, cl_mem weight, cl_mem bias, int flagCount, cl_event* flagEvent, cl_event *nextEvent)
{
    int output_size = img_size / patch_size;
    cl_int err;
    cl_event writeEvent;
    cl_event execEvent;

    cl_mem inputBuf = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY, sizeof(float) * img_size * img_size * in_chans, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(WRITE_COMMAND_QUEUE, inputBuf, CL_FALSE, 0, sizeof(float) * img_size * img_size * in_chans, input, flagCount, flagEvent, &writeEvent);
    CHECK_ERROR(err);

    err = clSetKernelArg(CONV2D_KERNEL, 0, sizeof(cl_mem), &inputBuf);
    CHECK_ERROR(err);
    err = clSetKernelArg(CONV2D_KERNEL, 1, sizeof(cl_mem), &output);
    CHECK_ERROR(err);
    err = clSetKernelArg(CONV2D_KERNEL, 2, sizeof(cl_mem), &weight);
    CHECK_ERROR(err);
    err = clSetKernelArg(CONV2D_KERNEL, 3, sizeof(cl_mem), &bias);
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

    err = clEnqueueNDRangeKernel(EXEC_COMMAND_QUEUE, CONV2D_KERNEL, 3, NULL,
        global_size, local_size, 1, &writeEvent, nextEvent);
    CHECK_ERROR(err);

    //profileEvents(&execEvent, 1, &CONV_EXEC_TIME);
    clReleaseMemObject(inputBuf);
}


void postConv2d(cl_mem input, cl_mem output, cl_mem cls, cl_mem pos, int flagCount, cl_event* flagEvent, cl_event *nextEvent) {
    cl_int err;
	cl_event writeEvent;
	cl_event execEvent;

    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;
    int tokens = num_patches + 1;	

	err = clSetKernelArg(POST_PROCESS_KERNEL, 0, sizeof(cl_mem), &input);
	CHECK_ERROR(err);
	err = clSetKernelArg(POST_PROCESS_KERNEL, 1, sizeof(cl_mem), &cls);
	CHECK_ERROR(err);
	err = clSetKernelArg(POST_PROCESS_KERNEL, 2, sizeof(cl_mem), &pos);
	CHECK_ERROR(err);
	err = clSetKernelArg(POST_PROCESS_KERNEL, 3, sizeof(cl_mem), &output);
	CHECK_ERROR(err);
	cl_int imgSize = img_size;
	err = clSetKernelArg(POST_PROCESS_KERNEL, 4, sizeof(cl_int), &imgSize);
	CHECK_ERROR(err);
	cl_int patchSize = patch_size;
	err = clSetKernelArg(POST_PROCESS_KERNEL, 5, sizeof(cl_int), &patchSize);
	CHECK_ERROR(err);
	cl_int embedDim= embed_dim;
	err = clSetKernelArg(POST_PROCESS_KERNEL, 6, sizeof(cl_int), &embedDim);
	CHECK_ERROR(err);

    size_t global_size = tokens* embed_dim;

	err = clEnqueueNDRangeKernel(EXEC_COMMAND_QUEUE, POST_PROCESS_KERNEL, 1, NULL,
		&global_size, NULL, flagCount, flagEvent, nextEvent);
	CHECK_ERROR(err);

	//profileEvents(&execEvent, 1, &CONV_EXEC_TIME);
	CONV_EXEC_COUNT += 1;
}

void layer_norm(cl_mem input, cl_mem output, cl_mem weight, cl_mem bias, cl_event* flagEvents, int flagCount, cl_event* nextFlag)
{
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    cl_int err;
    
    err = clSetKernelArg(LAYERNORM_KERNEL, 0, sizeof(cl_mem), &input);
	CHECK_ERROR(err);
    err = clSetKernelArg(LAYERNORM_KERNEL, 1, sizeof(cl_mem), &weight);
	CHECK_ERROR(err);
	err = clSetKernelArg(LAYERNORM_KERNEL, 2, sizeof(cl_mem), &bias);
	CHECK_ERROR(err);
	err = clSetKernelArg(LAYERNORM_KERNEL, 3, sizeof(cl_mem), &output);
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
		EXEC_COMMAND_QUEUE,
		LAYERNORM_KERNEL,
		2,
		NULL,
		globalSize,
		localSize,
		flagCount, flagEvents,
		nextFlag);
	CHECK_ERROR(err);

    /*profileEvents(writeEvent, 1, &LAYERNORM_WRITE_TIME);
	profileEvents(&execEvent, 1, &LAYERNORM_EXEC_TIME);
	profileEvents(&readEvent, 1, &LAYERNORM_READ_TIME);*/
	LAYERNORM_EXEC_COUNT += 1;
    //printf("Layer Normalization: %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);
}

void multihead_attn(cl_mem input, cl_mem output,cl_mem QKV_BUF, cl_mem ATTN_BUF,
                    cl_mem in_weight, cl_mem in_bias, cl_mem out_weight, cl_mem out_bias, cl_event* flagEvent, int flagCount, cl_event* nextFlag)
{
    int head_dim = embed_dim / num_heads, tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    /*Allocate Q, K, V : tokens * dim*/
    int Q_dim = 0, K_dim = embed_dim, V_dim = embed_dim * 2;
    cl_int err;
    cl_event execEvent[2];

	//clock_t startTime = clock();

   	err = clSetKernelArg(QKV_KERNEL, 0, sizeof(cl_mem), &input);
	CHECK_ERROR(err);
    err = clSetKernelArg(QKV_KERNEL, 1, sizeof(cl_mem), &in_weight);
    CHECK_ERROR(err);
    err = clSetKernelArg(QKV_KERNEL, 2, sizeof(cl_mem), &QKV_BUF);
    CHECK_ERROR(err);
    err = clSetKernelArg(QKV_KERNEL, 3, sizeof(cl_mem), &in_bias);
    CHECK_ERROR(err);
    cl_int rowSize= tokens;
    err = clSetKernelArg(QKV_KERNEL, 4, sizeof(cl_int), &rowSize);
    CHECK_ERROR(err);
    cl_int middleSize = embed_dim; 
    err = clSetKernelArg(QKV_KERNEL, 5, sizeof(cl_int), &middleSize);
	CHECK_ERROR(err);

    int tileSize = 8;
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
        EXEC_COMMAND_QUEUE,
		QKV_KERNEL,
		3,
		NULL,
		global_size,
		local_size,
		flagCount, flagEvent,
		execEvent); 
    CHECK_ERROR(err);
    // --- 
    int print_tokens = tokens < 5 ? tokens : 5;
    int print_dims = embed_dim < 10 ? embed_dim : 10;

    /*Attn 결과를 저장할 버퍼*/
	CHECK_ERROR(err);
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 0, sizeof(cl_mem), &QKV_BUF);
	CHECK_ERROR(err);
	err = clSetKernelArg(QKV_TO_SCOREV_KERNEL, 1, sizeof(cl_mem), &ATTN_BUF);
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
        EXEC_COMMAND_QUEUE,
        QKV_TO_SCOREV_KERNEL,
        3, 
        NULL, 
        global_QKV_TO_SCOREV_size, 
        local_QKV_TO_SCOREV_size, 
        1, 
        execEvent, &execEvent[1]);
    CHECK_ERROR(err);

    // 최종 선형 프로젝션
    err = clSetKernelArg(LL_KERNEL, 0, sizeof(cl_mem), &output);
	CHECK_ERROR(err);
	err = clSetKernelArg(LL_KERNEL, 1, sizeof(cl_mem), &out_weight);
	CHECK_ERROR(err);
	err = clSetKernelArg(LL_KERNEL, 2, sizeof(cl_mem), &ATTN_BUF);
	CHECK_ERROR(err);
	err = clSetKernelArg(LL_KERNEL, 3, sizeof(cl_mem), &out_bias);
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

    tileSize = 8;
    size_t global_LL_size[] = {
        ((tokens + tileSize - 1) / tileSize) * tileSize,        // Round up to multiple of 16
        ((embed_dim + tileSize - 1) / tileSize) * tileSize
    };
    size_t local_LL_size[] = { tileSize, tileSize };

	err = clEnqueueNDRangeKernel(
		EXEC_COMMAND_QUEUE, 
        LL_KERNEL,
		2, 
		NULL, 
		global_LL_size, 
		local_LL_size, 
		1, 
		&execEvent[1], nextFlag);
	CHECK_ERROR(err); 

    //clFinish(READ_COMMAND_QUEUE);

    //profileEvents(writeEvent, 1, &QKV_WRITE_TIME);
    //profileEvents(execEvent, 1, &QKV_EXEC_TIME);
    //profileEvents(execEvent + 1, 1, &QKV_TO_SCORE_EXEC_TIME);
    //profileEvents(execEvent + 2, 1, &QKV_FINAL_LL_EXEC_TIME);
    //profileEvents(&readEvent, 1, &QKV_READ_TIME);

    QKV_EXEC_COUNT += 1;
    //printf("QKV total time: %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);
}


void linear_layer(cl_mem input, cl_mem output, int tokens, int in_features, int out_features, cl_mem weight, cl_mem bias, bool doGelu,
                    cl_event* flagEvent, int flagCount, cl_event* nextFlag) {
    cl_int err;
    
	err = clSetKernelArg(LL_KERNEL, 0, sizeof(cl_mem), &output);
	CHECK_ERROR(err);
    err = clSetKernelArg(LL_KERNEL, 1, sizeof(cl_mem), &weight);
    CHECK_ERROR(err);
    err = clSetKernelArg(LL_KERNEL, 2, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(LL_KERNEL, 3, sizeof(cl_mem), &bias);
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

    int tileSize = 8;
	size_t global_size[] = {
        ((tokens + tileSize - 1) / tileSize) * tileSize,        // Round up to multiple of 16
        ((out_features + tileSize - 1) / tileSize) * tileSize
    };
    size_t local_size[] = { tileSize, tileSize };  // Must match TILE_SIZE

	err = clEnqueueNDRangeKernel(
		EXEC_COMMAND_QUEUE,
		LL_KERNEL,
		2,
		NULL,
		global_size,
		local_size,
		flagCount,
		flagEvent,
		nextFlag);
    CHECK_ERROR(err);

    //profileEvents(&writeEvent, 1, &LL_WRITE_TIME);
    //profileEvents(&execEvent, 1, &LL_EXEC_TIME);
    //profileEvents(&readEvent, 1, &LL_READ_TIME);
    LL_EXEC_COUNT += 1;
    //printf("ll : %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);
}

void mlp_block(cl_mem input, cl_mem output, cl_mem LL_BUF,
    cl_mem fc1_weight, cl_mem fc1_bias, cl_mem fc2_weight, cl_mem fc2_bias, cl_event* firstFlag, cl_event* secondFlag, cl_event* finalFlag)
{
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; // 197
    int Embed_dim = embed_dim;                                            // 768
    int hidden_dim = ((int)(embed_dim * mlp_ratio));                      // 3072

    linear_layer(input, LL_BUF, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias, true, firstFlag, 1, secondFlag); 
    linear_layer(LL_BUF, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias, false,secondFlag, 1, finalFlag);
}

void residual( cl_mem output,cl_mem input, cl_mem toAdd, int row, int col, int flagCount, cl_event* flagEvent, cl_event* nextFlag) {
	cl_int err;

	err = clSetKernelArg(RESIDUAL_KERNEL, 0, sizeof(cl_mem), &input);
	CHECK_ERROR(err);
	err = clSetKernelArg(RESIDUAL_KERNEL, 1, sizeof(cl_mem), &toAdd);
	CHECK_ERROR(err);
    err = clSetKernelArg(RESIDUAL_KERNEL, 2, sizeof(cl_mem), &output);
	CHECK_ERROR(err);

	size_t global_size[] = { row, col};
	err = clEnqueueNDRangeKernel(
		EXEC_COMMAND_QUEUE,
		RESIDUAL_KERNEL,
		2,
		NULL,
		global_size,
		NULL,
		flagCount,
		flagEvent,
		nextFlag);
	CHECK_ERROR(err);
}

////////////////////////////////////// Encoder Architecture //////////////////////////////////////
void Encoder(cl_mem input, cl_mem output, int imageIndex, cl_event* flagEvent, ImageBufferSet bufSet,
             cl_mem ln1_w, cl_mem ln1_b, cl_mem attn_w, cl_mem attn_b, 
             cl_mem attn_out_w, cl_mem attn_out_b, cl_mem ln2_w, cl_mem ln2_b, 
             cl_mem mlp1_w, cl_mem mlp1_b, cl_mem mlp2_w, cl_mem mlp2_b)
{
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; 
    int flagStartIndex = 5 * encoderCount;
    //printf("reached encoder %d %d\n", imageIndex, encoderCount);
    /*LN1*/
    layer_norm(input, bufSet.ln1Out, ln1_w, ln1_b, flagEvent, 1, &ENCODER_FLAGS[flagStartIndex]);
    //if (findNaN(ln1_out, tokens, embed_dim)) printf("ln 1 is nan on %d, encoder:%d\n", __LINE__, encoderCount);
    //printf("reached ln1\n");

    /*Attn*/
    multihead_attn(bufSet.ln1Out, bufSet.mthOut, bufSet.QKV_BUF, bufSet.ATTN_BUF,
        attn_w, attn_b, attn_out_w, attn_out_b, &ENCODER_FLAGS[flagStartIndex], 1,&DO_RESIDUAL[encoderCount]);
    //if (findNaN(ln1_out, tokens, embed_dim)) printf("multihead is nan on %d, encoder:%d\n", __LINE__, encoderCount);
    //printf("reached mth\n");

    residual(bufSet.residual, input, bufSet.mthOut, tokens, embed_dim, 1, &DO_RESIDUAL[encoderCount], &ENCODER_FLAGS[flagStartIndex + 1]);
    //printf("reached rsd1\n");

    /*LN2*/
    layer_norm(bufSet.residual, bufSet.ln2Out, ln2_w, ln2_b, &ENCODER_FLAGS[flagStartIndex + 1], 1, &ENCODER_FLAGS[flagStartIndex + 2]);
    //if (findNaN(ln1_out, tokens, embed_dim)) printf("ln 2 is nan on %d, encoder:%d\n", __LINE__, encoderCount);
    //printf("reached ln2\n");

    /*MLP*/

    mlp_block(bufSet.ln2Out, bufSet.mlpOut, bufSet.LL_BUF, mlp1_w, mlp1_b, mlp2_w, mlp2_b , 
        &ENCODER_FLAGS[flagStartIndex + 2],  &ENCODER_FLAGS[flagStartIndex + 3],&DO_RESIDUAL[1200 + encoderCount]);
    //printf("reached mlp\n");

    /*Residual2*/
    residual(output, bufSet.residual, bufSet.mlpOut, tokens, embed_dim, 1, &DO_RESIDUAL[1200 + encoderCount], &ENCODER_FLAGS[flagStartIndex + 4]);
    //printf("reached rsd2\n");
    
    encoderCount += 1;
}

void Softmax(cl_mem logits, cl_mem softMaxBuf,float *probabilities, int length, int flagCount, cl_event* flagEvent, cl_event* nextFlag)
{
    cl_int err;
    cl_event execEvent;
	err = clSetKernelArg(SOFTMAX_KERNEL, 0, sizeof(cl_mem), &logits);
	CHECK_ERROR(err);
	err = clSetKernelArg(SOFTMAX_KERNEL, 1, sizeof(cl_mem), &softMaxBuf);
	CHECK_ERROR(err);
	cl_int clTokens = length;
	err = clSetKernelArg(SOFTMAX_KERNEL, 2, sizeof(cl_int), &clTokens);
	CHECK_ERROR(err);

	size_t globalSize = 1024;
	size_t localSize = 1024;
    err = clEnqueueNDRangeKernel(
        EXEC_COMMAND_QUEUE,
        SOFTMAX_KERNEL,
        1,
        NULL,
        &globalSize,
        &localSize,
        flagCount, flagEvent,
        &execEvent);
	CHECK_ERROR(err);

     err = clEnqueueReadBuffer(READ_COMMAND_QUEUE, softMaxBuf, CL_FALSE, 0,
        sizeof(float) * length,
        probabilities, 1, &execEvent, nextFlag);
    CHECK_ERROR(err);
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
    float *layer[3];
    float *enc_layer[13];
    float *enc_output;
    int hidden_dim = ((int)(embed_dim * mlp_ratio));
    // printf("%d %d = %d\n", token_size, hidden_dim, token_size * hidden_dim);

    for (int i = 0; i < 3; i++)
    {
        layer[i] = (float *)malloc(sizeof(float) * size[i]);
        if (layer[i] == NULL) printf("malloc failed in line %d\n", __LINE__);
    }
    for (int i = 0; i < 13; i++)
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

    WRITE_COMMAND_QUEUE = clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err);
    CHECK_ERROR(err);
    EXEC_COMMAND_QUEUE = clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err);
	CHECK_ERROR(err);    
    READ_COMMAND_QUEUE = clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err);
    CHECK_ERROR(err);

    cl_command_queue encoderWritingPipe[2] = {
        clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err),
        clCreateCommandQueueWithProperties(CONTEXT, DEVICE, props, &err)
    };


    // for ll
	kernel_source = get_source_code("ll.cl", &kernel_source_size);
	LL_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(LL_PROGRAM, 1, &DEVICE, "", NULL, NULL);
	build_error(LL_PROGRAM, DEVICE, err);
	CHECK_ERROR(err);
	LL_KERNEL = clCreateKernel(LL_PROGRAM, "linear_layer", &err);
	CHECK_ERROR(err);

    // for conv2d
    kernel_source = get_source_code("conv2d.cl", &kernel_source_size);
    CONV2D_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);
    err = clBuildProgram(CONV2D_PROGRAM, 1, &DEVICE, "", NULL, NULL);
    build_error(CONV2D_PROGRAM, DEVICE, err);
    CHECK_ERROR(err);
    CONV2D_KERNEL = clCreateKernel(CONV2D_PROGRAM, "conv2d_kernel", &err);
	CHECK_ERROR(err);
    POST_PROCESS_KERNEL = clCreateKernel(CONV2D_PROGRAM, "postprocess", &err);
	CHECK_ERROR(err);
    
    // for layer_norm 
	kernel_source = get_source_code("layer_norm.cl", &kernel_source_size);
	LAYERNORM_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);
	CHECK_ERROR(clBuildProgram(LAYERNORM_PROGRAM, 1, &DEVICE, "", NULL, NULL));
	build_error(LAYERNORM_PROGRAM, DEVICE, err);
	CHECK_ERROR(err);
    LAYERNORM_KERNEL= clCreateKernel(LAYERNORM_PROGRAM, "layerNorm", &err);
	CHECK_ERROR(err);
    RESIDUAL_KERNEL = clCreateKernel(LAYERNORM_PROGRAM, "encoderResidual", &err);
    CHECK_ERROR(err);
    // for softmax
    kernel_source = get_source_code("miniSoftMax.cl", &kernel_source_size);
	SOFTMAX_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);
	CHECK_ERROR(clBuildProgram(SOFTMAX_PROGRAM, 1, &DEVICE, "", NULL, NULL));
	build_error(SOFTMAX_PROGRAM, DEVICE, err);
	CHECK_ERROR(err);
	SOFTMAX_KERNEL = clCreateKernel(SOFTMAX_PROGRAM, "softMax", &err);
	CHECK_ERROR(err);

    LoadedOnceWeight finalWeights = initLoadedOnceWeight(networks);
    fillLoadedOnceWeight(finalWeights, networks);
    printf("setup time: %.6f sec\n\n" , (double)(clock() - setupTime) / CLK_TCK);

    EncoderWeights encoderWeightBuffers[12];
    for (int i = 0; i< 12; i ++)
		encoderWeightBuffers[i] = initEncoderWeight();
    
    int pipeDepth = 4;
    for (int i = 0; i < 12; i++) {
        int waitCount = min(i * 12, pipeDepth * 12);
        int flagIndex = i < pipeDepth ? 0 : i - pipeDepth;
        cl_event* flagEvent = i == 0 ? NULL : &ENCODER_WEIGHT_WRITE_EVENT[flagIndex * 12];
        fillEncoderWeight(encoderWeightBuffers[i], networks, i, WRITE_COMMAND_QUEUE, waitCount, flagEvent, &ENCODER_WEIGHT_WRITE_EVENT[i * 12]);
    }

    clEnqueueMarkerWithWaitList(EXEC_COMMAND_QUEUE, 12 * 12, ENCODER_WEIGHT_WRITE_EVENT, &START_CONV[0]); // dummy

    for (int i = 0; i < image->n; i++)
    {
        double startTime = clock();
        ImageBufferSet bufferSet = initImageBufferSet();

        /*patch embedding*/
        Conv2d(image[i].data, bufferSet.CONV_BUF, finalWeights.conv_w, finalWeights.conv_b, 1, &START_CONV[i], &START_POST_CONV[i]);
        /*flatten and transpose*/
        postConv2d(bufferSet.CONV_BUF, bufferSet.enc_layer1, finalWeights.cls, finalWeights.pos, 1, &START_POST_CONV[i], &END_PATCHING[i]);
 
        //printf("pre-processing : %.6f sec\n", (double)(clock() - startTime) / CLK_TCK);

        cl_mem enc_layerList[] = { bufferSet.enc_layer1, bufferSet.enc_layer2 };

        /*Encoder - 12 Layers*/
        for (int j = 0; j < 12; j++) {
            cl_event* encoderFlag = j == 0 ? &END_PATCHING[i] : &ENCODER_FLAGS[(encoderCount)*5 - 1];

            Encoder(enc_layerList[j % 2], enc_layerList[(j + 1) % 2], i, encoderFlag, bufferSet,
                encoderWeightBuffers[j].ln1_w, encoderWeightBuffers[j].ln1_b, encoderWeightBuffers[j].attn_w, encoderWeightBuffers[j].attn_b,
                encoderWeightBuffers[j].attn_out_w, encoderWeightBuffers[j].attn_out_b, encoderWeightBuffers[j].ln2_w, encoderWeightBuffers[j].ln2_b,
                encoderWeightBuffers[j].mlp1_w, encoderWeightBuffers[j].mlp1_b, encoderWeightBuffers[j].mlp2_w, encoderWeightBuffers[j].mlp2_b);
        }
        //printf("escaped encoder\n");
        
        layer_norm(enc_layerList[0], enc_layerList[1], finalWeights.ln_w, finalWeights.ln_b, &ENCODER_FLAGS[(i + 1) * 12 * 5 - 1], 1, &FINAL_FLAG[i]); 
        //printf("reached ln3\n");

        cl_event* nextEvent = &START_CONV[i + 1];
        linear_layer(enc_layerList[1], bufferSet.softMaxInputBuf, 1, embed_dim, num_classes, finalWeights.mlp_w, finalWeights.mlp_b, false, &FINAL_FLAG[i], 1, &DO_SOFTMAX[i]);
        //printf("reached ll\n");

        /* 확률분포 추출 */
        Softmax(bufferSet.softMaxInputBuf, bufferSet.softMaxBuf, probabilities[i], num_classes, 1, &DO_SOFTMAX[i], nextEvent);
        //printf("reached softmax\n");
        //if (findNaN(probabilities[i], 1, num_classes)) printf("result has nan\n");
        registerCleanup(START_CONV[i + 1], bufferSet);

        printf("picture #%d: %.6f sec\n\n", i, (double)(clock() - startTime) / CLK_TCK);
    }
    //printEventProfile();

    CHECK_ERROR(clReleaseKernel(LL_KERNEL));
    CHECK_ERROR(clReleaseKernel(RESIDUAL_KERNEL));
    CHECK_ERROR(clReleaseKernel(QKV_KERNEL));
    CHECK_ERROR(clReleaseKernel(QKV_TO_SCOREV_KERNEL));
    CHECK_ERROR(clReleaseKernel(CONV2D_KERNEL));
    CHECK_ERROR(clReleaseKernel(LAYERNORM_KERNEL));
	CHECK_ERROR(clReleaseProgram(MULTIHEAD_PROGRAM));
    CHECK_ERROR(clReleaseProgram(LL_PROGRAM));
    CHECK_ERROR(clReleaseProgram(CONV2D_PROGRAM));
    CHECK_ERROR(clReleaseProgram(LAYERNORM_PROGRAM));
	CHECK_ERROR(clReleaseCommandQueue(WRITE_COMMAND_QUEUE));
    CHECK_ERROR(clReleaseCommandQueue(EXEC_COMMAND_QUEUE));
    CHECK_ERROR(clReleaseCommandQueue(READ_COMMAND_QUEUE));
    releaseLoadedOncelWeights(finalWeights);
    for (int i = 0; i < 12; i++) releaseEncoderWeights(encoderWeightBuffers[i]);
    err = clReleaseContext(CONTEXT);

    CHECK_ERROR(err);
}

void profileEvents(cl_event *events, int eventCount, cl_ulong* timeGlobalVariable) { 
    cl_ulong start, end, minStart = ULLONG_MAX, maxEnd = 0;
    for (int i = 0; i < eventCount; i++) {
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                               sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, 
                               sizeof(cl_ulong), &end, NULL);
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
        printf("  Exec:  %.6f sec (avg: %.6f sec)\n",
               LAYERNORM_EXEC_TIME / 1000000000.0,
               LAYERNORM_EXEC_TIME / 1000000000.0 / LAYERNORM_EXEC_COUNT);
    }
    
    if (QKV_EXEC_COUNT> 0) {
        printf("\nMultiHeadAttention (%d executions):\n", QKV_EXEC_COUNT);
        printf("  Exec(QKV):  %.6f sec (avg: %.6f sec)\n",
               QKV_EXEC_TIME / 1000000000.0,
               QKV_EXEC_TIME / 1000000000.0 / QKV_EXEC_COUNT);
        printf("  Exec(QKV_TO_SCOREV):  %.6f sec (avg: %.6f sec)\n",
			   QKV_TO_SCORE_EXEC_TIME / 1000000000.0,
			   QKV_TO_SCORE_EXEC_TIME / 1000000000.0 / QKV_EXEC_COUNT);
        printf("  Exec(final LL):  %.6f sec (avg: %.6f sec)\n",
			   QKV_FINAL_LL_EXEC_TIME / 1000000000.0,
			   QKV_FINAL_LL_EXEC_TIME / 1000000000.0 / QKV_EXEC_COUNT);
    }
    
    if (LL_EXEC_COUNT > 0) {
        printf("\nLL (%d executions):\n", LL_EXEC_COUNT);
        printf("  Exec:  %.6f sec (avg: %.6f sec)\n",
               LL_EXEC_TIME / 1000000000.0,
               LL_EXEC_TIME / 1000000000.0 / LL_EXEC_COUNT);
    }

	if (CONV_EXEC_COUNT> 0) {
		printf("\nCONV (%d executions):\n", CONV_EXEC_COUNT);
		printf("  Exec:  %.6f sec (avg: %.6f sec)\n",
			   CONV_EXEC_TIME / 1000000000.0,
			   CONV_EXEC_TIME / 1000000000.0 / CONV_EXEC_COUNT);
	}
    cl_ulong allExec= CONV_EXEC_TIME + LL_EXEC_TIME + LAYERNORM_EXEC_TIME + QKV_EXEC_TIME + QKV_TO_SCORE_EXEC_TIME + QKV_FINAL_LL_EXEC_TIME;

    printf("\nTotal Execution Time By Jobs:\n");
	printf("  Total Exec:  %.6f sec \n", allExec / 1000000000.0);

	printf("=============Encoder Weight Write event\n ");
    double offWrite = OFF_WRITE_TIME / 1000000000.0;
    printf("  ENCODER WEIGHT WRITE TIME: %.6f\n", offWrite);
    printf("\n=======================================\n");
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