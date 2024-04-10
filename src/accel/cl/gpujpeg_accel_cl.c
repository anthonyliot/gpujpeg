/*
 * Copyright (c) 2020, CESNET z.s.p.o
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpujpeg_accel_cl.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define CL_CHECK(stmt)                                                                     \
    do {                                                                                   \
        cl_int err = stmt;                                                                 \
        if (err != CL_SUCCESS) {                                                           \
            fprintf(stderr, "[GPUJPEG] [Error] OpenCL error %08x (%d), at %s:%d for %s\n", \
                    err, err, __FILE__, __LINE__, #stmt);                                  \
        }                                                                                  \
    } while(0)

static int init_opencl(const struct gpujpeg_accel* accel, int flags) {
    //NOT_YET_IMPLEMENTED
    return 0;
}

static void get_compute_capabilities_opencl(const struct gpujpeg_accel* accel, int* compute_major, int* compute_minor) {

    // TODO: Fix the above code to get the compute capabilities of the device
    *compute_major = 0;
    *compute_minor = 0;
}

static int get_info_opencl(const struct gpujpeg_accel* accel, struct gpujpeg_devices_info* devices_info) {

    cl_int status;
    cl_uint num_platforms;
    cl_platform_id platform;
    CL_CHECK(clGetPlatformIDs(1, &platform, &num_platforms));

    cl_uint num_devices;
    cl_device_id devices[GPUJPEG_MAX_DEVICE_COUNT];
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, GPUJPEG_MAX_DEVICE_COUNT, devices, &num_devices));

    devices_info->device_count = num_devices;

    for (cl_uint device_id = 0; device_id < num_devices; device_id++) {
        struct gpujpeg_device_info* device_info = &(devices_info->device[device_id]);
        device_info->id = device_id;

        cl_device_id device = devices[device_id];
        char device_name[256];
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL));
        strncpy(device_info->name, device_name, sizeof device_info->name);
        get_compute_capabilities_opencl(accel, &device_info->major, &device_info->minor);

        cl_uint device_compute_units;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_compute_units), &device_compute_units, NULL));
        devices_info->device[device_id].multiprocessor_count = device_compute_units;

        cl_ulong device_global_mem_size;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_global_mem_size), &device_global_mem_size, NULL));
        devices_info->device[device_id].global_memory = device_global_mem_size;

        cl_ulong device_constant_mem_size;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(device_constant_mem_size), &device_constant_mem_size, NULL));
        devices_info->device[device_id].constant_memory = device_constant_mem_size;

        size_t device_shared_mem_size;
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_shared_mem_size), &device_shared_mem_size, NULL));
        devices_info->device[device_id].shared_memory = device_shared_mem_size;

        // cl_uint device_max_regs;
        // CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_REGISTERS_PER_BLOCK, sizeof(device_max_regs), &device_max_regs, NULL));
        // devices_info->device[device_id].register_count = device_max_regs;

        devices_info->device[device_id].register_count = 0;
    }
    return 0;
}

static void* alloc_opencl(size_t size) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static void* alloc_host_opencl(size_t size) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static void release_opencl(void* addr) {
    NOT_YET_IMPLEMENTED
}

static void release_host_opencl(void* addr) {
    NOT_YET_IMPLEMENTED
}

static void memset_opencl(void* addr, uint8_t value, size_t count) {
    NOT_YET_IMPLEMENTED
}

static void memcpy_opencl(void* dstAddr, void* srcAddr, size_t count, int kind) {
    NOT_YET_IMPLEMENTED
}

static void memcpy_async_opencl(void* dstAddr, void* srcAddr, size_t count, int kind, void* stream) {
    NOT_YET_IMPLEMENTED
}

static void synchronise_opencl() {
    NOT_YET_IMPLEMENTED
}

static void synchronise_stream_opencl(void* stream) {
    NOT_YET_IMPLEMENTED
}

static int dct_opencl(struct gpujpeg_encoder* encoder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int idct_opencl(struct gpujpeg_decoder* decoder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static void decoder_destroy_opencl(struct gpujpeg_decoder* decoder) {
    NOT_YET_IMPLEMENTED
}

static struct gpujpeg_huffman_decoder* huffman_decoder_init_opencl(void) {
    NOT_YET_IMPLEMENTED
    return NULL;
}

static void huffman_decoder_destroy_opencl(struct gpujpeg_huffman_decoder* decoder) {
    NOT_YET_IMPLEMENTED
}

static int huffman_decoder_decode_opencl(struct gpujpeg_decoder* decoder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static struct gpujpeg_huffman_encoder* huffman_encoder_init_opencl(struct gpujpeg_encoder* encoder) {
    NOT_YET_IMPLEMENTED
    return NULL;
}

static void huffman_encoder_destroy_opencl(struct gpujpeg_huffman_encoder* huffman_encoder) {
    NOT_YET_IMPLEMENTED
}

static int huffman_encoder_encode_opencl(struct gpujpeg_encoder* encoder, struct gpujpeg_huffman_encoder* huffman_encoder, unsigned int* output_byte_count) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int preprocessor_decoder_init_opencl(struct gpujpeg_coder* coder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int preprocessor_decode_opencl(struct gpujpeg_coder* coder, void* stream) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int preprocessor_encoder_init_opencl(struct gpujpeg_coder* coder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int preprocessor_encode_opencl(struct gpujpeg_encoder* encoder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int create_timer_opencl(struct gpujpeg_timer* name) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int destroy_timer_opencl(struct gpujpeg_timer* name) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static int start_timer_opencl(struct gpujpeg_timer* name, int record_perf, void* stream) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static void stop_timer_opencl(struct gpujpeg_timer* name, int record_perf, void* stream) {
    NOT_YET_IMPLEMENTED
}

static double elapsed_timer_opencl(struct gpujpeg_timer* timer) {
    NOT_YET_IMPLEMENTED
    return -1.0; // Return negative value to indicate error
}

struct gpujpeg_accel* gpujpeg_accel_opencl_create(const struct gpujpeg_device* device) {
    struct gpujpeg_accel_opencl* accel_opencl = (struct gpujpeg_accel_opencl*)malloc(sizeof(struct gpujpeg_accel_opencl));
    if (!accel_opencl) {
        fprintf(stderr, "[GPUJPEG] Error allocating memory for gpujpeg_accel_opencl\n");
        return NULL;
    }

    accel_opencl->base.device = device;
    accel_opencl->base.init = init_opencl;
    accel_opencl->base.get_info = get_info_opencl;
    accel_opencl->base.get_compute_capabilities = get_compute_capabilities_opencl;
    accel_opencl->base.alloc = alloc_opencl;
    accel_opencl->base.alloc_host = alloc_host_opencl;
    accel_opencl->base.release = release_opencl;
    accel_opencl->base.release_host = release_host_opencl;
    accel_opencl->base.memoryset = memset_opencl;
    accel_opencl->base.memorycpy = memcpy_opencl;
    accel_opencl->base.memorycpy_async = memcpy_async_opencl;
    accel_opencl->base.synchronise = synchronise_opencl;
    accel_opencl->base.synchronise_stream = synchronise_stream_opencl;
    accel_opencl->base.dct = dct_opencl;
    accel_opencl->base.idct = idct_opencl;
    accel_opencl->base.decoder_destroy = decoder_destroy_opencl;
    accel_opencl->base.huffman_decoder_init = huffman_decoder_init_opencl;
    accel_opencl->base.huffman_encoder_init = huffman_encoder_init_opencl;
    accel_opencl->base.huffman_decoder_destroy = huffman_decoder_destroy_opencl;
    accel_opencl->base.huffman_encoder_destroy = huffman_encoder_destroy_opencl;
    accel_opencl->base.huffman_decoder_decode = huffman_decoder_decode_opencl;
    accel_opencl->base.huffman_encoder_encode = huffman_encoder_encode_opencl;
    accel_opencl->base.preprocessor_decoder_init = preprocessor_decoder_init_opencl;
    accel_opencl->base.preprocessor_decode = preprocessor_decode_opencl;
    accel_opencl->base.preprocessor_encoder_init = preprocessor_encoder_init_opencl;
    accel_opencl->base.preprocessor_encode = preprocessor_encode_opencl;

    // Timer specific
    accel_opencl->base.timer.create_timer = create_timer_opencl;
    accel_opencl->base.timer.destroy_timer = destroy_timer_opencl;
    accel_opencl->base.timer.start_timer = start_timer_opencl;
    accel_opencl->base.timer.stop_timer = stop_timer_opencl;
    accel_opencl->base.timer.elapsed_timer = elapsed_timer_opencl;

    return &accel_opencl->base;
}
