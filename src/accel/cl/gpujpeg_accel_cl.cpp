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
//#include "../../gpujpeg_util.h"
//#include "../common/gpujpeg_preprocessor_common.h"
#include "../common/gpujpeg_huffman_decoder.h"
#include "../common/gpujpeg_huffman_encoder.h"

#include <string>
#include <cmath>
#include <algorithm>

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

// Keep Device Data gloval
struct DeviceData {
    bool initialize;
    cl_platform_id platform;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
};
DeviceData device_data;

static int init_opencl(const struct gpujpeg_accel* accel, int flags) {
    if (device_data.initialize) return -1;

    cl_uint num_platforms;
    cl_platform_id platform;
    CL_CHECK(clGetPlatformIDs(1, &platform, &num_platforms));

    cl_uint num_devices;
    cl_device_id devices[GPUJPEG_MAX_DEVICE_COUNT];
    memset(devices, 0x0, GPUJPEG_MAX_DEVICE_COUNT * sizeof(cl_device_id));
    
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, GPUJPEG_MAX_DEVICE_COUNT, devices, &num_devices));

    cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);
    cl_device_id device = devices[std::max(accel->device->_id, num_devices - 1)];
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
   
    device_data.platform = platform;
    device_data.context = context;
    device_data.queue = queue;

    // Create kernel
    // cl_kernel kernel = clCreateKernel(NULL, "example_kernel", NULL);
    // deviceData.kernel = kernel;

    device_data.initialize = true;
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
    cl_int err;
    void* addr = clCreateBuffer(device_data.context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) {
        // Handle error
        return nullptr;
    }
    return addr;
}

static void* alloc_host_opencl(size_t size) {
    return malloc(size);
}

static void release_opencl(void* addr) {
    cl_mem mem = (cl_mem)addr;
    clReleaseMemObject(mem);
}

static void release_host_opencl(void* addr) {
    free(addr);
}

static void memset_opencl(void* addr, uint8_t value, size_t count) {
    CL_CHECK(clEnqueueFillBuffer(device_data.queue, (cl_mem)addr, &value, sizeof(uint8_t), 0, count, 0, NULL, NULL));
}

static void memcpy_opencl(void* dstAddr, void* srcAddr, size_t count, int kind) {
    switch (kind) {
        case 0 /*HostToHost*/:
            ::memcpy(reinterpret_cast<void*>(dstAddr), reinterpret_cast<void*>(srcAddr), count);
            break;
        case 1 /*HostToDevice*/:
            void(clEnqueueWriteBuffer(device_data.queue, reinterpret_cast<cl_mem>(dstAddr), CL_TRUE, 0, count, reinterpret_cast<void*>(srcAddr), 0, NULL, NULL)); // C-Style
            break;
        case 2 /*DeviceToHost*/:
            void(clEnqueueReadBuffer(device_data.queue, reinterpret_cast<cl_mem>(srcAddr), CL_TRUE, 0, count, reinterpret_cast<void*>(dstAddr), 0, NULL, NULL)); // C-Style
            break;
        case 3 /*DeviceToDevice*/:
            void(clEnqueueCopyBuffer(device_data.queue, reinterpret_cast<cl_mem>(srcAddr), reinterpret_cast<cl_mem>(dstAddr), 0, 0, count, 0, NULL, NULL)); // C-Style
            break;
        default:
            // throw exceptions::AccelerateOpenGLCopyKindException();
            break;
    }
}

static void memcpy_async_opencl(void* dstAddr, void* srcAddr, size_t count, int kind, void* stream) {
    switch (kind) {
        case 0 /*HostToHost*/:
            ::memcpy(reinterpret_cast<void*>(dstAddr), reinterpret_cast<void*>(srcAddr), count);
            break;
        case 1 /*HostToDevice*/:
            void(clEnqueueWriteBuffer(device_data.queue, reinterpret_cast<cl_mem>(dstAddr), CL_TRUE, 0, count, reinterpret_cast<void*>(srcAddr), 0, NULL, NULL)); // C-Style
            break;
        case 2 /*DeviceToHost*/:
            void(clEnqueueReadBuffer(device_data.queue, reinterpret_cast<cl_mem>(srcAddr), CL_TRUE, 0, count, reinterpret_cast<void*>(dstAddr), 0, NULL, NULL)); // C-Style
            break;
        case 3 /*DeviceToDevice*/:
            void(clEnqueueCopyBuffer(device_data.queue, reinterpret_cast<cl_mem>(srcAddr), reinterpret_cast<cl_mem>(dstAddr), 0, 0, count, 0, NULL, NULL)); // C-Style
            break;
        default:
            // throw exceptions::AccelerateOpenGLCopyKindException();
            break;
    }
}

static void synchronise_opencl() {
    CL_CHECK(clFinish(device_data.queue));
}

static void synchronise_stream_opencl(void* stream) {
    CL_CHECK(clFinish(device_data.queue));
}

static int dct_opencl(struct gpujpeg_encoder* encoder) {
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    // // Encode each component
    // for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
    //     // Get component
    //     struct gpujpeg_component* component = &coder->component[comp];

    //     // Get quantization table
    //     enum gpujpeg_component_type type = encoder->coder.component[comp].type;
    //     const float* const d_quantization_table = encoder->table_quantization[type].d_table_forward;

    //     int roi_width = component->data_width;
    //     int roi_height = component->data_height;
    //     assert(GPUJPEG_BLOCK_SIZE == 8);

    //     int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
    //     int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

    //     const int WARP_COUNT = 4;

    //     // Perform block-wise DCT processing
    //     size_t global_work_size[2] = { gpujpeg_div_and_round_up(block_count_x, 4), gpujpeg_div_and_round_up(block_count_y, WARP_COUNT) };
    //     size_t local_work_size[2] = { 4 * 8, WARP_COUNT };

    //     // Call the OpenCL kernel
    //     cl_kernel kernel = // Initialize the kernel using clCreateKernel

    //     clSetKernelArg(kernel, 0, sizeof(int), &block_count_x);
    //     clSetKernelArg(kernel, 1, sizeof(int), &block_count_y);
    //     clSetKernelArg(kernel, 2, sizeof(cl_mem), &component->d_data);
    //     clSetKernelArg(kernel, 3, sizeof(int), &component->data_width);
    //     clSetKernelArg(kernel, 4, sizeof(cl_mem), &component->d_data_quantized);
    //     clSetKernelArg(kernel, 5, sizeof(int), &component->data_width * GPUJPEG_BLOCK_SIZE);
    //     clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_quantization_table);

    //     clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    // }

    return 0;
}

static int idct_opencl(struct gpujpeg_decoder* decoder) {
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // // Encode each component
    // for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
    //     // Get component
    //     struct gpujpeg_component* component = &coder->component[comp];

    //     int roi_width = component->data_width;
    //     int roi_height = component->data_height;
    //     assert(GPUJPEG_BLOCK_SIZE == 8);

    //     int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
    //     int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

    //     // Get quantization table
    //     uint16_t* d_quantization_table = decoder->table_quantization[decoder->comp_table_quantization_map[comp]].d_table;

    //     cl_mem d_quantization_table_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 64 * sizeof(uint16_t), d_quantization_table, &err);
    //     assert(err == CL_SUCCESS);

    //     cl_kernel kernel = // Initialize the kernel using clCreateKernel
    //     clSetKernelArg(kernel, 0, sizeof(cl_mem), &component->d_data_quantized);
    //     clSetKernelArg(kernel, 1, sizeof(cl_mem), &component->d_data);
    //     clSetKernelArg(kernel, 2, sizeof(int), &component->data_width);
    //     clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_quantization_table_cl);

    //     clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    //     clFinish(queue);

    //     clReleaseMemObject(d_quantization_table_cl);
    // }

    return 0;
}

static struct gpujpeg_huffman_decoder* huffman_decoder_init_opencl(void) {
    struct gpujpeg_huffman_decoder *huffman_decoder = (struct gpujpeg_huffman_decoder *) calloc(1, sizeof(struct gpujpeg_huffman_decoder));
    
    cl_int err = CL_SUCCESS;
    #ifdef HUFFMAN_CONST_TABLES
        NOT_YET_IMPLEMENTED
    #else
        huffman_decoder->d_order_natural = reinterpret_cast<int*>(clCreateBuffer(device_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int), (void*)gpujpeg_order_natural, &err));
        assert(err == CL_SUCCESS);
    #endif
    
    huffman_decoder->d_tables_full = reinterpret_cast<uint16_t*>(clCreateBuffer(device_data.context, CL_MEM_READ_WRITE, 4 * (1 << 16) * sizeof(uint16_t), NULL, &err));
    assert(err == CL_SUCCESS);
    huffman_decoder->d_tables_quick = reinterpret_cast<uint16_t*>(clCreateBuffer(device_data.context, CL_MEM_READ_WRITE, QUICK_TABLE_ITEMS * sizeof(uint16_t), NULL, &err));
    assert(err == CL_SUCCESS);

    return huffman_decoder;
}

static void huffman_decoder_destroy_opencl(struct gpujpeg_huffman_decoder* decoder) {
    if (decoder == NULL) {
        return;
    }

    CL_CHECK(clReleaseMemObject((cl_mem)decoder->d_order_natural));
    CL_CHECK(clReleaseMemObject((cl_mem)decoder->d_tables_full));
    CL_CHECK(clReleaseMemObject((cl_mem)decoder->d_tables_quick));

    free(decoder);
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

static void decoder_destroy_opencl(struct gpujpeg_decoder* decoder) {
    for (int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++) {
        if (decoder->table_quantization[comp_type].d_table != NULL) {
            CL_CHECK(clReleaseMemObject((cl_mem)decoder->table_quantization[comp_type].d_table));
        }
    }

    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            CL_CHECK(clReleaseMemObject((cl_mem)decoder->d_table_huffman[comp_type][huff_type]));
        }
    }

    if (decoder->reader != NULL) {
        free(decoder->reader);
    }

    if (decoder->huffman_decoder != NULL) {
        huffman_decoder_destroy_opencl(decoder->huffman_decoder);
    }
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

static int create_timer_opencl(struct gpujpeg_timer* timer) {
    if (timer != NULL) {
        cl_int cl_status;

        timer->start = clCreateUserEvent(device_data.context, &cl_status);
        if (cl_status != CL_SUCCESS) {
            return -1;
        }

        timer->stop = clCreateUserEvent(device_data.context, &cl_status);
        if (cl_status != CL_SUCCESS) {
            clReleaseEvent((cl_event)timer->start);
            return -1;
        }

        timer->started = 0;
        return 0;
    } else {
        return -1;
    }
}

static int destroy_timer_opencl(struct gpujpeg_timer* timer) {
    cl_int err_action = clReleaseEvent((cl_event)timer->start);
    if (err_action != CL_SUCCESS) {
        return -1;
    }
    
    err_action = clReleaseEvent((cl_event)timer->stop);
    if (err_action != CL_SUCCESS) {
        return -1;
    }
    return 0;
}

static int start_timer_opencl(struct gpujpeg_timer* timer, int record_perf, void* stream) {
    if (record_perf) {
        timer->started = 1;
        
        cl_event clstart = (cl_event)timer->start;
        cl_int err_action = clEnqueueMarker(device_data.queue, &clstart);
        if (err_action != CL_SUCCESS) {
           return -1;
        }
    } else {
        timer->started = -1;
    }

    return 0;

}

static void stop_timer_opencl(struct gpujpeg_timer* timer, int record_perf, void* stream) {
    if (record_perf) {
        timer->started = 0;
        cl_event clstop = (cl_event)timer->stop;
        cl_int err_action = clEnqueueMarker(device_data.queue, &clstop);
        clWaitForEvents(1, &clstop);
        if (err_action != CL_SUCCESS) {
           return;
        }
    }
}

static double elapsed_timer_opencl(struct gpujpeg_timer* timer) {
    if (timer != NULL) {
        cl_ulong start_time, stop_time;

        clGetEventProfilingInfo((cl_event)timer->start, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start_time, NULL);
        clGetEventProfilingInfo((cl_event)timer->stop, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &stop_time, NULL);

        return (stop_time - start_time) * 1.0e-6; // Convert nanoseconds to milliseconds
    } else {
        return -1.0; // Return negative value to indicate error
    }
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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

#ifdef __cplusplus
}
#endif // __cplusplus
