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

// Define some naming not required for cpu
#define __constant__    
#define __global__
#define __device__
#define __shared__
#define __CUDA_ARCH__ 200
#define CUDART_VERSION 9000
#define __launch_bounds__(X, Y)
#define __umulhi(numerator, pre_div_mul) ((unsigned long long)(numerator) * (unsigned long long)(pre_div_mul) >> 32)


#ifdef GPUJPEG_USE_OPENMP
#include <omp.h>
#define PRAGMA _Pragma
#else
#define PRAGMA(X)
#endif

#include <chrono>
#include <cmath>
#include <algorithm>
#include <mutex>

// Define a mutex
std::mutex mutex;

#define MAX std::max
#define ROUND std::round

// Simulated function to count leading zeros
int __clz(unsigned int x) {
    int count = 0;
    while ((x & (1u << 31)) == 0 && count < 32) {
        x <<= 1;
        count++;
    }
    return count;
}

// Simulated function to emulate ballot operation
unsigned int __ballot_sync(unsigned int mask, int value) {
    unsigned int result = 0;
    if (value != 0)
        result = (1 << value) & mask;
    return result;
}

// Function to count the number of set bits in an integer
int __popc(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

// CPU-supported equivalent of atomicAdd
unsigned int atomicAdd(unsigned int* address, unsigned int val) {
    std::lock_guard<std::mutex> lock(mutex); // Lock the mutex to ensure atomicity
    unsigned int oldVal = *address;
    *address += val;
    return oldVal;
}

// Create dim3 structure
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1) : x(_x), y(_y), z(_z) {}
};

struct uchar4 {
    unsigned char x, y, z, w;
    uchar4(unsigned char _x = 0, unsigned char _y = 0, unsigned char _z = 0, unsigned char _w = 0) : x(_x), y(_y), z(_z), w(_w) {}
};

struct int4 {
    int32_t x, y, z, w;
    int4(int32_t _x = 0, int32_t _y = 0, int32_t _z = 0, int32_t _w = 0) : x(_x), y(_y), z(_z), w(_w) {}
};

struct uint4 {
    uint32_t x, y, z, w;
    uint4(uint32_t _x = 0, uint32_t _y = 0, uint32_t _z = 0, uint32_t _w = 0) : x(_x), y(_y), z(_z), w(_w) {}
};
static uint4 make_uint4(uint32_t _x = 0, uint32_t _y = 0, uint32_t _z = 0, uint32_t _w = 0)
{
    uint4 elt;
    elt.x = _x;
    elt.y = _y;
    elt.z = _z;
    elt.w = _w;
    return elt;
}

#define GRID_BLOCK ,dim3 gridDim ,dim3 blockDim
#define GRID_BLOCK_NO_COMMA dim3 gridDim ,dim3 blockDim

// We close all the for loop of the thread to ensure the sync
#define __syncthreads()                                                                  \
                    }                                                                    \
                }                                                                        \
            }                                                                            \
            PRAGMA("omp parallel for collapse(3)")                                       \
            for (int threadIdx_x = 0; threadIdx_x < blockDim.x; threadIdx_x++) {         \
                for (int threadIdx_y = 0; threadIdx_y < blockDim.y; threadIdx_y++) {     \
                    for (int threadIdx_z = 0; threadIdx_z < blockDim.z; threadIdx_z++) { \
                        dim3 threadIdx(threadIdx_x, threadIdx_y, threadIdx_z);

#define LOOP_KERNEL_BEGIN                                                                    \
    PRAGMA("omp parallel for collapse(3)")                                                   \
    for (int blockIdx_x = 0; blockIdx_x < gridDim.x; blockIdx_x++) {                         \
        for (int blockIdx_y = 0; blockIdx_y < gridDim.y; blockIdx_y++) {                     \
            for (int blockIdx_z = 0; blockIdx_z < gridDim.z; blockIdx_z++) {                 \
                dim3 blockIdx(blockIdx_x, blockIdx_y, blockIdx_z);                           \
                PRAGMA("omp parallel for collapse(3)")                                       \
                for (int threadIdx_x = 0; threadIdx_x < blockDim.x; threadIdx_x++) {         \
                    for (int threadIdx_y = 0; threadIdx_y < blockDim.y; threadIdx_y++) {     \
                        for (int threadIdx_z = 0; threadIdx_z < blockDim.z; threadIdx_z++) { \
                            dim3 threadIdx(threadIdx_x, threadIdx_y, threadIdx_z);


#define LOOP_KERNEL_END     \
                        }   \
                    }       \
                }           \
            }               \
        }                   \
	}


#include "gpujpeg_accel_cpu.h"
#include "../../gpujpeg_util.h"
#include "../common/gpujpeg_preprocessor_common.h"
#include "../common/gpujpeg_huffman_decoder.h"
#include "../common/gpujpeg_huffman_encoder.h"

// Kernels 
namespace cpu {
    #include "../common/gpujpeg_idct.kernel"
    #include "../common/gpujpeg_dct.kernel"
    #include "../common/gpujpeg_preprocessor.kernel"
    #include "../common/gpujpeg_postprocessor.kernel"
    #include "../common/gpujpeg_huffman_encoder.kernel"
}

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static int init_cpu(const struct gpujpeg_accel* accel, int flags) {
    // TODO: AL Nothing to do ?
    return 0;
}

static int get_info_cpu(const struct gpujpeg_accel* accel, struct gpujpeg_devices_info* devices_info) {

    devices_info->device_count = 1;

    // TODO: AL TO COMPLETE
    // Maybe we should use OpenMP for CPU
    for ( int device_id = 0; device_id < devices_info->device_count; device_id++ ) {
        struct gpujpeg_device_info* device_info = &(devices_info->device[device_id]);
        device_info->id = device_id;
        strncpy(device_info->name, accel->device->_device, sizeof device_info->name);
        device_info->major = 0;
        device_info->minor = 0;
        device_info->global_memory = 0;
        device_info->constant_memory = 0;
        device_info->shared_memory = 0;
        device_info->register_count = 0;
    }

    return 0;
}

static void get_compute_capabilities_cpu(const struct gpujpeg_accel* accel, int* compute_major, int* compute_minor) {
    // TODO
    *compute_major = 2;
    *compute_minor = 0;
}

static void* alloc_cpu(size_t size) {
    void* addr = malloc(size);
    return (void*)addr;
}

static void* alloc_host_cpu(size_t size) {
    void* addr = malloc(size);
    return (void*)addr;
}

static void release_cpu(void* addr) {
    free((void*)addr);
}

static void release_host_cpu(void* addr) {
    free((void*)addr);
}

static void memset_cpu(void* addr, uint8_t value, size_t count) {
    memset((void*)addr, value, count);
}

static void memcpy_cpu(void* dstAddr, void* srcAddr, size_t count, int kind) {
    memcpy((void*)dstAddr, (void*)srcAddr, count);
}

static void memcpy_async_cpu(void* dstAddr, void* srcAddr, size_t count, int kind, void* stream) {
    // TODO:
    // A correct equivalent should use std::thread + stream ti stre the std::thread
    // std::thread copy_thread(memcpy_cpu, dstAddr, srcAddr, count);
    memcpy((void*)dstAddr, (void*)srcAddr, count);
}

static void memcpy_2d_async_cpu(void* dstAddr, int dpitch, void* srcAddr, int spitch, int width, int height, int kind, void* stream) {
    for (int i = 0; i < height; ++i) {
        memcpy_async_cpu((void*)((uint8_t*)dstAddr + i * dpitch), (void*)((uint8_t*)srcAddr + i * spitch), width, kind, stream);
    }
}

static void synchronise_cpu() {
    NOT_YET_IMPLEMENTED
}

static void synchronise_stream_cpu(void* stream) {
    // TODO:
    // If we create a cpuStream similar to cudaStream we could use that to join
    // all async thread    
}

static int dct_cpu(struct gpujpeg_encoder* encoder) {
// Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    cudaStream_t stream = (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL;

    // Encode each component
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        // Get quantization table
        enum gpujpeg_component_type type = encoder->coder.component[comp].type;
        const float* const d_quantization_table = encoder->table_quantization[type].d_table_forward;

        // copy the quantization table into constant memory for devices of CC < 2.0
        if( encoder->coder.compute_major < 2 ) {
            memcpy(cpu::gpujpeg_dct_quantization_table_const, d_quantization_table, sizeof(cpu::gpujpeg_dct_quantization_table_const));
        }

        int roi_width = component->data_width;
        int roi_height = component->data_height;
        assert(GPUJPEG_BLOCK_SIZE == 8);

        int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
        int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

        enum { WARP_COUNT = 4 };

        // Perform block-wise DCT processing
        dim3 dct_grid(
            gpujpeg_div_and_round_up(block_count_x, 4),
            gpujpeg_div_and_round_up(block_count_y, WARP_COUNT),
            1
        );

        dim3 dct_block(4 * 8, WARP_COUNT);
        cpu::gpujpeg_dct_kernel<WARP_COUNT>(
            block_count_x,
            block_count_y,
            component->d_data,
            component->data_width,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            d_quantization_table,
            dct_grid,
            dct_block
        );
    }

    return 0;
}

static int idct_cpu(struct gpujpeg_decoder* decoder) {
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Encode each component
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        int roi_width = component->data_width;
        int roi_height = component->data_height;
        assert(GPUJPEG_BLOCK_SIZE == 8);

        int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
        int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

        // Get quantization table
        uint16_t* d_quantization_table = decoder->table_quantization[decoder->comp_table_quantization_map[comp]].d_table;

        // Copy quantization table to constant memory
        memcpy_async_cpu((void*)cpu::gpujpeg_idct_quantization_table, (void*)d_quantization_table, 64 * sizeof(uint16_t), 3 /*DeviceToDevice*/, decoder->stream);

        dim3 grid(
            gpujpeg_div_and_round_up(block_count_x * block_count_y, (GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_Z) / GPUJPEG_BLOCK_SIZE), 
            1
        );
        dim3 threads(
            GPUJPEG_IDCT_BLOCK_X, 
            GPUJPEG_IDCT_BLOCK_Y, 
            GPUJPEG_IDCT_BLOCK_Z
        );

        // <<<grid, threads, 0, decoder->stream>>>
#ifdef GPUJPEG_USE_OPENMP
        printf("Number of threads = %d - ", omp_get_num_threads());
#endif        
        printf("Running cpu kernel (idct_cpu): %dx%dx%d - %dx%dx%d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);
        cpu::gpujpeg_idct_kernel(
            component->d_data_quantized,
            component->d_data,
            component->data_width,
            d_quantization_table,
            grid,
            threads
        );
    }
    return 0;
}

static struct gpujpeg_huffman_decoder* huffman_decoder_init_cpu(void) {
    struct gpujpeg_huffman_decoder *huffman_decoder = (struct gpujpeg_huffman_decoder *) calloc(1, sizeof(struct gpujpeg_huffman_decoder));

    // Allocate memory for order_natural array and copy data
    huffman_decoder->d_order_natural = (int*)malloc(GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int));
    memcpy(huffman_decoder->d_order_natural, gpujpeg_order_natural, GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int));
    
    // Allocate memory for tables_full and tables_quick arrays
    huffman_decoder->d_tables_full = (uint16_t*)malloc(4 * (1 << 16) * sizeof(uint16_t));
    huffman_decoder->d_tables_quick = (uint16_t*)malloc(QUICK_TABLE_ITEMS * sizeof(uint16_t));

    return huffman_decoder;
}

static void huffman_decoder_destroy_cpu(struct gpujpeg_huffman_decoder* decoder) {
    if (decoder == NULL) {
        return;
    }

    free(decoder->d_order_natural);
    free(decoder->d_tables_full);
    free(decoder->d_tables_quick);
    free(decoder);
}

static void huffman_encoder_destroy_cpu(struct gpujpeg_huffman_encoder* encoder) {
    if (encoder == NULL) {
        return;
    }
    
    if (encoder->d_gpujpeg_huffman_output_byte_count != NULL) {
        free(encoder->d_gpujpeg_huffman_output_byte_count);
    }

    free(encoder);
}

static void encoder_destroy_cpu(struct gpujpeg_encoder* encoder) {

    if (encoder->huffman_encoder != NULL) {
        huffman_encoder_destroy_cpu(encoder->huffman_encoder);
    }

    for (int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++) {
        if (encoder->table_quantization[comp_type].d_table != NULL) {
            free(encoder->table_quantization[comp_type].d_table);
        }
        if (encoder->table_quantization[comp_type].d_table_forward != NULL) {
            free(encoder->table_quantization[comp_type].d_table_forward);
        }
    }

    if (encoder->writer != NULL) {
        gpujpeg_writer_destroy(encoder->writer);
    }
}

static void decoder_destroy_cpu(struct gpujpeg_decoder* decoder) {
    for (int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++) {
        if (decoder->table_quantization[comp_type].d_table != NULL) {
            free(decoder->table_quantization[comp_type].d_table);
        }
    }

    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            free(decoder->d_table_huffman[comp_type][huff_type]);
        }
    }

    if (decoder->reader != NULL) {
        free(decoder->reader);
    }

    if (decoder->huffman_decoder != NULL) {
        huffman_decoder_destroy_cpu(decoder->huffman_decoder);
    }
}


static int huffman_decoder_decode_cpu(struct gpujpeg_decoder* decoder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static struct gpujpeg_huffman_encoder* huffman_encoder_init_cpu(struct gpujpeg_encoder* encoder) {
struct gpujpeg_huffman_encoder * huffman_encoder = (struct gpujpeg_huffman_encoder *) malloc(sizeof(struct gpujpeg_huffman_encoder));
    if ( huffman_encoder == NULL ) {
        return NULL;
    }
    memset(huffman_encoder, 0, sizeof(struct gpujpeg_huffman_encoder));

    // Allocate
    huffman_encoder->d_gpujpeg_huffman_output_byte_count = (unsigned int *)malloc(sizeof(unsigned int));

    // Initialize decomposition lookup table
    dim3 grid(32);
    dim3 threads(256);
#ifdef GPUJPEG_USE_OPENMP
    printf("Number of threads = %d - ", omp_get_num_threads());
#endif
    printf("Running cpu kernel (huffman_encoder_value_decomposition_init_kernel): %dx%dx%d - %dx%dx%d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);
    cpu::gpujpeg_huffman_encoder_value_decomposition_init_kernel(grid, threads);  // 8192 threads total
    synchronise_stream_cpu(encoder->stream);

    // compose GPU version of the huffman LUT and copy it into GPU memory (for CC >= 2.0)
    uint32_t gpujpeg_huffman_cpu_lut[(256 + 1) * 4];
    cpu::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 0, &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC], true);
    cpu::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 1, &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC], false);
    cpu::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 2, &encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC], true);
    cpu::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 3, &encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC], false);
    
    // Copy data to CPU-side symbol
    memcpy(cpu::gpujpeg_huffman_lut, gpujpeg_huffman_cpu_lut, (256 + 1) * 4 * sizeof(*cpu::gpujpeg_huffman_lut));

    // Copy Huffman coding table to GPU memory
    memcpy(cpu::gpujpeg_huffman_encoder_table_huffman, &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC], sizeof(cpu::gpujpeg_huffman_encoder_table_huffman));

    // Copy natural order to constant device memory
    memcpy(cpu::gpujpeg_huffman_encoder_order_natural, gpujpeg_order_natural, GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int));

    return huffman_encoder;
}

static int huffman_encoder_encode_cpu(struct gpujpeg_encoder* encoder, struct gpujpeg_huffman_encoder* huffman_encoder, unsigned int* output_byte_count) {
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    assert(coder->param.restart_interval > 0);

    // Select encoder kernel which either expects continuos segments of blocks or uses block lists
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param_image.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);

    // Select encoder kernel based on compute capability
    if ( encoder->coder.compute_major < 2 ) {
        // Run kernel
        dim3 thread(THREAD_BLOCK_SIZE);
        dim3 grid(gpujpeg_div_and_round_up(coder->segment_count, thread.x));
        cpu::gpujpeg_huffman_encoder_encode_kernel(
            coder->d_component,
            coder->d_segment,
            comp_count,
            coder->segment_count,
            coder->d_temp_huffman,
            huffman_encoder->d_gpujpeg_huffman_output_byte_count,
            grid,
            thread
        );
    } else {
        // Run encoder kernel
        dim3 thread(32 * WARPS_NUM);
        dim3 grid = cpu::gpujpeg_huffman_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, (thread.x / 32)));
        if(comp_count == 1) {
            cpu::gpujpeg_huffman_encoder_encode_kernel_warp<true>(
                coder->d_segment,
                coder->segment_count,
                coder->d_data_compressed,
                coder->d_block_list,
                coder->d_data_quantized,
                coder->d_component,
                comp_count,
                huffman_encoder->d_gpujpeg_huffman_output_byte_count,
                grid,
                thread
            );
        } else {
            cpu::gpujpeg_huffman_encoder_encode_kernel_warp<false>(
                coder->d_segment,
                coder->segment_count,
                coder->d_data_compressed,
                coder->d_block_list,
                coder->d_data_quantized,
                coder->d_component,
                comp_count,
                huffman_encoder->d_gpujpeg_huffman_output_byte_count,
                grid,
                thread
            );
        }

        // Run codeword serialization kernel
        const int num_serialization_tblocks = gpujpeg_div_and_round_up(coder->segment_count, SERIALIZATION_THREADS_PER_TBLOCK);
        cpu::gpujpeg_huffman_encoder_serialization_kernel(
            coder->d_segment,
            coder->segment_count,
            coder->d_data_compressed,
            coder->d_temp_huffman,
            num_serialization_tblocks,
            SERIALIZATION_THREADS_PER_TBLOCK
        );
    }

    // No atomic operations in CC 1.0 => run output size computation kernel to allocate the output buffer space
    if ( encoder->coder.compute_major == 1 && encoder->coder.compute_minor == 0 ) {
        cpu::gpujpeg_huffman_encoder_allocation_kernel(coder->d_segment, coder->segment_count, huffman_encoder->d_gpujpeg_huffman_output_byte_count, 1, 512);
    }

    // Run output compaction kernel (one warp per segment)
    const dim3 compaction_thread(32, WARPS_NUM);
    const dim3 compaction_grid = cpu::gpujpeg_huffman_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, WARPS_NUM));
    cpu::gpujpeg_huffman_encoder_compaction_kernel(
        coder->d_segment,
        coder->segment_count,
        coder->d_temp_huffman,
        coder->d_data_compressed,
        huffman_encoder->d_gpujpeg_huffman_output_byte_count,
        compaction_grid,
        compaction_thread
    );

    // Read and return number of occupied bytes
    memcpy_async_cpu(output_byte_count, huffman_encoder->d_gpujpeg_huffman_output_byte_count, sizeof(unsigned int), 2 /*DeviceToHost*/, encoder->stream);

    // indicate success
    return 0;
}

static int preprocessor_decoder_init_cpu(struct gpujpeg_coder* coder) {
    coder->preprocessor = NULL;

    if (!gpujpeg_pixel_format_is_interleaved(coder->param_image.pixel_format) &&
            cpu::gpujpeg_preprocessor_decode_no_transform(coder) &&
            cpu::gpujpeg_preprocessor_decode_aligned(coder)) {
        if ( coder->param.verbose >= 2 ) {
            printf("Matching format detected - not using postprocessor, using memcpy instead.");
        }
        return 0;
    }

    if (coder->param_image.comp_count == 1 && gpujpeg_pixel_format_get_comp_count(coder->param_image.pixel_format) > 1) {
        coder->param.verbose >= 0 && fprintf(stderr, "[GPUJPEG] [Error] Decoding single component JPEG allowed only to single component output format!\n");
        return -1;
    }

    assert(coder->param_image.comp_count == 3 || coder->param_image.comp_count == 4);

    if (coder->param.color_space_internal == GPUJPEG_NONE) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_NONE>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_RGB) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_RGB>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT601>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT709) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT709>(coder);
    }
    else {
        assert(false);
    }
    if (coder->preprocessor == NULL) {
        return -1;
    }
    return 0;
}

static int preprocessor_decoder_copy_planar_data_cpu(struct gpujpeg_coder * coder, void* stream)
{
    assert(coder->param_image.comp_count == 1 ||
            coder->param_image.comp_count == 3);
    size_t data_raw_offset = 0;
    bool needs_stride = false; // true if width is not divisible by MCU width
    for (int i = 0; i < coder->param_image.comp_count; ++i) {
        needs_stride = needs_stride || coder->component[i].width != coder->component[i].data_width;
    }

    if (!needs_stride) {
        for (int i = 0; i < coder->param_image.comp_count; ++i) {
                size_t component_size = coder->component[i].width * coder->component[i].height;
                memcpy_async_cpu((void*)(coder->d_data_raw + data_raw_offset), (void*)(coder->component[i].d_data), component_size, 3 /*DeviceToDevice*/, stream);
                data_raw_offset += component_size;
        }
    } else {
        for (int i = 0; i < coder->param_image.comp_count; ++i) {
                int spitch = coder->component[i].data_width;
                int dpitch = coder->component[i].width;
                size_t component_size = spitch * coder->component[i].height;
                memcpy_2d_async_cpu((void*)(coder->d_data_raw + data_raw_offset), dpitch, (void*)(coder->component[i].d_data), spitch, coder->component[i].width, coder->component[i].height, 3 /*DeviceToDevice*/, stream);
                data_raw_offset += component_size;
        }
    }
    return 0;
}

static int preprocessor_decode_cpu(struct gpujpeg_coder* coder, void* stream) {

    if (!coder->preprocessor) {
        return preprocessor_decoder_copy_planar_data_cpu(coder, stream);
    }

    // Select kernel
    cpu::gpujpeg_preprocessor_decode_kernel kernel = (cpu::gpujpeg_preprocessor_decode_kernel)coder->preprocessor;
    assert(kernel != NULL);

    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;

    // When saving 4:2:2 data of odd width, the data should have even width, so round it
    if (coder->param_image.pixel_format == GPUJPEG_422_U8_P1020) {
        image_width = gpujpeg_div_and_round_up(coder->param_image.width, 2) * 2;
    }

    // Prepare unit size
    /// @todo this stuff doesn't look correct - we multiply by unitSize and then divide by it
    int unitSize = gpujpeg_pixel_format_get_unit_size(coder->param_image.pixel_format);
    if (unitSize == 0) {
        unitSize = 1;
    }

    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);
    if ( grid.x > GPUJPEG_CUDA_MAXIMUM_GRID_SIZE ) {
        grid.y = gpujpeg_div_and_round_up(grid.x, GPUJPEG_CUDA_MAXIMUM_GRID_SIZE);
        grid.x = GPUJPEG_CUDA_MAXIMUM_GRID_SIZE;
    }

    // Run kernel
    struct gpujpeg_preprocessor_data data;
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        assert(coder->sampling_factor.horizontal % coder->component[comp].sampling_factor.horizontal == 0);
        assert(coder->sampling_factor.vertical % coder->component[comp].sampling_factor.vertical == 0);
        data.comp[comp].d_data = coder->component[comp].d_data;
        data.comp[comp].sampling_factor.horizontal = coder->sampling_factor.horizontal / coder->component[comp].sampling_factor.horizontal;
        data.comp[comp].sampling_factor.vertical = coder->sampling_factor.vertical / coder->component[comp].sampling_factor.vertical;
        data.comp[comp].data_width = coder->component[comp].data_width;
    }
    
    //kernel<<<grid, threads, 0, stream>>>(
#ifdef GPUJPEG_USE_OPENMP
    printf("Number of threads = %d - ", omp_get_num_threads());
#endif
    printf("Running cpu kernel (preprocessor_decode_cpu): %dx%dx%d - %dx%dx%d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);
    kernel(
        data,
        coder->d_data_raw,
        image_width,
        image_height,
        grid,
        threads
    );

    return 0;
}

static int preprocessor_encoder_init_cpu(struct gpujpeg_coder* coder) {
    coder->preprocessor = NULL;

    if ( coder->param_image.comp_count == 1 ) {
        return 0;
    }

    if ( cpu::gpujpeg_preprocessor_encode_no_transform(coder) ) {
        if ( coder->param.verbose >= 2 ) {
            printf("Matching format detected - not using preprocessor, using memcpy instead.");
        }
        return 0;
    }

    assert(coder->param_image.comp_count == 3 || coder->param_image.comp_count == 4);

    if (coder->param.color_space_internal == GPUJPEG_NONE) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_NONE>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_RGB) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_RGB>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT601>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT709) {
        coder->preprocessor = (void*)cpu::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT709>(coder);
    }

    if ( coder->preprocessor == NULL ) {
        return -1;
    }

    return 0;
}

static int preprocessor_encode_interlaced_cpu(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder* coder = &encoder->coder;

    // Select kernel
    cpu::gpujpeg_preprocessor_encode_kernel kernel = (cpu::gpujpeg_preprocessor_encode_kernel) coder->preprocessor;
    assert(kernel != NULL);

    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;

    // When loading 4:2:2 data of odd width, the data in fact has even width, so round it
    // (at least imagemagick convert tool generates data stream in this way)
    if (coder->param_image.pixel_format == GPUJPEG_422_U8_P1020) {
        image_width = (coder->param_image.width + 1) & ~1;
    }

    // Prepare unit size
    /// @todo this stuff doesn't look correct - we multiply by unitSize and then divide by it
    int unitSize = gpujpeg_pixel_format_get_unit_size(coder->param_image.pixel_format);
    if (unitSize == 0) {
        unitSize = 1;
    }

    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);
    while ( grid.x > GPUJPEG_CUDA_MAXIMUM_GRID_SIZE ) {
        grid.y *= 2;
        grid.x = gpujpeg_div_and_round_up(grid.x, 2);
    }

    // Decompose input image width for faster division using multiply-high and right shift
    uint32_t width_div_mul, width_div_shift;
    gpujpeg_const_div_prepare(image_width, width_div_mul, width_div_shift);

    // Run kernel
    struct gpujpeg_preprocessor_data data;
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        assert(coder->sampling_factor.horizontal % coder->component[comp].sampling_factor.horizontal == 0);
        assert(coder->sampling_factor.vertical % coder->component[comp].sampling_factor.vertical == 0);
        data.comp[comp].d_data = coder->component[comp].d_data;
        data.comp[comp].sampling_factor.horizontal = coder->sampling_factor.horizontal / coder->component[comp].sampling_factor.horizontal;
        data.comp[comp].sampling_factor.vertical = coder->sampling_factor.vertical / coder->component[comp].sampling_factor.vertical;
        data.comp[comp].data_width = coder->component[comp].data_width;
    }
    
    //kernel<<<grid, threads, 0, stream>>>(
#ifdef GPUJPEG_USE_OPENMP
    printf("Number of threads = %d - ", omp_get_num_threads());
#endif
    printf("Running cpu kernel (preprocessor_encode_interlaced_cpu): %dx%dx%d - %dx%dx%d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);
    kernel(
        data,
        coder->d_data_raw,
        coder->d_data_raw + coder->data_raw_size,
        image_width,
        image_height,
        width_div_mul,
        width_div_shift,
        grid,
        threads
    );

    return 0;
}

static int preprocessor_encoder_copy_planar_data_cpu(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder * coder = &encoder->coder;
    assert(coder->param_image.comp_count == 1 ||
            coder->param_image.comp_count == 3);

    size_t data_raw_offset = 0;
    bool needs_stride = false; // true if width is not divisible by MCU width
    for (int i = 0; i < coder->param_image.comp_count; ++i) {
        needs_stride = needs_stride || coder->component[i].width != coder->component[i].data_width;
    }
    if (!needs_stride) {
            for (int i = 0; i < coder->param_image.comp_count; ++i) {
                    size_t component_size = coder->component[i].width * coder->component[i].height;
                    memcpy_async_cpu(coder->component[i].d_data, coder->d_data_raw + data_raw_offset, component_size, 3 /*DeviceToDevice*/, encoder->stream);
                    data_raw_offset += component_size;
            }
    } else {
            for (int i = 0; i < coder->param_image.comp_count; ++i) {
                    int spitch = coder->component[i].width;
                    int dpitch = coder->component[i].data_width;
                    size_t component_size = spitch * coder->component[i].height;
                    memcpy_2d_async_cpu(coder->component[i].d_data, dpitch, coder->d_data_raw + data_raw_offset, spitch, spitch, coder->component[i].height, 3 /*DeviceToDevice*/, encoder->stream);
                    data_raw_offset += component_size;
            }
    }
    return 0;
}

static int preprocessor_encode_cpu(struct gpujpeg_encoder* encoder) {
    struct gpujpeg_coder * coder = &encoder->coder;
    if (coder->preprocessor) {
        return preprocessor_encode_interlaced_cpu(encoder);
    } else {
        return preprocessor_encoder_copy_planar_data_cpu(encoder);
    }
}

static int create_timer_cpu(struct gpujpeg_timer* timer) {
    if (timer != NULL) {
        timer->started = 0;
        return 0;
    } else {
        return -1;
    }
}

static int destroy_timer_cpu(struct gpujpeg_timer* timer) {
    return 0; // No resource deallocation needed for CPU-based timer
}

static int start_timer_cpu(struct gpujpeg_timer* timer, int record_perf, void* stream) {
    if (timer != NULL) {
        if (record_perf) {
            // Start the timer using whatever method you use when not using CUDA
            timer->start = static_cast<void*>(new decltype(std::chrono::steady_clock::now())(std::chrono::steady_clock::now()));
            timer->stop = NULL;
            timer->started = 1;
        } else {
            timer->started = -1;
        }
        return 0;
    } else {
        return -1;
    }
}

static void stop_timer_cpu(struct gpujpeg_timer* timer, int record_perf, void* stream) {
    if (timer != NULL && timer->started) {
        if (record_perf) {
            timer->stop =  static_cast<void*>(new decltype(std::chrono::steady_clock::now())(std::chrono::steady_clock::now()));
            // Stop the timer and calculate elapsed time using whatever method you use when not using CUDA
            timer->started = 0;
        }
    }
}

static double elapsed_timer_cpu(struct gpujpeg_timer* timer) {
    if (timer != NULL) {
        auto* start = static_cast<decltype(std::chrono::steady_clock::now())*>(timer->start);
        auto* stop = static_cast<decltype(std::chrono::steady_clock::now())*>(timer->stop);
        if (start == NULL || stop == NULL) return 0.0;
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(*stop - *start)/1000.0
        ;
        return elapsed_time.count();
    } else {
        return -1.0; // Return negative value to indicate error
    }
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct gpujpeg_accel* gpujpeg_accel_cpu_create(const struct gpujpeg_device* device) {
    struct gpujpeg_accel_cpu* accel_cpu = (struct gpujpeg_accel_cpu*)malloc(sizeof(struct gpujpeg_accel_cpu));
    if (!accel_cpu) {
        fprintf(stderr, "[GPUJPEG] Error allocating memory for gpujpeg_accel_cpu\n");
        return NULL;
    }

    accel_cpu->base.device = device;
    accel_cpu->base.init = init_cpu;
    accel_cpu->base.get_info = get_info_cpu;
    accel_cpu->base.get_compute_capabilities = get_compute_capabilities_cpu;
    accel_cpu->base.alloc = alloc_cpu;
    accel_cpu->base.alloc_host = alloc_host_cpu;
    accel_cpu->base.release = release_cpu;
    accel_cpu->base.release_host = release_host_cpu;
    accel_cpu->base.memoryset = memset_cpu;
    accel_cpu->base.memorycpy = memcpy_cpu;
    accel_cpu->base.memorycpy_async = memcpy_async_cpu;
    accel_cpu->base.synchronise = synchronise_cpu;
    accel_cpu->base.synchronise_stream = synchronise_stream_cpu;
    accel_cpu->base.dct = dct_cpu;
    accel_cpu->base.idct = idct_cpu;
    accel_cpu->base.encoder_destroy = encoder_destroy_cpu;
    accel_cpu->base.decoder_destroy = decoder_destroy_cpu;
    accel_cpu->base.huffman_decoder_init = huffman_decoder_init_cpu;
    accel_cpu->base.huffman_decoder_destroy = huffman_decoder_destroy_cpu;
    accel_cpu->base.huffman_decoder_decode = huffman_decoder_decode_cpu;
    accel_cpu->base.huffman_encoder_init = huffman_encoder_init_cpu;
    accel_cpu->base.huffman_encoder_destroy = huffman_encoder_destroy_cpu;
    accel_cpu->base.huffman_encoder_encode = huffman_encoder_encode_cpu;
    accel_cpu->base.preprocessor_decoder_init = preprocessor_decoder_init_cpu;
    accel_cpu->base.preprocessor_decode = preprocessor_decode_cpu;
    accel_cpu->base.preprocessor_encoder_init = preprocessor_encoder_init_cpu;
    accel_cpu->base.preprocessor_encode = preprocessor_encode_cpu;

    // Timer specific
    accel_cpu->base.timer.create_timer = create_timer_cpu;
    accel_cpu->base.timer.destroy_timer = destroy_timer_cpu;
    accel_cpu->base.timer.start_timer = start_timer_cpu;
    accel_cpu->base.timer.stop_timer = stop_timer_cpu;
    accel_cpu->base.timer.elapsed_timer = elapsed_timer_cpu;

    return &accel_cpu->base;
}

#ifdef __cplusplus
}
#endif // __cplusplus
