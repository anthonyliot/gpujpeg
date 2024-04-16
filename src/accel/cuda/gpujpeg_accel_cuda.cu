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

#define MAX max
#define ROUND round
#define GRID_BLOCK
#define GRID_BLOCK_NO_COMMA
#define LOOP_KERNEL_BEGIN
#define LOOP_KERNEL_END
#define EXTRA_BLOCK(X)

#include "gpujpeg_accel_cuda.h"
#include "../../gpujpeg_util.h"
#include "../common/gpujpeg_preprocessor_common.h"
#include "../common/gpujpeg_huffman_decoder.h"
#include "../common/gpujpeg_huffman_encoder.h"

// Kernels 
namespace cuda {
    #include "../common/gpujpeg_idct.kernel"
    #include "../common/gpujpeg_dct.kernel"
    #include "../common/gpujpeg_preprocessor.kernel"
    #include "../common/gpujpeg_postprocessor.kernel"
    #include "../common/gpujpeg_huffman_encoder.kernel"
}

#define CUDA_CHECK(stmt)                                                                                                                        \
do {                                                                                                                                            \
    cudaError_t err = (cudaError_t)stmt;                                                                                                        \
    if (err != cudaSuccess) {                                                                                                                   \
        fprintf(stderr, "[GPUJPEG] [Error] CUDA error %08x (%s), at %s:%i for %s", err, cudaGetErrorString(err), #stmt, __LINE__, __FILE__);    \
    }                                                                                                                                           \
} while (0)

static int init_cuda(const struct gpujpeg_accel* accel, int flags) {
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));

    if ( dev_count == 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] No CUDA enabled device\n");
        return -1;
    }

    int device_id = accel->device->_id;
    if ( device_id < 0 || device_id >= dev_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
            device_id, 0, dev_count - 1);
        return -1;
    }

    struct cudaDeviceProp devProp;
    if ( cudaSuccess != cudaGetDeviceProperties(&devProp, device_id) ) {
        fprintf(stderr,
            "[GPUJPEG] [Error] Can't get CUDA device properties!\n"
            "[GPUJPEG] [Error] Do you have proper driver for CUDA installed?\n"
        );
        return -1;
    }

    if ( devProp.major < 1 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Device %d does not support CUDA\n", device_id);
        return -1;
    }

#if defined GPUJPEG_USE_OPENGL && CUDART_VERSION < 5000
    if ( flags & GPUJPEG_OPENGL_INTEROPERABILITY ) {
        CUDA_CHECK(cudaGLSetGLDevice(device_id));
    }
#endif

    if ( flags & GPUJPEG_VERBOSE ) {
        int cuda_driver_version = 0;
        cudaDriverGetVersion(&cuda_driver_version);
        printf("CUDA driver version:   %d.%d\n", cuda_driver_version / 1000, (cuda_driver_version % 100) / 10);

        int cuda_runtime_version = 0;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        printf("CUDA runtime version:  %d.%d\n", cuda_runtime_version / 1000, (cuda_runtime_version % 100) / 10);

        printf("Using Device #%d:       %s (c.c. %d.%d)\n", device_id, devProp.name, devProp.major, devProp.minor);
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    // Test by simple copying that the device is ready
    uint8_t data[] = {8};
    uint8_t* d_data = NULL;
    cudaMalloc((void**)&d_data, 1);
    cudaMemcpy(d_data, data, 1, cudaMemcpyHostToDevice);
    cudaFree(d_data);
    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to initialize CUDA device: %s\n", cudaGetErrorString(error));
        if ( flags & GPUJPEG_OPENGL_INTEROPERABILITY )
            fprintf(stderr, "[GPUJPEG] [Info]  OpenGL interoperability is used, is OpenGL context available?\n");
        return -1;
    }
    return 0;
}

static void get_compute_capabilities_cuda(const struct gpujpeg_accel* accel, int* compute_major, int* compute_minor) {
    struct cudaDeviceProp device_properties;
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, accel->device->_id));
    *compute_major = device_properties.major;
    *compute_minor = device_properties.minor;
    if (device_properties.major < 2) {
        fprintf(stderr, "GPUJPEG coder is currently broken on cards with cc < 2.0\n");
    }
}

static int get_info_cuda(const struct gpujpeg_accel* accel, struct gpujpeg_devices_info* devices_info) {
    CUDA_CHECK(cudaGetDeviceCount(&devices_info->device_count));

    if ( devices_info->device_count > GPUJPEG_MAX_DEVICE_COUNT ) {
        fprintf(stderr, "[GPUJPEG] [Warning] There are available more CUDA devices (%d) than maximum count (%d).\n",
            devices_info->device_count, GPUJPEG_MAX_DEVICE_COUNT);
        fprintf(stderr, "[GPUJPEG] [Warning] Using maximum count (%d).\n", GPUJPEG_MAX_DEVICE_COUNT);
        devices_info->device_count = GPUJPEG_MAX_DEVICE_COUNT;
    }

    for ( int device_id = 0; device_id < devices_info->device_count; device_id++ ) {
        struct cudaDeviceProp device_properties;
        cudaGetDeviceProperties(&device_properties, device_id);

        struct gpujpeg_device_info* device_info = &devices_info->device[device_id];

        device_info->id = device_id;
        strncpy(device_info->name, device_properties.name, sizeof device_info->name);
        get_compute_capabilities_cuda(accel, &device_info->major, &device_info->minor);
        device_info->global_memory = device_properties.totalGlobalMem;
        device_info->constant_memory = device_properties.totalConstMem;
        device_info->shared_memory = device_properties.sharedMemPerBlock;
        device_info->register_count = device_properties.regsPerBlock;
#if CUDART_VERSION >= 2000
        device_info->multiprocessor_count = device_properties.multiProcessorCount;
#endif
    }
    return 0;
}

static void* alloc_cuda(size_t size) {
    uint8_t* addr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&addr, size));
    return (void*)addr;
}

static void* alloc_host_cuda(size_t size) {
    uint8_t* addr = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&addr, size));
    return (void*)addr;
}

static void release_cuda(void* addr) {
    CUDA_CHECK(cudaFree((void*)addr));
}

static void release_host_cuda(void* addr) {
    CUDA_CHECK(cudaFreeHost((void*)addr));
}

static void memset_cuda(void* addr, uint8_t value, size_t count) {
    CUDA_CHECK(cudaMemset((void*)addr, value, count));
}

static void memcpy_cuda(void* dstAddr, void* srcAddr, size_t count, int kind) {
    CUDA_CHECK(cudaMemcpy(dstAddr, srcAddr, count, (cudaMemcpyKind)kind));
}

static void memcpy_async_cuda(void* dstAddr, void* srcAddr, size_t count, int kind, void* stream) {
    CUDA_CHECK(cudaMemcpyAsync(dstAddr, srcAddr, count, (cudaMemcpyKind)kind, stream?*(cudaStream_t*)stream:NULL));
}

static void memcpy_2d_async_cuda(void* dstAddr, int dpitch, void* srcAddr, int spitch, int width, int height, int kind, void* stream) {
    CUDA_CHECK(cudaMemcpy2DAsync(dstAddr, dpitch, srcAddr, spitch, width, height, (cudaMemcpyKind)kind, stream?*(cudaStream_t*)stream:NULL));
}

static void synchronise_cuda(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void synchronise_stream_cuda(void* stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream?*(cudaStream_t*)stream:NULL));
}

static int dct_cuda(struct gpujpeg_encoder* encoder) {
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
            cudaMemcpyToSymbolAsync(
                cuda::gpujpeg_dct_quantization_table_const,
                d_quantization_table,
                sizeof(cuda::gpujpeg_dct_quantization_table_const),
                0,
                cudaMemcpyDeviceToDevice,
                stream
            );
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
        cuda::gpujpeg_dct_kernel<WARP_COUNT><<<dct_grid, dct_block, 0, stream>>>(
            block_count_x,
            block_count_y,
            component->d_data,
            component->data_width,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            d_quantization_table
        );
    }

    return 0;
}

static int idct_cuda(struct gpujpeg_decoder* decoder) {
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

        

        cudaStream_t stream = (decoder->stream)?*(cudaStream_t*)decoder->stream:NULL;
        // Copy quantization table to constant memory
        cudaMemcpyToSymbolAsync(
            cuda::gpujpeg_idct_quantization_table,
            d_quantization_table,
            64 * sizeof(uint16_t),
            0,
            cudaMemcpyDeviceToDevice,
            stream
        );

        dim3 grid(gpujpeg_div_and_round_up(block_count_x * block_count_y,
				(GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_Z) / GPUJPEG_BLOCK_SIZE), 1);
        dim3 block(GPUJPEG_IDCT_BLOCK_X, GPUJPEG_IDCT_BLOCK_Y, GPUJPEG_IDCT_BLOCK_Z);
 
        printf("Running cuda kernel (idct_cuda): %dx%dx%d - %dx%dx%d\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        cuda::gpujpeg_idct_kernel<<<grid, block, 0, stream>>>(
            component->d_data_quantized,
            component->d_data,
            component->data_width,
            d_quantization_table
        );
        
        CUDA_CHECK(cudaGetLastError());
    }

    return 0;
}

static struct gpujpeg_huffman_decoder* huffman_decoder_init_cuda() {
    struct gpujpeg_huffman_decoder *huffman_decoder = (struct gpujpeg_huffman_decoder *) calloc(1, sizeof(struct gpujpeg_huffman_decoder));

    #ifdef HUFFMAN_CONST_TABLES
        // Copy natural order to constant device memory
        CUDA_CHECK(cudaMemcpyToSymbol(
            gpujpeg_huffman_decoder_order_natural,
            gpujpeg_order_natural,
            GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
            0,
            cudaMemcpyHostToDevice
        ));
    #else
        CUDA_CHECK(cudaMalloc((void**)&huffman_decoder->d_order_natural, GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            huffman_decoder->d_order_natural,
            gpujpeg_order_natural,
            GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
            cudaMemcpyHostToDevice
        ));
    #endif
    
    CUDA_CHECK(cudaMalloc((void**)&huffman_decoder->d_tables_full, 4 * (1 << 16) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&huffman_decoder->d_tables_quick, QUICK_TABLE_ITEMS * sizeof(uint16_t)));

    return huffman_decoder;
}

static void huffman_decoder_destroy_cuda(struct gpujpeg_huffman_decoder* decoder) {
    if (decoder == NULL) {
        return;
    }

    CUDA_CHECK(cudaFree(decoder->d_order_natural));
    CUDA_CHECK(cudaFree(decoder->d_tables_full));
    CUDA_CHECK(cudaFree(decoder->d_tables_quick));
    free(decoder);
}

static void huffman_encoder_destroy_cuda(struct gpujpeg_huffman_encoder* encoder) {
    if (encoder == NULL) {
        return;
    }
    
    if (encoder->d_gpujpeg_huffman_output_byte_count != NULL) {
        CUDA_CHECK(cudaFree(encoder->d_gpujpeg_huffman_output_byte_count));
    }

    free(encoder);
}

static void encoder_destroy_cuda(struct gpujpeg_encoder* encoder) {
    
    if (encoder->huffman_encoder != NULL) {
        huffman_encoder_destroy_cuda(encoder->huffman_encoder);
    }

    for (int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++) {
        if (encoder->table_quantization[comp_type].d_table != NULL) {
            CUDA_CHECK(cudaFree(encoder->table_quantization[comp_type].d_table));
        }
        if (encoder->table_quantization[comp_type].d_table_forward != NULL) {
            CUDA_CHECK(cudaFree(encoder->table_quantization[comp_type].d_table_forward));
        }
    }
    
    if (encoder->writer != NULL) {
        gpujpeg_writer_destroy(encoder->writer);
    }
}

static void decoder_destroy_cuda(struct gpujpeg_decoder* decoder) {
    for (int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++) {
        if (decoder->table_quantization[comp_type].d_table != NULL) {
            CUDA_CHECK(cudaFree(decoder->table_quantization[comp_type].d_table));
        }
    }

    for ( int comp_type = 0; comp_type < GPUJPEG_MAX_COMPONENT_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            CUDA_CHECK(cudaFree(decoder->d_table_huffman[comp_type][huff_type]));
        }
    }

    if (decoder->reader != NULL) {
        free(decoder->reader);
    }

    if (decoder->huffman_decoder != NULL) {
        huffman_decoder_destroy_cuda(decoder->huffman_decoder);
    }
}

static int huffman_decoder_decode_cuda(struct gpujpeg_decoder* decoder) {
    NOT_YET_IMPLEMENTED
    return 0;
}

static struct gpujpeg_huffman_encoder* huffman_encoder_init_cuda(struct gpujpeg_encoder* encoder) {
    struct gpujpeg_huffman_encoder * huffman_encoder = (struct gpujpeg_huffman_encoder *) malloc(sizeof(struct gpujpeg_huffman_encoder));
    if ( huffman_encoder == NULL ) {
        return NULL;
    }
    memset(huffman_encoder, 0, sizeof(struct gpujpeg_huffman_encoder));

    // Allocate
    CUDA_CHECK(cudaMalloc((void**)&huffman_encoder->d_gpujpeg_huffman_output_byte_count, sizeof(unsigned int)));

    // Initialize decomposition lookup table
    cudaFuncSetCacheConfig(cuda::gpujpeg_huffman_encoder_value_decomposition_init_kernel, cudaFuncCachePreferShared);
    cuda::gpujpeg_huffman_encoder_value_decomposition_init_kernel<<<32, 256, 0, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL>>>();  // 8192 threads total
    cudaStreamSynchronize((encoder->stream)?*(cudaStream_t*)encoder->stream:NULL);

    // compose GPU version of the huffman LUT and copy it into GPU memory (for CC >= 2.0)
    uint32_t gpujpeg_huffman_cpu_lut[(256 + 1) * 4];
    cuda::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 0, &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC], true);
    cuda::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 1, &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC], false);
    cuda::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 2, &encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC], true);
    cuda::gpujpeg_huffman_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 3, &encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC], false);
    cudaMemcpyToSymbol(
        cuda::gpujpeg_huffman_lut,
        gpujpeg_huffman_cpu_lut,
        (256 + 1) * 4 * sizeof(*cuda::gpujpeg_huffman_lut),
        0,
        cudaMemcpyHostToDevice
    );

    // Copy original Huffman coding tables to GPU memory (for CC 1.x)
    cudaMemcpyToSymbol(
        cuda::gpujpeg_huffman_encoder_table_huffman,
        &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC],
        sizeof(cuda::gpujpeg_huffman_encoder_table_huffman),
        0,
        cudaMemcpyHostToDevice
    );

    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        cuda::gpujpeg_huffman_encoder_order_natural,
        gpujpeg_order_natural,
        GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );

    // Configure more shared memory for all kernels
    cudaFuncSetCacheConfig(cuda::gpujpeg_huffman_encoder_encode_kernel_warp<true>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(cuda::gpujpeg_huffman_encoder_encode_kernel_warp<false>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(cuda::gpujpeg_huffman_encoder_serialization_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(cuda::gpujpeg_huffman_encoder_compaction_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(cuda::gpujpeg_huffman_encoder_encode_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(cuda::gpujpeg_huffman_encoder_allocation_kernel, cudaFuncCachePreferShared);

    return huffman_encoder;
}

static int huffman_encoder_encode_cuda(struct gpujpeg_encoder* encoder, struct gpujpeg_huffman_encoder* huffman_encoder, unsigned int* output_byte_count) {
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
        cuda::gpujpeg_huffman_encoder_encode_kernel<<<grid, thread, 0, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL>>>(
            coder->d_component,
            coder->d_segment,
            comp_count,
            coder->segment_count,
            coder->d_temp_huffman,
            huffman_encoder->d_gpujpeg_huffman_output_byte_count
        );
    } else {
        // Run encoder kernel
        dim3 thread(32 * WARPS_NUM);
        dim3 grid = cuda::gpujpeg_huffman_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, (thread.x / 32)));
        if(comp_count == 1) {
            cuda::gpujpeg_huffman_encoder_encode_kernel_warp<true><<<grid, thread, 0, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL>>>(
                coder->d_segment,
                coder->segment_count,
                coder->d_data_compressed,
                coder->d_block_list,
                coder->d_data_quantized,
                coder->d_component,
                comp_count,
                huffman_encoder->d_gpujpeg_huffman_output_byte_count
            );
        } else {
            cuda::gpujpeg_huffman_encoder_encode_kernel_warp<false><<<grid, thread, 0, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL>>>(
                coder->d_segment,
                coder->segment_count,
                coder->d_data_compressed,
                coder->d_block_list,
                coder->d_data_quantized,
                coder->d_component,
                comp_count,
                huffman_encoder->d_gpujpeg_huffman_output_byte_count
            );
        }

        // Run codeword serialization kernel
        const int num_serialization_tblocks = gpujpeg_div_and_round_up(coder->segment_count, SERIALIZATION_THREADS_PER_TBLOCK);
        cuda::gpujpeg_huffman_encoder_serialization_kernel<<<num_serialization_tblocks, SERIALIZATION_THREADS_PER_TBLOCK, 0, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL>>>(
            coder->d_segment,
            coder->segment_count,
            coder->d_data_compressed,
            coder->d_temp_huffman
        );
    }

    // No atomic operations in CC 1.0 => run output size computation kernel to allocate the output buffer space
    if ( encoder->coder.compute_major == 1 && encoder->coder.compute_minor == 0 ) {
        cuda::gpujpeg_huffman_encoder_allocation_kernel<<<1, 512, 0, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL>>>(coder->d_segment, coder->segment_count, huffman_encoder->d_gpujpeg_huffman_output_byte_count);
    }

    // Run output compaction kernel (one warp per segment)
    const dim3 compaction_thread(32, WARPS_NUM);
    const dim3 compaction_grid = cuda::gpujpeg_huffman_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, WARPS_NUM));
    cuda::gpujpeg_huffman_encoder_compaction_kernel<<<compaction_grid, compaction_thread, 0, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL>>>(
        coder->d_segment,
        coder->segment_count,
        coder->d_temp_huffman,
        coder->d_data_compressed,
        huffman_encoder->d_gpujpeg_huffman_output_byte_count
    );

    // Read and return number of occupied bytes
    cudaMemcpyAsync(output_byte_count, huffman_encoder->d_gpujpeg_huffman_output_byte_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, (encoder->stream)?*(cudaStream_t*)encoder->stream:NULL);

    // indicate success
    return 0;
}

static int preprocessor_decoder_init_cuda(struct gpujpeg_coder* coder) {
    coder->preprocessor = NULL;

    if (!gpujpeg_pixel_format_is_interleaved(coder->param_image.pixel_format) &&
            cuda::gpujpeg_preprocessor_decode_no_transform(coder) &&
            cuda::gpujpeg_preprocessor_decode_aligned(coder)) {
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
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_NONE>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_RGB) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_RGB>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT601>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT709) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_decode_kernel<GPUJPEG_YCBCR_BT709>(coder);
    }
    else {
        assert(false);
    }
    if (coder->preprocessor == NULL) {
        return -1;
    }
    return 0;
}

static int
preprocessor_decoder_copy_planar_data_cuda(struct gpujpeg_coder * coder, cudaStream_t stream)
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
                memcpy_async_cuda((void*)(coder->d_data_raw + data_raw_offset), (void*)(coder->component[i].d_data), component_size, 3 /*DeviceToDevice*/, stream);
                data_raw_offset += component_size;
        }
    } else {
        for (int i = 0; i < coder->param_image.comp_count; ++i) {
                int spitch = coder->component[i].data_width;
                int dpitch = coder->component[i].width;
                size_t component_size = spitch * coder->component[i].height;
                memcpy_2d_async_cuda((void*)(coder->d_data_raw + data_raw_offset), dpitch, (void*)(coder->component[i].d_data), spitch, coder->component[i].width, coder->component[i].height, 3 /*DeviceToDevice*/, stream);
                data_raw_offset += component_size;
        }
    }
    return 0;
}

static int preprocessor_decode_cuda(struct gpujpeg_coder* coder, void* stream) {
    if (!coder->preprocessor) {
        return preprocessor_decoder_copy_planar_data_cuda(coder, stream?*(cudaStream_t*)stream:NULL);
    }

    // Select kernel
    cuda::gpujpeg_preprocessor_decode_kernel kernel = (cuda::gpujpeg_preprocessor_decode_kernel)coder->preprocessor;
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

    printf("Running cuda kernel (preprocessor_decode_cuda): %dx%dx%d - %dx%dx%d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);
    kernel<<<grid, threads, 0, stream?*(cudaStream_t*)stream:NULL>>>(
        data,
        coder->d_data_raw,
        image_width,
        image_height
    );
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int preprocessor_encoder_init_cuda(struct gpujpeg_coder* coder) {
    coder->preprocessor = NULL;

    if ( coder->param_image.comp_count == 1 ) {
        return 0;
    }

    if ( cuda::gpujpeg_preprocessor_encode_no_transform(coder) ) {
        if ( coder->param.verbose >= 2 ) {
            printf("Matching format detected - not using preprocessor, using memcpy instead.");
        }
        return 0;
    }

    assert(coder->param_image.comp_count == 3 || coder->param_image.comp_count == 4);

    if (coder->param.color_space_internal == GPUJPEG_NONE) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_NONE>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_RGB) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_RGB>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT601>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT601_256LVLS) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT601_256LVLS>(coder);
    }
    else if (coder->param.color_space_internal == GPUJPEG_YCBCR_BT709) {
        coder->preprocessor = (void*)cuda::gpujpeg_preprocessor_select_encode_kernel<GPUJPEG_YCBCR_BT709>(coder);
    }

    if ( coder->preprocessor == NULL ) {
        return -1;
    }

    return 0;
}

int
preprocessor_encode_interlaced(struct gpujpeg_encoder * encoder)
{
    struct gpujpeg_coder* coder = &encoder->coder;

    // Select kernel
    cuda::gpujpeg_preprocessor_encode_kernel kernel = (cuda::gpujpeg_preprocessor_encode_kernel) coder->preprocessor;
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
    kernel<<<grid, threads, 0, encoder->stream?*(cudaStream_t*)encoder->stream:NULL>>>(
        data,
        coder->d_data_raw,
        coder->d_data_raw + coder->data_raw_size,
        image_width,
        image_height,
        width_div_mul,
        width_div_shift
    );

    return 0;
}

/**
 * Copies raw data from source image to GPU memory without running
 * any preprocessor kernel.
 *
 * This assumes that the JPEG has same color space as input raw image and
 * currently also that the component subsampling correspond between raw and
 * JPEG (although at least different horizontal subsampling can be quite
 * easily done).
 *
 * @invariant preprocessor_encode_no_transform(coder) != 0
 */
static int
preprocessor_encoder_copy_planar_data(struct gpujpeg_encoder * encoder)
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
                    cudaMemcpyAsync(coder->component[i].d_data, coder->d_data_raw + data_raw_offset, component_size, cudaMemcpyDeviceToDevice, encoder->stream?*(cudaStream_t*)encoder->stream:NULL);
                    data_raw_offset += component_size;
            }
    } else {
            for (int i = 0; i < coder->param_image.comp_count; ++i) {
                    int spitch = coder->component[i].width;
                    int dpitch = coder->component[i].data_width;
                    size_t component_size = spitch * coder->component[i].height;
                    cudaMemcpy2DAsync(coder->component[i].d_data, dpitch, coder->d_data_raw + data_raw_offset, spitch, spitch, coder->component[i].height, cudaMemcpyDeviceToDevice, encoder->stream?*(cudaStream_t*)encoder->stream:NULL);
                    data_raw_offset += component_size;
            }
    }
    return 0;
}

static int preprocessor_encode_cuda(struct gpujpeg_encoder* encoder) {
    struct gpujpeg_coder * coder = &encoder->coder;
    if (coder->preprocessor) {
            return preprocessor_encode_interlaced(encoder);
    } else {
        return preprocessor_encoder_copy_planar_data(encoder);
    }
}

static int create_timer_cuda(struct gpujpeg_timer* timer) {
    if (timer != NULL) {
        cudaError_t cuda_status;

        cuda_status = cudaEventCreate((cudaEvent_t*)&(timer->start));
        if (cuda_status != cudaSuccess) {
            return -1;
        }

        cuda_status = cudaEventCreate((cudaEvent_t*)&(timer->stop));
        if (cuda_status != cudaSuccess) {
            cudaEventDestroy((cudaEvent_t)timer->start);
            return -1;
        }

        timer->started = 0;
        return 0;
    } else {
        return -1;
    }
}

static int destroy_timer_cuda(struct gpujpeg_timer* timer) {
    int err_action = cudaEventDestroy((cudaEvent_t)timer->start);
    if (err_action != cudaSuccess) {
        return -1;
    }
    
    err_action = cudaEventDestroy((cudaEvent_t)timer->stop);
    if (err_action != cudaSuccess) {
        return -1;
    }
    return 0;
}

static int start_timer_cuda(struct gpujpeg_timer* timer, int record_perf, void* stream) {
    if (record_perf) {
        timer->started = 1;

        int err_action = cudaEventRecord((cudaEvent_t)timer->start, stream?*(cudaStream_t*)stream:NULL);
        if (err_action != cudaSuccess) {
           return -1;
        }
    } else {
        timer->started = -1;
    }

    return 0;
}

static void stop_timer_cuda(struct gpujpeg_timer* timer, int record_perf, void* stream) {
    if (record_perf) {
        timer->started = 0;
        int err_action = cudaEventRecord((cudaEvent_t)timer->stop, stream?*(cudaStream_t*)stream:NULL);
        cudaEventSynchronize((cudaEvent_t)timer->stop);
        if (err_action != cudaSuccess) {
           return;
        }
    }
}

double elapsed_timer_cuda(struct gpujpeg_timer* timer) {
    if (timer != NULL) {
        float milliseconds = 0.0f;
        if (timer->started) {
            cudaEventSynchronize((cudaEvent_t)timer->stop);
        }
        cudaEventElapsedTime(&milliseconds, (cudaEvent_t)timer->start, (cudaEvent_t)timer->stop);
        return milliseconds;
    } else {
        return -1.0; // Return negative value to indicate error
    }
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct gpujpeg_accel* gpujpeg_accel_cuda_create(const struct gpujpeg_device* device) {
    struct gpujpeg_accel_cuda* accel_cuda = (struct gpujpeg_accel_cuda*)malloc(sizeof(struct gpujpeg_accel_cuda));
    if (!accel_cuda) {
        fprintf(stderr, "[GPUJPEG] Error allocating memory for gpujpeg_accel_cuda\n");
        return NULL;
    }

    accel_cuda->base.device = device;
    accel_cuda->base.init = init_cuda;
    accel_cuda->base.get_info = get_info_cuda;
    accel_cuda->base.get_compute_capabilities = get_compute_capabilities_cuda;
    accel_cuda->base.alloc = alloc_cuda;
    accel_cuda->base.alloc_host = alloc_host_cuda;
    accel_cuda->base.release = release_cuda;
    accel_cuda->base.release_host = release_host_cuda;
    accel_cuda->base.memoryset = memset_cuda;
    accel_cuda->base.memorycpy = memcpy_cuda;
    accel_cuda->base.memorycpy_async = memcpy_async_cuda;
    accel_cuda->base.synchronise = synchronise_cuda;
    accel_cuda->base.synchronise_stream = synchronise_stream_cuda;
    accel_cuda->base.dct = dct_cuda;
    accel_cuda->base.idct = idct_cuda;
    accel_cuda->base.encoder_destroy = encoder_destroy_cuda;
    accel_cuda->base.decoder_destroy = decoder_destroy_cuda;
    accel_cuda->base.huffman_decoder_init = huffman_decoder_init_cuda;
    accel_cuda->base.huffman_encoder_init = huffman_encoder_init_cuda;
    accel_cuda->base.huffman_decoder_destroy = huffman_decoder_destroy_cuda;
    accel_cuda->base.huffman_encoder_destroy = huffman_encoder_destroy_cuda;
    accel_cuda->base.huffman_decoder_decode = huffman_decoder_decode_cuda;
    accel_cuda->base.huffman_encoder_encode = huffman_encoder_encode_cuda;
    accel_cuda->base.preprocessor_decoder_init = preprocessor_decoder_init_cuda;
    accel_cuda->base.preprocessor_decode = preprocessor_decode_cuda;
    accel_cuda->base.preprocessor_encoder_init = preprocessor_encoder_init_cuda;
    accel_cuda->base.preprocessor_encode = preprocessor_encode_cuda;

    // Timer specific
    accel_cuda->base.timer.create_timer = create_timer_cuda;
    accel_cuda->base.timer.destroy_timer = destroy_timer_cuda;
    accel_cuda->base.timer.start_timer = start_timer_cuda;
    accel_cuda->base.timer.stop_timer = stop_timer_cuda;
    accel_cuda->base.timer.elapsed_timer = elapsed_timer_cuda;

    return &accel_cuda->base;
}

#ifdef __cplusplus
}
#endif // __cplusplus