
/**
 * @file
 * Copyright (c) 2011-2023, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
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
 
#ifndef GPUJPEG_HUFFMAN_ENCODER_H
#define GPUJPEG_HUFFMAN_ENCODER_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define WARPS_NUM 8

#define SERIALIZATION_THREADS_PER_TBLOCK 192

/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_encoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE];

/**
 * Huffman coding tables in constant memory - each has 257 items (256 + 1 extra)
 * There are are 4 of them - one after another, in following order:
 *    - luminance (Y) AC
 *    - luminance (Y) DC
 *    - chroma (cb/cr) AC
 *    - chroma (cb/cr) DC
 */
__device__ uint32_t gpujpeg_huffman_lut[(256 + 1) * 4];

/**
 * Value decomposition in constant memory (input range from -4096 to 4095  ... both inclusive)
 * Mapping from coefficient value into the code for the value ind its bit size.
 */
__device__ unsigned int gpujpeg_huffman_value_decomposition[8 * 1024];

/** Allocate huffman tables in constant memory */
__device__ struct gpujpeg_table_huffman_encoder gpujpeg_huffman_encoder_table_huffman[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];

struct gpujpeg_huffman_encoder
{
    /** Size of occupied part of output buffer */
    unsigned int * d_gpujpeg_huffman_output_byte_count;
};

#endif // GPUJPEG_HUFFMAN_ENCODER_H