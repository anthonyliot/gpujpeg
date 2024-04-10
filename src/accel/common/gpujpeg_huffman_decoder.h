
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
 
#ifndef GPUJPEG_HUFFMAN_DECODER_H
#define GPUJPEG_HUFFMAN_DECODER_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * Entry of pre-built Huffman fast-decoding table.
 */
struct gpujpeg_table_huffman_decoder_entry {
    
    int value_nbits;
};

/** Number of code bits to be checked first (with high chance for the code to fit into this number of bits). */
#define QUICK_CHECK_BITS 10
#define QUICK_TABLE_ITEMS (4 * (1 << QUICK_CHECK_BITS))
// TODO: try to tweak QUICK table size and memory space

struct gpujpeg_huffman_decoder {
    /**
     * 4 pre-built tables for faster Huffman decoding (codewords up-to 16 bit length):
     *   - 0x00000 to 0x0ffff: luminance DC table
     *   - 0x10000 to 0x1ffff: luminance AC table
     *   - 0x20000 to 0x2ffff: chrominance DC table
     *   - 0x30000 to 0x3ffff: chrominance AC table
     *
     * Each entry consists of:
     *   - Number of bits of code corresponding to this entry (0 - 16, both inclusive) - bits 4 to 8
     *   - Number of run-length coded zeros before currently decoded coefficient + 1 (1 - 64, both inclusive) - bits 9 to 15
     *   - Number of bits representing the value of currently decoded coefficient (0 - 15, both inclusive) - bits 0 to 3
     * @code
     * bit #:    15                      9   8               4   3           0
     *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
     * value:  |      RLE zero count       |   code bit size   | value bit size|
     *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
     * @endcode
     */
    uint16_t *d_tables_full;

    /** Table with same format as the full table, except that all-zero-entry means that the full table should be consulted. */
    uint16_t *d_tables_quick;

    /** Natural order */
    int *d_order_natural;
};

#ifdef HUFFMAN_CONST_TABLES
/** Same table as above, but copied into constant memory */
__constant__ uint16_t gpujpeg_huffman_decoder_tables_quick_const[QUICK_TABLE_ITEMS];
 
/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_decoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE];
#endif

#endif
