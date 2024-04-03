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

#include "gpujpeg_accel.h"
#include "cpu/gpujpeg_accel_cpu.h"

#ifdef GPUJPEG_USE_CUDA
#include "cuda/gpujpeg_accel_cuda.h"
#endif

#ifdef GPUJPEG_USE_OPENCL
#include "cl/gpujpeg_accel_cl.h"
#endif

// TODO - RELEASE
static struct gpujpeg_accel* instances[3] = {NULL};

struct gpujpeg_accel*
gpujpeg_accel_get(const struct gpujpeg_device* device) {
    int index = -1;
    if (device->_type == GPUJPEG_DEVICE_TYPE_CPU) {
        index = 0;
    }
#ifdef GPUJPEG_USE_CUDA
    else if (device->_type == GPUJPEG_DEVICE_TYPE_CUDA) {
        index = 1;
    }
#endif
#ifdef GPUJPEG_USE_OPENCL
    else if (device->_type == GPUJPEG_DEVICE_TYPE_OPENCL) {
        index = 2;
    }
#endif
    else {
        fprintf(stderr, "[GPUJPEG] Unknown device for acceleration '%s'\n", device->_device);
        return NULL;
    }

    if (!instances[index]) {
        switch (index) {
            case 0:
                instances[index] = gpujpeg_accel_cpu_create(device);
                break;
#ifdef GPUJPEG_USE_CUDA
            case 1:
                instances[index] = gpujpeg_accel_cuda_create(device);
                break;
#endif
#ifdef GPUJPEG_USE_OPENCL
            case 2:
                instances[index] = gpujpeg_accel_opencl_create(device);
                break;
#endif
            default:
                fprintf(stderr, "[GPUJPEG] Error creating acceleration instance\n");
                return NULL;
        }
    }

    return instances[index];
}