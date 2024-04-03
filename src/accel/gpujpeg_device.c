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

#include "../../libgpujpeg/gpujpeg_common.h"
#include "gpujpeg_accel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static const char* device_names[] = {
    [GPUJPEG_DEVICE_TYPE_CPU] = "CPU",
    [GPUJPEG_DEVICE_TYPE_CUDA] = "CUDA",
    [GPUJPEG_DEVICE_TYPE_OPENCL] = "OPENCL",
    [GPUJPEG_DEVICE_TYPE_UNKNOWN] = "UNKNOWN"
};

static int device_index_from_string(const char* type) {
    const char* index_str = strchr(type, ':');
    if (index_str != NULL) {
        return atoi(index_str + 1);
    } else {
        return 0; // Default index is 0 if not specified
    }
}

static enum gpujpeg_device_type device_type_from_string(const char* type) {
    if (strcmp(type, "cpu") == 0) {
         return GPUJPEG_DEVICE_TYPE_CPU;
     } else if (strncmp(type, "cuda", 4) == 0) {
         return GPUJPEG_DEVICE_TYPE_CUDA;
     } else if (strncmp(type, "opencl", 6) == 0) {
         return GPUJPEG_DEVICE_TYPE_OPENCL;
     } else {
         return GPUJPEG_DEVICE_TYPE_UNKNOWN;
     }
}

void
gpujpeg_device_create(struct gpujpeg_device* device) {
    device->_device = GPUJPEG_DEVICE_STR_CPU;
    device->_type_str = device_names[GPUJPEG_DEVICE_TYPE_CPU];
    device->_id = 0;
    device->_type = GPUJPEG_DEVICE_TYPE_CPU;
}

void 
gpujpeg_device_create_with_type(struct gpujpeg_device* device, const char* type) {
    enum gpujpeg_device_type dev_type = device_type_from_string(type);
    int dev_index = device_index_from_string(type);
    
    if (dev_type == GPUJPEG_DEVICE_TYPE_UNKNOWN || (dev_index != 0 && dev_index != 1)) {
        fprintf(stderr, "[GPUJPEG] [Warning] Invalid type device '%s' Fallback to CPU\n", type);
        gpujpeg_device_create(device);
        return;
    }
    
    device->_device = strdup(type);
    device->_id = dev_index;
    device->_type = dev_type;
    device->_type_str = device_names[dev_type];
}

void 
gpujpeg_device_create_with_type_and_index(struct gpujpeg_device* device, enum gpujpeg_device_type t, uint32_t i) {
    if (i > UINT32_MAX) {
        fprintf(stderr, "[GPUJPEG] [Warning] Invalid device index '%u' Fallback to CPU\n", i);
        gpujpeg_device_create(device);
        return;
    }
    device->_id = i;
    device->_type = t;
    device->_type_str = device_names[t];
    switch (t) {
        case GPUJPEG_DEVICE_TYPE_CPU:
            device->_device = strdup(GPUJPEG_DEVICE_STR_CPU);
            break;
        case GPUJPEG_DEVICE_TYPE_CUDA:
            if (i == 0) {
                device->_device = strdup(GPUJPEG_DEVICE_STR_CUDA_0);
            } else if (i == 1) {
                device->_device = strdup(GPUJPEG_DEVICE_STR_CUDA_1);
            } else {
                fprintf(stderr, "[GPUJPEG] [Error] Invalid CUDA device index '%u' Fallback to CPU\n", i);
                gpujpeg_device_create(device);
                return;
            }
            break;
        case GPUJPEG_DEVICE_TYPE_OPENCL:
            if (i == 0) {
                device->_device = strdup(GPUJPEG_DEVICE_STR_OPENCL_0);
            } else if (i == 1) {
                device->_device = strdup(GPUJPEG_DEVICE_STR_OPENCL_1);
            } else {
                fprintf(stderr, "[GPUJPEG] [Error] Invalid OpenCL device index '%u' Fallback to CPU\n", i);
                gpujpeg_device_create(device);
                return;
            }
            break;
        default:
            fprintf(stderr, "[GPUJPEG] [Error] Invalid device type '%d' Fallback to CPU\n", t);
            gpujpeg_device_create(device);
            return;
    }
}

const char* gpujpeg_device_name(const struct gpujpeg_device* device) {
    if (device == NULL) {
        return NULL;
    }
    return device_names[device->_type];
}

enum gpujpeg_device_type gpujpeg_device_type(const struct gpujpeg_device* device) {
    if (device == NULL) {
        return -1;
    }
    return device->_type;
}

const char* gpujpeg_device_type_str(const struct gpujpeg_device* device) {
    if (device == NULL) {
        return NULL;
    }

    return device_names[device->_type];
}

const char* gpujpeg_device_str(const struct gpujpeg_device* device) {
    return device->_device;
}

uint32_t gpujpeg_device_index(const struct gpujpeg_device* device) {
    if (device == NULL) {
        return -1;
    }
    return device->_id;
}

int gpujpeg_device_is_supported(const struct gpujpeg_device* device) {
    if (device == NULL) {
        return -1;
    }
    switch (device->_type) {
        case GPUJPEG_DEVICE_TYPE_CPU:
            return 0;
        case GPUJPEG_DEVICE_TYPE_CUDA:
#ifdef GPUJPEG_USE_CUDA
            return (device->_id < 2) ? 0 : -1; // Assuming maximum 2 CUDA devices
#else
            return -1;
#endif
        case GPUJPEG_DEVICE_TYPE_OPENCL:
#ifdef GPUJPEG_USE_OPENCL
            return (device->_id < 2) ? 0 : -1; // Assuming maximum 2 OpenCL devices
#else
            return -1;
#endif
        default:
            return -1;
    }
}


/* Documented at declaration */
struct gpujpeg_devices_info
gpujpeg_device_get_info(const struct gpujpeg_device* device)
{
    struct gpujpeg_devices_info device_info = { 0 };
    
    if (gpujpeg_device_is_supported(device) != 0) {
        fprintf(stderr, "[GPUJPEG] [Error] Device '%s' is not supported\n", gpujpeg_device_str(device));
        device_info.device_count = 0;
    } else {
        struct gpujpeg_accel* accel = gpujpeg_accel_get(device);
        accel->get_info(accel, &device_info);
    }

    return device_info;
}

/* Documented at declaration */
int
gpujpeg_device_print_info(const struct gpujpeg_device* device)
{
    if (device == NULL) {
        char** device_names = (char*[]) { "cpu", "cuda", "opencl" };
        
        for (int i = 0; i < 3; i++) {
            struct gpujpeg_device device_tmp = { 0 };
            gpujpeg_device_create_with_type(&device_tmp , device_names[i]);
            gpujpeg_device_print_info(&device_tmp);
        }

        return 0;
    } else {
        struct gpujpeg_devices_info devices_info = gpujpeg_device_get_info(device);
        
        if ( devices_info.device_count == 0 ) {
            printf("\nThere is no device supporting %s\n", device->_type_str);
            return -1;
        } else if ( devices_info.device_count == 1 ) {
            printf("\nThere is 1 device supporting %s\n", device->_type_str);
        } else {
            printf("\nThere are %d devices supporting %s\n", devices_info.device_count, device->_type_str);
        }

        for ( int device_id = 0; device_id < devices_info.device_count; device_id++ ) {
            struct gpujpeg_device_info* device_info = &devices_info.device[device_id];
            printf("Device #%d: \"%s\"\n", device_info->id, device_info->name);
            printf("  Compute capability: %d.%d\n", device_info->major, device_info->minor);
            printf("  Total amount of global memory: %zu KiB\n", device_info->global_memory / 1024);
            printf("  Total amount of constant memory: %zu KiB\n", device_info->constant_memory / 1024);
            printf("  Total amount of shared memory per block: %zu KiB\n", device_info->shared_memory / 1024);
            printf("  Total number of registers available per block: %d\n", device_info->register_count);
            printf("  Multiprocessors: %d\n", device_info->multiprocessor_count);
        }

        if (device->_id >= devices_info.device_count) {
            fprintf(stderr, "[GPUJPEG] [Error] Device index %d is out of range\n", device->_id);
            return -1;
        }
        
        return 0;
    }
}
