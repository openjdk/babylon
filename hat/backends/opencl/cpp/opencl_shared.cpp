/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
#include "opencl_shared.h"
#include <cstdio>

const char *error(cl_int status) {
   static struct {
      cl_int code;
      const char *msg;
   } error_table[] = {
      {CL_SUCCESS,                         "success"},
      {CL_DEVICE_NOT_FOUND,                "device not found",},
      {CL_DEVICE_NOT_AVAILABLE,            "device not available",},
      {CL_COMPILER_NOT_AVAILABLE,          "compiler not available",},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE,   "mem object allocation failure",},
      {CL_OUT_OF_RESOURCES,                "out of resources",},
      {CL_OUT_OF_HOST_MEMORY,              "out of host memory",},
      {CL_PROFILING_INFO_NOT_AVAILABLE,    "profiling not available",},
      {CL_MEM_COPY_OVERLAP,                "memcopy overlaps",},
      {CL_IMAGE_FORMAT_MISMATCH,           "image format mismatch",},
      {CL_IMAGE_FORMAT_NOT_SUPPORTED,      "image format not supported",},
      {CL_BUILD_PROGRAM_FAILURE,           "build program failed",},
      {CL_MAP_FAILURE,                     "map failed",},
      {CL_INVALID_VALUE,                   "invalid value",},
      {CL_INVALID_DEVICE_TYPE,             "invalid device type",},
      {CL_INVALID_PLATFORM,                "invlaid platform",},
      {CL_INVALID_DEVICE,                  "invalid device",},
      {CL_INVALID_CONTEXT,                 "invalid context",},
      {CL_INVALID_QUEUE_PROPERTIES,        "invalid queue properties",},
      {CL_INVALID_COMMAND_QUEUE,           "invalid command queue",},
      {CL_INVALID_HOST_PTR,                "invalid host ptr",},
      {CL_INVALID_MEM_OBJECT,              "invalid mem object",},
      {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "invalid image format descriptor ",},
      {CL_INVALID_IMAGE_SIZE,              "invalid image size",},
      {CL_INVALID_SAMPLER,                 "invalid sampler",},
      {CL_INVALID_BINARY,                  "invalid binary",},
      {CL_INVALID_BUILD_OPTIONS,           "invalid build options",},
      {CL_INVALID_PROGRAM,                 "invalid program ",},
      {CL_INVALID_PROGRAM_EXECUTABLE,      "invalid program executable",},
      {CL_INVALID_KERNEL_NAME,             "invalid kernel name",},
      {CL_INVALID_KERNEL_DEFINITION,       "invalid definition",},
      {CL_INVALID_KERNEL,                  "invalid kernel",},
      {CL_INVALID_ARG_INDEX,               "invalid arg index",},
      {CL_INVALID_ARG_VALUE,               "invalid arg value",},
      {CL_INVALID_ARG_SIZE,                "invalid arg size",},
      {CL_INVALID_KERNEL_ARGS,             "invalid kernel args",},
      {CL_INVALID_WORK_DIMENSION,          "invalid work dimension",},
      {CL_INVALID_WORK_GROUP_SIZE,         "invalid work group size",},
      {CL_INVALID_WORK_ITEM_SIZE,          "invalid work item size",},
      {CL_INVALID_GLOBAL_OFFSET,           "invalid global offset",},
      {CL_INVALID_EVENT_WAIT_LIST,         "invalid event wait list",},
      {CL_INVALID_EVENT,                   "invalid event",},
      {CL_INVALID_OPERATION,               "invalid operation",},
      {CL_INVALID_GL_OBJECT,               "invalid gl object",},
      {CL_INVALID_BUFFER_SIZE,             "invalid buffer size",},
      {CL_INVALID_MIP_LEVEL,               "invalid mip level",},
      {CL_INVALID_GLOBAL_WORK_SIZE,        "invalid global work size",},
      {0, NULL},
   };
   static char unknown[256];
   int ii;

   for (ii = 0; error_table[ii].msg != NULL; ii++) {
      if (error_table[ii].code == status) {
         //std::cerr << " clerror '" << error_table[ii].msg << "'" << std::endl;
         return error_table[ii].msg;
      }
   }
   SNPRINTF(unknown, sizeof(unknown), "unknown error %d", status);
   return unknown;
}

