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

void info() {
   cl_int status = CL_SUCCESS;
   cl_uint platformc;

   status = clGetPlatformIDs(0, NULL, &platformc);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "clGetPlatformIDs(0,NULL,&platformc) failed!\n%s\n", error(status));
      exit(1);
   }
   fprintf(stderr, "There %s %d platform%s\n", ((platformc == 1) ? "is" : "are"), platformc, ((platformc == 1) ? "" : "s"));
   cl_platform_id *platformIds = new cl_platform_id[platformc];
   status = clGetPlatformIDs(platformc, platformIds, NULL);
   if (status != CL_SUCCESS) {
      fprintf(stderr, "clGetPlatformIDs(platformc,platformIds,NULL) failed!\n%s\n", error(status));
      exit(1);
   }
   for (unsigned platformIdx = 0; platformIdx < platformc; ++platformIdx) {
      fprintf(stderr, "platform %d{\n", platformIdx);
      char platformVersionName[512];
      status = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_VERSION, sizeof(platformVersionName), platformVersionName, NULL);

      char platformVendorName[512];
      char platformName[512];
      status = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_VENDOR, sizeof(platformVendorName), platformVendorName, NULL);
      status = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
      fprintf(stderr, "   CL_PLATFORM_VENDOR..\"%s\"\n", platformVendorName);
      fprintf(stderr, "   CL_PLATFORM_VERSION.\"%s\"\n", platformVersionName);
      fprintf(stderr, "   CL_PLATFORM_NAME....\"%s\"\n", platformName);
      cl_uint deviceIdc;
      cl_device_type requestedDeviceType = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU;
      status = clGetDeviceIDs(platformIds[platformIdx], requestedDeviceType, 0, NULL, &deviceIdc);
      fprintf(stderr, "   Platform %d has %d device%s{\n", platformIdx, deviceIdc, ((deviceIdc == 1) ? "" : "s"));
      if (status == CL_SUCCESS && deviceIdc > 0) {
         cl_device_id *deviceIds = new cl_device_id[deviceIdc];
         status = clGetDeviceIDs(platformIds[platformIdx], requestedDeviceType, deviceIdc, deviceIds, NULL);
         if (status == CL_SUCCESS) {
            for (unsigned deviceIdx = 0; deviceIdx < deviceIdc; deviceIdx++) {
               fprintf(stderr, "      Device %d{\n", deviceIdx);

               cl_device_type deviceType;
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
               fprintf(stderr, "         CL_DEVICE_TYPE..................... ");
               if (deviceType & CL_DEVICE_TYPE_DEFAULT) {
                  deviceType &= ~CL_DEVICE_TYPE_DEFAULT;
                  fprintf(stderr, "Default ");
               }
               if (deviceType & CL_DEVICE_TYPE_CPU) {
                  deviceType &= ~CL_DEVICE_TYPE_CPU;
                  fprintf(stderr, "CPU ");
               }
               if (deviceType & CL_DEVICE_TYPE_GPU) {
                  deviceType &= ~CL_DEVICE_TYPE_GPU;
                  fprintf(stderr, "GPU ");
               }
               if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
                  deviceType &= ~CL_DEVICE_TYPE_ACCELERATOR;
                  fprintf(stderr, "Accelerator ");
               }
               fprintf(stderr, LongHexNewline, deviceType);

               cl_uint maxComputeUnits;
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
               fprintf(stderr, "         CL_DEVICE_MAX_COMPUTE_UNITS........ %u\n", maxComputeUnits);

               cl_uint maxWorkItemDimensions;
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);
               fprintf(stderr, "         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. %u\n", maxWorkItemDimensions);

               size_t *maxWorkItemSizes = new size_t[maxWorkItemDimensions];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxWorkItemDimensions, maxWorkItemSizes, NULL);
               for (unsigned dimIdx = 0; dimIdx < maxWorkItemDimensions; dimIdx++) {
                  fprintf(stderr, "             dim[%d] = %ld\n", dimIdx, maxWorkItemSizes[dimIdx]);
               }

               size_t maxWorkGroupSize;
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
               fprintf(stderr, "         CL_DEVICE_MAX_WORK_GROUP_SIZE...... " Size_tNewline, maxWorkGroupSize);

               cl_ulong maxMemAllocSize;
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
               fprintf(stderr, "         CL_DEVICE_MAX_MEM_ALLOC_SIZE....... " LongUnsignedNewline, maxMemAllocSize);

               cl_ulong globalMemSize;
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
               fprintf(stderr, "         CL_DEVICE_GLOBAL_MEM_SIZE.......... " LongUnsignedNewline, globalMemSize);

               cl_ulong localMemSize;
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
               fprintf(stderr, "         CL_DEVICE_LOCAL_MEM_SIZE........... " LongUnsignedNewline, localMemSize);


               char profile[2048];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_PROFILE, sizeof(profile), &profile, NULL);
               fprintf(stderr, "         CL_DEVICE_PROFILE.................. %s\n", profile);

               char deviceVersion[2048];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_VERSION, sizeof(deviceVersion), &deviceVersion, NULL);
               fprintf(stderr, "         CL_DEVICE_VERSION.................. %s\n", deviceVersion);

               char driverVersion[2048];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DRIVER_VERSION, sizeof(driverVersion), &driverVersion, NULL);
               fprintf(stderr, "         CL_DRIVER_VERSION.................. %s\n", driverVersion);

               char cVersion[2048];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_OPENCL_C_VERSION, sizeof(cVersion), &cVersion, NULL);
               fprintf(stderr, "         CL_DEVICE_OPENCL_C_VERSION......... %s\n", cVersion);

               char name[2048];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_NAME, sizeof(name), &name, NULL);
               fprintf(stderr, "         CL_DEVICE_NAME..................... %s\n", name);
               char extensions[2048];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_EXTENSIONS, sizeof(extensions), &extensions, NULL);
               fprintf(stderr, "         CL_DEVICE_EXTENSIONS............... %s\n", extensions);
               char builtInKernels[2048];
               status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_BUILT_IN_KERNELS, sizeof(builtInKernels), &builtInKernels, NULL);
               fprintf(stderr, "         CL_DEVICE_BUILT_IN_KERNELS......... %s\n", builtInKernels);

               fprintf(stderr, "      }\n");
            }

         }
         fprintf(stderr, "   }\n");
      }
      fprintf(stderr, "}\n");
   }
}
int main(){
   info();
}

