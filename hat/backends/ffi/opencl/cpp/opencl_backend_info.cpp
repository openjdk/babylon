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

#include "opencl_backend.h"

template<typename T>
static T info(cl_device_id device_id, cl_device_info device_info){
    T v;
    cl_int status = clGetDeviceInfo(device_id, device_info, sizeof(T), &v, nullptr);
    return v;
}

static char *strInfo(cl_device_id device_id, cl_device_info device_info){
    size_t sz;
    cl_int  status = clGetDeviceInfo(device_id, device_info, 0, nullptr,  &sz);
    char *ptr = new char[sz+1];
    status = clGetDeviceInfo(device_id, device_info, sz, ptr,nullptr);
    return ptr;
}

static char *strInfo(cl_platform_id platform_id, cl_platform_info platform_info){
    size_t sz;
    cl_int  status = clGetPlatformInfo(platform_id, platform_info, 0, nullptr,  &sz);
    char *ptr = new char[sz+1];
    status = clGetPlatformInfo(platform_id, platform_info, sz, ptr,nullptr);
    return ptr;
}

PlatformInfo::DeviceInfo::DeviceInfo(OpenCLBackend *openclBackend):
    openclBackend(openclBackend),
    maxComputeUnits(info<cl_int>(openclBackend->device_id,CL_DEVICE_MAX_COMPUTE_UNITS)),
    maxWorkItemDimensions(info<cl_int>(openclBackend->device_id,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)),
    maxWorkGroupSize(info<size_t>(openclBackend->device_id,CL_DEVICE_MAX_WORK_GROUP_SIZE)),
    maxWorkItemSizes( new size_t[maxWorkItemDimensions]),
    maxMemAllocSize(info<cl_ulong>(openclBackend->device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE)),
    globalMemSize(info<cl_ulong>(openclBackend->device_id,CL_DEVICE_GLOBAL_MEM_SIZE)),
    localMemSize(info<cl_ulong>(openclBackend->device_id,CL_DEVICE_LOCAL_MEM_SIZE)),
    profile(strInfo(openclBackend->device_id,CL_DEVICE_PROFILE)),
    deviceVersion(strInfo(openclBackend->device_id, CL_DEVICE_VERSION)),
    driverVersion(strInfo(openclBackend->device_id, CL_DRIVER_VERSION)),
    cVersion(strInfo(openclBackend->device_id, CL_DEVICE_OPENCL_C_VERSION)),
    name(strInfo(openclBackend->device_id, CL_DEVICE_NAME)),
    extensions(strInfo(openclBackend->device_id, CL_DEVICE_EXTENSIONS)),
    builtInKernels(strInfo(openclBackend->device_id,CL_DEVICE_BUILT_IN_KERNELS)){

    clGetDeviceInfo(openclBackend->device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxWorkItemDimensions, maxWorkItemSizes, NULL);
    clGetDeviceInfo(openclBackend->device_id, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
    char buf[512];
    buf[0]='\0';
    if (CL_DEVICE_TYPE_CPU == (deviceType & CL_DEVICE_TYPE_CPU)) {
       std::strcat(buf, "CPU ");
    }
    if (CL_DEVICE_TYPE_GPU == (deviceType & CL_DEVICE_TYPE_GPU)) {
       std::strcat(buf, "GPU ");
    }
    if (CL_DEVICE_TYPE_ACCELERATOR == (deviceType & CL_DEVICE_TYPE_ACCELERATOR)) {
       std::strcat(buf, "ACC ");
    }
    deviceTypeStr = new char[std::strlen(buf)];
    std::strcpy(deviceTypeStr, buf);
}

PlatformInfo::DeviceInfo::~DeviceInfo(){
    delete [] deviceTypeStr;
    delete [] profile;
    delete [] deviceVersion;
    delete [] driverVersion;
    delete [] cVersion;
    delete [] name;
    delete [] extensions;
    delete [] builtInKernels;
    delete [] maxWorkItemSizes;
}

PlatformInfo::PlatformInfo(OpenCLBackend *openclBackend):
    openclBackend(openclBackend),
    versionName(strInfo(openclBackend->platform_id, CL_PLATFORM_VERSION)),
    vendorName(strInfo(openclBackend->platform_id, CL_PLATFORM_VENDOR)),
    name(strInfo(openclBackend->platform_id, CL_PLATFORM_NAME)),
    deviceInfo(openclBackend){
}
PlatformInfo::~PlatformInfo(){
    delete [] versionName;
    delete [] vendorName;
    delete [] name;
}


void OpenCLBackend::info() {
    PlatformInfo platformInfo(this);
    cl_int status;
    std::cerr << "platform{" <<std::endl;
    std::cerr << "   CL_PLATFORM_VENDOR..\"" << platformInfo.vendorName <<"\""<<std::endl;
    std::cerr << "   CL_PLATFORM_VERSION.\"" << platformInfo.versionName <<"\""<<std::endl;
    std::cerr << "   CL_PLATFORM_NAME....\"" << platformInfo.name <<"\""<<std::endl;
    std::cerr << "         CL_DEVICE_TYPE..................... " <<  platformInfo.deviceInfo.deviceTypeStr << " "<<  platformInfo.deviceInfo.deviceType<<std::endl;
    std::cerr << "         CL_DEVICE_MAX_COMPUTE_UNITS........ " <<  platformInfo.deviceInfo.maxComputeUnits<<std::endl;
    std::cerr << "         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. " <<  platformInfo.deviceInfo.maxWorkItemDimensions << " {";
    for (unsigned dimIdx = 0; dimIdx <  platformInfo.deviceInfo.maxWorkItemDimensions; dimIdx++) {
        std::cerr<<  platformInfo.deviceInfo.maxWorkItemSizes[dimIdx] << " ";
    }
    std::cerr<< "}"<<std::endl;
    std::cerr <<  "         CL_DEVICE_MAX_WORK_GROUP_SIZE...... "<<  platformInfo.deviceInfo.maxWorkGroupSize<<std::endl;
    std::cerr <<  "         CL_DEVICE_MAX_MEM_ALLOC_SIZE....... "<<  platformInfo.deviceInfo.maxMemAllocSize<<std::endl;
    std::cerr <<  "         CL_DEVICE_GLOBAL_MEM_SIZE.......... "<<  platformInfo.deviceInfo.globalMemSize<<std::endl;
    std::cerr <<  "         CL_DEVICE_LOCAL_MEM_SIZE........... "<<  platformInfo.deviceInfo.localMemSize<<std::endl;
    std::cerr <<  "         CL_DEVICE_PROFILE.................. "<<  platformInfo.deviceInfo.profile<<std::endl;
    std::cerr <<  "         CL_DEVICE_VERSION.................. "<<  platformInfo.deviceInfo.deviceVersion<<std::endl;
    std::cerr <<  "         CL_DRIVER_VERSION.................. "<<  platformInfo.deviceInfo.driverVersion<<std::endl;
    std::cerr <<  "         CL_DEVICE_OPENCL_C_VERSION......... "<<  platformInfo.deviceInfo.cVersion<<std::endl;
    std::cerr <<  "         CL_DEVICE_NAME..................... "<<  platformInfo.deviceInfo.name<<std::endl;
    std::cerr <<  "         CL_DEVICE_EXTENSIONS............... "<<  platformInfo.deviceInfo.extensions<<std::endl;
    std::cerr <<  "         CL_DEVICE_BUILT_IN_KERNELS......... "<<  platformInfo.deviceInfo.builtInKernels<<std::endl;
    std::cerr <<  "}"<<std::endl;
}

int OpenCLBackend::getMaxComputeUnits() {
    PlatformInfo platformInfo(this);
    return platformInfo.deviceInfo.maxComputeUnits;
}

