/*
* Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
/*
You probably should not edit this this file!!!
It was auto generated 2025-10-02 14:07:12.618 by hat.FFIConfigCreator
*/
#pragma once

#include <iostream>


struct BasicConfig{
    static constexpr int MINIMIZE_COPIES_BIT              = 1<<0x8;
    static constexpr int TRACE_BIT                        = 1<<0x9;
    static constexpr int PROFILE_BIT                      = 1<<0xa;
    static constexpr int SHOW_CODE_BIT                    = 1<<0xb;
    static constexpr int SHOW_KERNEL_MODEL_BIT            = 1<<0xc;
    static constexpr int SHOW_COMPUTE_MODEL_BIT           = 1<<0xd;
    static constexpr int INFO_BIT                         = 1<<0xe;
    static constexpr int TRACE_COPIES_BIT                 = 1<<0xf;
    static constexpr int TRACE_SKIPPED_COPIES_BIT         = 1<<0x10;
    static constexpr int TRACE_ENQUEUES_BIT               = 1<<0x11;
    static constexpr int TRACE_CALLS_BIT                  = 1<<0x12;
    static constexpr int SHOW_WHY_BIT                     = 1<<0x13;
    static constexpr int SHOW_STATE_BIT                   = 1<<0x14;
    static constexpr int PTX_BIT                          = 1<<0x15;
    static constexpr int INTERPRET_BIT                    = 1<<0x16;
    static constexpr int NO_BUFFER_TAGGING_BIT            = 1<<0x17;
    static constexpr int NO_DIALECT_BIT                   = 1<<0x18;
    static constexpr int NO_MODULE_OP_BIT                 = 1<<0x19;
    static constexpr int HEADLESS_BIT                     = 1<<0x1a;
    const static char *bitNames[]; // See below for initialization
    const static char *bitDescriptions[]; // See below for initialization
    int configBits;
    bool minimizeCopies;
    bool trace;
    bool profile;
    bool showCode;
    bool showKernelModel;
    bool showComputeModel;
    bool info;
    bool traceCopies;
    bool traceSkippedCopies;
    bool traceEnqueues;
    bool traceCalls;
    bool showWhy;
    bool showState;
    bool ptx;
    bool interpret;
    bool noBufferTagging;
    bool noDialect;
    bool noModuleOp;
    bool headless;
    int platform;
    int device;
    bool alwaysCopy;
    explicit BasicConfig(int configBits):
        configBits(configBits),
        minimizeCopies((configBits & MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT),
        trace((configBits & TRACE_BIT)==TRACE_BIT),
        profile((configBits & PROFILE_BIT)==PROFILE_BIT),
        showCode((configBits & SHOW_CODE_BIT)==SHOW_CODE_BIT),
        showKernelModel((configBits & SHOW_KERNEL_MODEL_BIT)==SHOW_KERNEL_MODEL_BIT),
        showComputeModel((configBits & SHOW_COMPUTE_MODEL_BIT)==SHOW_COMPUTE_MODEL_BIT),
        info((configBits & INFO_BIT)==INFO_BIT),
        traceCopies((configBits & TRACE_COPIES_BIT)==TRACE_COPIES_BIT),
        traceSkippedCopies((configBits & TRACE_SKIPPED_COPIES_BIT)==TRACE_SKIPPED_COPIES_BIT),
        traceEnqueues((configBits & TRACE_ENQUEUES_BIT)==TRACE_ENQUEUES_BIT),
        traceCalls((configBits & TRACE_CALLS_BIT)==TRACE_CALLS_BIT),
        showWhy((configBits & SHOW_WHY_BIT)==SHOW_WHY_BIT),
        showState((configBits & SHOW_STATE_BIT)==SHOW_STATE_BIT),
        ptx((configBits & PTX_BIT)==PTX_BIT),
        interpret((configBits & INTERPRET_BIT)==INTERPRET_BIT),
        noBufferTagging((configBits & NO_BUFFER_TAGGING_BIT)==NO_BUFFER_TAGGING_BIT),
        noDialect((configBits & NO_DIALECT_BIT)==NO_DIALECT_BIT),
        noModuleOp((configBits & NO_MODULE_OP_BIT)==NO_MODULE_OP_BIT),
        headless((configBits & HEADLESS_BIT)==HEADLESS_BIT),
        platform(configBits & 0xf),
        alwaysCopy(!minimizeCopies),
        device((configBits & 0xf0) >> 4){
            if(info){
                std::cout << "native minimizeCopies " << minimizeCopies << std::endl;
                std::cout << "native trace " << trace << std::endl;
                std::cout << "native profile " << profile << std::endl;
                std::cout << "native showCode " << showCode << std::endl;
                std::cout << "native showKernelModel " << showKernelModel << std::endl;
                std::cout << "native showComputeModel " << showComputeModel << std::endl;
                std::cout << "native info " << info << std::endl;
                std::cout << "native traceCopies " << traceCopies << std::endl;
                std::cout << "native traceSkippedCopies " << traceSkippedCopies << std::endl;
                std::cout << "native traceEnqueues " << traceEnqueues << std::endl;
                std::cout << "native traceCalls " << traceCalls << std::endl;
                std::cout << "native showWhy " << showWhy << std::endl;
                std::cout << "native showState " << showState << std::endl;
                std::cout << "native ptx " << ptx << std::endl;
                std::cout << "native interpret " << interpret << std::endl;
                std::cout << "native noBufferTagging " << noBufferTagging << std::endl;
                std::cout << "native noDialect " << noDialect << std::endl;
                std::cout << "native noModuleOp " << noModuleOp << std::endl;
                std::cout << "native headless " << headless << std::endl;
                std::cout << "native platform " << platform << std::endl;
                std::cout << "native device " << device << std::endl;
            }
        }
    virtual ~BasicConfig()= default;
};

#ifdef shared_cpp
const char *BasicConfig::bitNames[]={
    "MINIMIZE_COPIES_BIT",
    "TRACE_BIT",
    "PROFILE_BIT",
    "SHOW_CODE_BIT",
    "SHOW_KERNEL_MODEL_BIT",
    "SHOW_COMPUTE_MODEL_BIT",
    "INFO_BIT",
    "TRACE_COPIES_BIT",
    "TRACE_SKIPPED_COPIES_BIT",
    "TRACE_ENQUEUES_BIT",
    "TRACE_CALLS_BIT",
    "SHOW_WHY_BIT",
    "SHOW_STATE_BIT",
    "PTX_BIT",
    "INTERPRET_BIT",
    "NO_BUFFER_TAGGING_BIT",
    "NO_DIALECT_BIT",
    "NO_MODULE_OP_BIT",
    "HEADLESS_BIT",
};
const char *BasicConfig::bitDescriptions[]={
    "FFI ONLY Try to minimize copies",
    "FFI ONLY trace code",
    "FFI ONLY Turn on profiling",
    "Show generated code (PTX/OpenCL/CUDA)",
    "Show (via OpWriter) Kernel Model",
    "Show (via OpWriter) Compute Model",
    "FFI ONLY Show platform and device info",
    "FFI ONLY trace copies",
    "FFI ONLY Trace skipped copies (see MINIMIZE_COPIES) ",
    "FFI ONLY trace enqueued tasks",
    "FFI ONLY trace calls (enter/leave)",
    "FFI ONLY show why we decided to copy buffer (H to D)",
    "Show iface buffer state changes",
    "FFI (NVIDIA) ONLY pass PTX rather than C99 CUDA code",
    "Interpret the code model rather than converting to bytecode",
    "Skip AUTO buffer tagging (rely on annotations)",
    "Skip generating HAT dialect ops",
    "Use original callgraph (not using Module Op)",
    "Don't show UI",
};
#endif
