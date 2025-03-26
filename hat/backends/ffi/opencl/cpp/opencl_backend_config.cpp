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
#define opencl_backend_config_cpp
#include "opencl_backend.h"

OpenCLBackend::OpenCLConfig::OpenCLConfig(int configBits):
       configBits(configBits),
       minimizeCopies((configBits&MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT),
       alwaysCopy(!minimizeCopies),
       trace((configBits&TRACE_BIT)==TRACE_BIT),
       traceCopies((configBits&TRACE_COPIES_BIT)==TRACE_COPIES_BIT),
       traceEnqueues((configBits&TRACE_ENQUEUES_BIT)==TRACE_ENQUEUES_BIT),
       traceCalls((configBits&TRACE_CALLS_BIT)==TRACE_CALLS_BIT),
       traceSkippedCopies((configBits&TRACE_SKIPPED_COPIES_BIT)==TRACE_SKIPPED_COPIES_BIT),
       info((configBits&INFO_BIT)==INFO_BIT),
       showCode((configBits&SHOW_CODE_BIT)==SHOW_CODE_BIT),
       profile((configBits&PROFILE_BIT)==PROFILE_BIT),
       showWhy((configBits&SHOW_WHY_BIT)==SHOW_WHY_BIT),
       showState((configBits&SHOW_STATE_BIT)==SHOW_STATE_BIT),

       platform((configBits&0xf)),
       device((configBits&0xf0)>>4){
       if (info){
          std::cout << "native showCode " << showCode <<std::endl;
          std::cout << "native info " << info<<std::endl;
          std::cout << "native minimizeCopies " << minimizeCopies<<std::endl;
          std::cout << "native alwaysCopy " << alwaysCopy<<std::endl;
          std::cout << "native trace " << trace<<std::endl;
          std::cout << "native traceSkippedCopies " << traceSkippedCopies<<std::endl;
          std::cout << "native traceCalls " << traceCalls<<std::endl;
          std::cout << "native traceCopies " << traceCopies<<std::endl;
          std::cout << "native traceEnqueues " << traceEnqueues<<std::endl;
          std::cout << "native profile " << profile<<std::endl;
          std::cout << "native showWhy " << showWhy<<std::endl;
          std::cout << "native showState " << showState<<std::endl;
          std::cout << "native platform " << platform<<std::endl;
          std::cout << "native device " << device<<std::endl;
       }
 }
 OpenCLBackend::OpenCLConfig::~OpenCLConfig(){
 }
