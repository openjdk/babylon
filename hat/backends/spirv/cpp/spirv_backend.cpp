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
#include "shared.h"

class SpirvBackend : public Backend {
public:
    class SpirvConfig : public Backend::Config {
    public :
    };

    class SpirvProgram : public Backend::Program {
        class SpirvKernel : public Backend::Program::Kernel {
        public:
            SpirvKernel(Backend::Program *program, char *name)
                    : Backend::Program::Kernel(program, name) {
            }

            ~SpirvKernel() {
            }

            long ndrange(void *argArray) {
                std::cout << "spirv ndrange() " << std::endl;
                return 0;
            }
        };

    public:
        SpirvProgram(Backend *backend, BuildInfo *buildInfo)
                : Backend::Program(backend, buildInfo) {
        }

        ~SpirvProgram() {
        }

        long getKernel(int nameLen, char *name) {
            return (long) new SpirvKernel(this, name);
        }

        bool programOK() {
            return true;
        }
    };

public:

    SpirvBackend(SpirvConfig *spirvConfig, int spirvConfigSchemeLen, char *spirvBackendSchema)
            : Backend(spirvConfig, spirvConfigSchemeLen, spirvBackendSchema) {
        if (spirvConfig == nullptr) {
            std::cout << "spirvConfig == null" << std::endl;
        } else {
            std::cout << "spirvConfig != null" << std::endl;
        }
    }

    ~SpirvBackend() {
    }

    int getMaxComputeUnits() {
        std::cout << "spirv getMaxComputeUnits()" << std::endl;
        return 0;
    }

    void info() {
        std::cout << "spirv info()" << std::endl;
    }

    long compileProgram(int len, char *source) {
        std::cout << "spirv compileProgram()" << std::endl;
        size_t srcLen = ::strlen(source);
        char *src = new char[srcLen + 1];
        ::strncpy(src, source, srcLen);
        src[srcLen] = '\0';
        std::cout << "native compiling " << src << std::endl;
        return (long) new SpirvProgram(this, new BuildInfo(src, nullptr, false));
    }
};

long getBackend(void *config, int configSchemaLen, char *configSchema) {
    SpirvBackend::SpirvConfig *spirvConfig = (SpirvBackend::SpirvConfig *) config;
    return (long) new SpirvBackend(spirvConfig, configSchemaLen, configSchema);
}
