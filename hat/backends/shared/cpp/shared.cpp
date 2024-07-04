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

/*
extern void dump(FILE *file, size_t len, void *ptr) {
    for (int i = 0; i < len; i++) {
        if (i % 16 == 0) {
            if (i != 0) {
                fprintf(file, "\n");
            }
            fprintf(file, "%lx :", ((unsigned long) ptr) + i);
        }
        fprintf(file, " %02x", ((int) (((unsigned char *) ptr)[i]) & 0xff));
    }
}

*/
/*extern "C" void dumpArgArray(void *ptr) {
    ArgSled argSled((ArgArray_t *) ptr);
    std::cout << "ArgArray->argc = " << argSled.argc() << std::endl;
    for (int i = 0; i < argSled.argc(); i++) {
        argSled.dumpArg(i);
    }
    std::cout << "schema = " << argSled.schema() << std::endl;
}*/

void hexdump(void *ptr, int buflen) {
    unsigned char *buf = (unsigned char *) ptr;
    int i, j;
    for (i = 0; i < buflen; i += 16) {
        printf("%06x: ", i);
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%02x ", buf[i + j]);
            else
                printf("   ");
        printf(" ");
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%c", isprint(buf[i + j]) ? buf[i + j] : '.');
        printf("\n");
    }
}

void Sled::show(std::ostream &out, void *argArray) {
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
                out << "Buf: of " << arg->value.buffer.sizeInBytes << " bytes " << std::endl;
                break;
            }
            case 'B': {
                out << "S8:" << arg->value.s8 << std::endl;
                break;
            }
            case 'Z': {
                out << "Z:" << arg->value.z1 << std::endl;
                break;
            }
            case 'C': {
                out << "U16:" << arg->value.u16 << std::endl;
                break;
            }
            case 'S': {
                out << "S16:" << arg->value.s16 << std::endl;
                break;
            }
            case 'I': {
                out << "S32:" << arg->value.s32 << std::endl;
                break;
            }
            case 'F': {
                out << "F32:" << arg->value.f32 << std::endl;
                break;
            }
            case 'J': {
                out << "S64:" << arg->value.s64 << std::endl;
                break;
            }
            case 'D': {
                out << "F64:" << arg->value.f64 << std::endl;
                break;
            }
            default: {
                std::cerr << "unexpected variant '" << (char) arg->variant << "'" << std::endl;
                exit(1);
            }
        }
    }
    out << "schema len = " << argSled.schemaLen() << std::endl;

    out << "schema = " << argSled.schema() << std::endl;
}


extern "C" int getMaxComputeUnits(long backendHandle) {
   // std::cout << "trampolining through backendHandle to backend.getMaxComputeUnits()" << std::endl;
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    return backend->getMaxComputeUnits();
}

extern "C" void info(long backendHandle) {
  //  std::cout << "trampolining through backendHandle to backend.info()" << std::endl;
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    backend->info();
}
extern "C" void releaseBackend(long backendHandle) {
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    delete backend;
}
extern "C" long compileProgram(long backendHandle, int len, char *source) {
    std::cout << "trampolining through backendHandle to backend.compileProgram() "
        <<std::hex<<backendHandle<< std::dec <<std::endl;
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    auto programHandle = backend->compileProgram(len, source);
    std::cout << "programHandle = "<<std::hex<<backendHandle<< std::dec <<std::endl;
    return programHandle;
}
extern "C" long getKernel(long programHandle, int nameLen, char *name) {
    std::cout << "trampolining through programHandle to program.getKernel()"
            <<std::hex<<programHandle<< std::dec <<std::endl;
    auto program = reinterpret_cast<Backend::Program *>(programHandle);
    return program->getKernel(nameLen, name);
}

extern "C" long ndrange(long kernelHandle, void *argArray) {
    std::cout << "trampolining through kernelHandle to kernel.ndrange(...) " << std::endl;
    auto kernel = reinterpret_cast<Backend::Program::Kernel *>(kernelHandle);
    kernel->ndrange( argArray);
    return (long) 0;
}
extern "C" void releaseKernel(long kernelHandle) {
    auto kernel = reinterpret_cast<Backend::Program::Kernel *>(kernelHandle);
    delete kernel;
}

extern "C" void releaseProgram(long programHandle) {
    auto program = reinterpret_cast<Backend::Program *>(programHandle);
    delete program;
}
extern "C" bool programOK(long programHandle) {
    auto program = reinterpret_cast<Backend::Program *>(programHandle);
    return program->programOK();
}


