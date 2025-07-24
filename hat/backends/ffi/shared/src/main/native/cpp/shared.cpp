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
#include <fstream>
#define shared_cpp

#include "shared.h"

#define INFO 0


void hexdump(void *ptr, int buflen) {
    auto *buf = static_cast<unsigned char *>(ptr);
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
        KernelArg *arg = argSled.arg(i);
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
                std::cerr << "unexpected variant (shared.cpp) '" << static_cast<char>(arg->variant) << "'" << std::endl;
                exit(1);
            }
        }
    }
    out << "schema len = " << argSled.schemaLen() << std::endl;

    out << "schema = " << argSled.schema() << std::endl;
}


extern "C" void info(long backendHandle) {
    if (INFO) {
        std::cout << "trampolining through backendHandle to backend.info()" << std::endl;
    }
    auto *backend = reinterpret_cast<Backend *>(backendHandle);
    backend->info();
}

extern "C" void computeStart(long backendHandle) {
    if (INFO) {
        std::cout << "trampolining through backendHandle to backend.computeStart()" << std::endl;
    }
    auto *backend = reinterpret_cast<Backend *>(backendHandle);
    backend->computeStart();
}

extern "C" void computeEnd(long backendHandle) {
    if (INFO) {
        std::cout << "trampolining through backendHandle to backend.computeEnd()" << std::endl;
    }
    auto *backend = reinterpret_cast<Backend *>(backendHandle);
    backend->computeEnd();
}

extern "C" void releaseBackend(long backendHandle) {
    auto *backend = reinterpret_cast<Backend *>(backendHandle);
    delete backend;
}

extern "C" long compile(long backendHandle, int len, char *source) {
    if (INFO) {
        std::cout << "trampolining through backendHandle to backend.compile() "
                << std::hex << backendHandle << std::dec << std::endl;
    }
    auto *backend = reinterpret_cast<Backend *>(backendHandle);
    long compilationUnitHandle = reinterpret_cast<long>(backend->compile(len, source));
    if (INFO) {
        std::cout << "compilationUnitHandle = " << std::hex << compilationUnitHandle << std::dec << std::endl;
    }
    return compilationUnitHandle;
}

extern "C" long getKernel(long compilationUnitHandle, int nameLen, char *name) {
    if (INFO) {
        std::cout << "trampolining through programHandle to compilationUnit.getKernel()"
                << std::hex << compilationUnitHandle << std::dec << std::endl;
    }
    auto compilationUnit = reinterpret_cast<Backend::CompilationUnit *>(compilationUnitHandle);
    return reinterpret_cast<long>(compilationUnit->getKernel(nameLen, name));
}

extern "C" long ndrange(long kernelHandle, void *argArray) {
    if (INFO) {
        std::cout << "trampolining through kernelHandle to kernel.ndrange(...) " << std::endl;
    }
    auto kernel = reinterpret_cast<Backend::CompilationUnit::Kernel *>(kernelHandle);
    kernel->ndrange(argArray);
    return (long) 0;
}

extern "C" void releaseKernel(long kernelHandle) {
    if (INFO) {
        std::cout << "trampolining through to releaseKernel " << std::endl;
    }
    auto kernel = reinterpret_cast<Backend::CompilationUnit::Kernel *>(kernelHandle);
    delete kernel;
}

extern "C" void releaseCompilationUnit(long compilationUnitHandle) {
    if (INFO) {
        std::cout << "trampolining through to releaseCompilationUnit " << std::endl;
    }
    auto compilationUnit = reinterpret_cast<Backend::CompilationUnit *>(compilationUnitHandle);
    delete compilationUnit;
}

extern "C" bool compilationUnitOK(long compilationUnitHandle) {
    if (INFO) {
        std::cout << "trampolining through to compilationUnitHandleOK " << std::endl;
    }
    auto compilationUnit = reinterpret_cast<Backend::CompilationUnit *>(compilationUnitHandle);
    return compilationUnit->compilationUnitOK();
}

extern "C" bool getBufferFromDeviceIfDirty(long backendHandle, long memorySegmentHandle, long memorySegmentLength) {
    if (INFO) {
        std::cout << "trampolining through to getBuffer " << std::endl;
    }
    auto backend = reinterpret_cast<Backend *>(backendHandle);
    auto memorySegment = reinterpret_cast<void *>(memorySegmentHandle);
    return backend->getBufferFromDeviceIfDirty(memorySegment, memorySegmentLength);
}


Backend::Config::Config(int configBits):BasicConfig(configBits) {

}

Backend::Config::~Config() = default;

Backend::Queue::Queue(Backend *backend)
    : backend(backend) {
}

Backend::Queue::~Queue() = default;

Text::Text(size_t len, char *text, bool isCopy)
    : len(len), text(text), isCopy(isCopy) {
    // std::cout << "in Text len="<<len<<" isCopy="<<isCopy << std::endl;
}

Text::Text(char *text, bool isCopy)
    : len(std::strlen(text)), text(text), isCopy(isCopy) {
    // std::cout << "in Text len="<<len<<" isCopy="<<isCopy << std::endl;
}

Text::Text(size_t len)
    : len(len), text(len > 0 ? new char[len] : nullptr), isCopy(true) {
    //  std::cout << "in Text len="<<len<<" isCopy="<<isCopy << std::endl;
}

void Text::write(const std::string &filename) const {
    std::ofstream out;
    out.open(filename, std::ofstream::trunc);
    out.write(text, len);
    out.close();
}

void Text::read(const std::string &filename) {
    if (isCopy && text) {
        delete[] text;
    }
    text = nullptr;
    isCopy = false;
    // std::cout << "reading from " << filename << std::endl;

    std::ifstream ptxStream;
    ptxStream.open(filename);


    ptxStream.seekg(0, std::ios::end);
    len = ptxStream.tellg();
    ptxStream.seekg(0, std::ios::beg);

    if (len > 0) {
        text = new char[len];
        isCopy = true;
        //std::cerr << "about to read  " << len << std::endl;
        ptxStream.read(text, len);
        ptxStream.close();
        //std::cerr << "read  " << len << std::endl;
        text[len - 1] = '\0';
        //std::cerr << "read text " << text << std::endl;
    }
}

Text::~Text() {
    if (isCopy && text) {
        delete[] text;
    }
    text = nullptr;
    isCopy = false;
    len = 0;
}

Log::Log(const size_t len)
    : Text(len) {
}

Log::Log(char *text)
    : Text(text, false) {
}

long Backend::CompilationUnit::Kernel::ndrange(void *argArray) {
    if (compilationUnit->backend->config->traceCalls) {
        std::cout << "kernelContext(\"" << name << "\"){" << std::endl;
    }
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    auto *profilableQueue = dynamic_cast<ProfilableQueue *>(compilationUnit->backend->queue);
    if (profilableQueue != nullptr) {
        profilableQueue->marker(ProfilableQueue::EnterKernelDispatchBits, name);
    }
    if (compilationUnit->backend->config->trace) {
        Sled::show(std::cout, argArray);
    }
    KernelContext *kernelContext = nullptr;
    for (int i = 0; i < argSled.argc(); i++) {
        KernelArg *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
                if (arg->idx == 0) {
                    kernelContext = static_cast<KernelContext *>(arg->value.buffer.memorySegment);
                }
                if (compilationUnit->backend->config->trace) {
                    std::cout << "arg[" << i << "] = " << std::hex << (int) (arg->value.buffer.access);
                    switch (arg->value.buffer.access) {
                        case RO_BYTE:
                            std::cout << " RO";
                            break;
                        case WO_BYTE:
                            std::cout << " WO";
                            break;
                        case RW_BYTE:
                            std::cout << " RW";
                            break;
                    }
                    std::cout << std::endl;
                }

                BufferState *bufferState = BufferState::of(arg);

                Buffer *buffer = compilationUnit->backend->getOrCreateBuffer(bufferState);

                bool kernelReadsFromThisArg = (arg->value.buffer.access == RW_BYTE) || (
                                                  arg->value.buffer.access == RO_BYTE);
                bool copyToDevice =
                        compilationUnit->backend->config->alwaysCopy
                        || (bufferState->state == BufferState::NEW_STATE)
                        || ((bufferState->state == BufferState::HOST_OWNED)
                        );

                if (compilationUnit->backend->config->showWhy) {
                    std::cout <<
                            "config.alwaysCopy=" << compilationUnit->backend->config->alwaysCopy
                            << " | arg.RW=" << (arg->value.buffer.access == RW_BYTE)
                            << " | arg.RO=" << (arg->value.buffer.access == RO_BYTE)
                            << " | kernel.needsToRead=" << kernelReadsFromThisArg
                            << " | Buffer state = " << BufferState::stateNames[bufferState->state]
                            << " so ";
                }
                if (copyToDevice) {
                    compilationUnit->backend->queue->copyToDevice(buffer);
                    // buffer->copyToDevice();
                    if (compilationUnit->backend->config->traceCopies) {
                        std::cout << "copying arg " << arg->idx << " to device " << std::endl;
                    }
                } else {
                    if (compilationUnit->backend->config->traceSkippedCopies) {
                        std::cout << "NOT copying arg " << arg->idx << " to device " << std::endl;
                    }
                }
                setArg(arg, buffer);
                if (compilationUnit->backend->config->trace) {
                    std::cout << "set buffer arg " << arg->idx << std::endl;
                }
                break;
            }
            case 'B':
            case 'S':
            case 'C':
            case 'I':
            case 'F':
            case 'J':
            case 'D': {
                setArg(arg);
                if (compilationUnit->backend->config->trace) {
                    std::cerr << "set " << arg->variant << " " << arg->idx << std::endl;
                }
                break;
            }
            default: {
                std::cerr << "unexpected variant setting args in OpenCLkernel::kernelContext " << (char) arg->variant <<
                        std::endl;
                exit(1);
            }
        }
    }

    if (kernelContext == nullptr) {
        std::cerr << "Looks like we recieved a kernel dispatch with xero args kernel='" << name << "'" << std::endl;
        exit(1);
    }

    if (compilationUnit->backend->config->trace) {
        std::cout << "kernelContext = " << kernelContext->maxX << std::endl;
    }

    // We 'double dispatch' back to the kernel to actually do the dispatch

    compilationUnit->backend->queue->dispatch(kernelContext, this);


    for (int i = 0; i < argSled.argc(); i++) {
        // note i = 1... we never need to copy back the KernelContext
        KernelArg *arg = argSled.arg(i);
        if (arg->variant == '&') {
            BufferState *bufferState = BufferState::of(arg);

            bool kernelWroteToThisArg = (arg->value.buffer.access == WO_BYTE) | (arg->value.buffer.access == RW_BYTE);
            if (compilationUnit->backend->config->showWhy) {
                std::cout <<
                        "config.alwaysCopy=" << compilationUnit->backend->config->alwaysCopy
                        << " | arg.WO=" << (arg->value.buffer.access == WO_BYTE)
                        << " | arg.RW=" << (arg->value.buffer.access == RW_BYTE)
                        << " | kernel.wroteToThisArg=" << kernelWroteToThisArg
                        << "Buffer state = " << BufferState::stateNames[bufferState->state]
                        << " so ";
            }

            auto *buffer = static_cast<Buffer *>(bufferState->vendorPtr);
            if (compilationUnit->backend->config->alwaysCopy) {
                compilationUnit->backend->queue->copyFromDevice(buffer);
                // buffer->copyFromDevice();
                if (compilationUnit->backend->config->traceCopies || compilationUnit->backend->config->traceEnqueues) {
                    std::cout << "copying arg " << arg->idx << " from device " << std::endl;
                }
            } else {
                if (compilationUnit->backend->config->traceSkippedCopies) {
                    std::cout << "NOT copying arg " << arg->idx << " from device " << std::endl;
                }
                if (kernelWroteToThisArg) {
                    bufferState->state = BufferState::DEVICE_OWNED;
                }
            }
        }
    }
    if (profilableQueue != nullptr) {
        profilableQueue->marker(Backend::ProfilableQueue::LeaveKernelDispatchBits, name);
    }
    compilationUnit->backend->queue->wait();
    compilationUnit->backend->queue->release();
    if (compilationUnit->backend->config->traceCalls) {
        std::cout << "\"" << name << "\"}" << std::endl;
    }
    return 0;
}
