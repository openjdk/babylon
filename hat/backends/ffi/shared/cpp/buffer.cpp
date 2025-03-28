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
#include <unistd.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <mutex>
#include "buffer.h"
#include "hex.h"

Buffer::Buffer()
        : max(0), memory(nullptr), size(0) {
  // std::cout << "Buffer() = "<< std::endl;
}

Buffer::Buffer(size_t size)
        : max(0), memory(nullptr), size(0) {
  // std::cout << "Buffer(size_t) = "<< std::endl;
    resize(size);
    ::memset(memory, '\0', size);
}

Buffer::Buffer(char *mem, size_t size)
        : max(0), memory(nullptr), size(0) {
  // std::cout << "Buffer(char * , size_t) = "<< std::endl;
   resize(size);
    ::memcpy(memory, mem, size);
}

Buffer::Buffer(char *fileName)
        : max(0), memory(nullptr), size(0) {
  // std::cout << "Buffer(char *) = "<< std::endl;
    struct stat st;
    stat(fileName, &st);
    if (S_ISREG(st.st_mode)) {
        int fd = ::open(fileName, O_RDONLY);
        read(fd, st.st_size);
        ::close(fd);
    }else{
       std::cout << "not reg file!"<< std::endl;
    }
}
Buffer::Buffer(std::string fileName)
        : max(0), memory(nullptr), size(0) {
 //  std::cout << "Buffer(std::string) = "<< std::endl;
    struct stat st;
    stat(fileName.c_str(), &st);
    if (S_ISREG(st.st_mode)) {
        int fd = ::open(fileName.c_str(), O_RDONLY);
        read(fd, st.st_size);
        ::close(fd);
    }else{
       std::cout << "not reg file!"<< std::endl;
    }
}

size_t Buffer::read(int fd, size_t fileSize) {
    resize(fileSize);
    size_t bytesRead = 0;
    size_t bytes = 0;
    while (bytesRead < size && (bytes = ::read(fd, memory + bytesRead, size - bytesRead)) >= 0) {
        bytesRead -= bytes;
    }
    return size;
}


void Buffer::resize(size_t newsize) {
    const static size_t CHUNK = 512;
    if ((newsize+1) > size) {
        // we are indeed asking to grow.
        if ((newsize+1)>=max) {
            max = (((newsize+1)%CHUNK)>0)?(((newsize+1)/CHUNK)+1)*CHUNK:(newsize+1); // should snap to CHUNK size
            if ((max %CHUNK)!=0 ){
                std::cerr <<" bad chunking" << std::endl;
                std::exit(1);
            }
            if (memory == nullptr){
                memory = new char[max];
            }else {
                char *newmemory = new char[max];
                ::memcpy(newmemory, memory, size);
                delete [] memory;
                memory = newmemory;
            }
        }
        size = newsize;
    }

}


void Buffer::dump(std::ostream &s) {
    Hex::bytes(s, memory, size, [&](auto &) {});
}

void Buffer::dump(std::ostream &s,  std::function<void(std::ostream &)> prefix) {
    Hex::bytes(s, memory, size, prefix);
}


Buffer::~Buffer() {
    if (memory != nullptr) {
        delete [] memory;
    }
}

char *Buffer::getStart() {return memory;}
char *Buffer::getEnd(){return memory+size;}
size_t Buffer::getSize(){return size;}
std::string Buffer::str(){return std::string(getStart(), getSize());}


size_t Buffer::write(int fd) {
    static size_t WRITECHUNK = 8192;
    size_t total = 0;
    while (total < size) {
        int toSend=  (size - total) > WRITECHUNK ? WRITECHUNK:(size-total) ;
        int bytesSent = ::write(fd, ((char *) memory) + total, toSend);
        if (bytesSent ==0) {
            std::cout << "0 bytes!" << std::endl;
        }
        if (bytesSent < 0) {
            std::cout << "error" << std::endl;
        }
        total += bytesSent;
    }
    return total;
}

GrowableBuffer::GrowableBuffer()
        :Buffer() {
}

GrowableBuffer::GrowableBuffer(size_t size)
        : Buffer(size) {
}

GrowableBuffer::GrowableBuffer(char *mem, size_t size)
        :  Buffer(mem, size)  {
}

GrowableBuffer::GrowableBuffer(char *fileName)
        :  Buffer(fileName)  {
}

void GrowableBuffer::add(void *contents, size_t bytes) {
    size_t oldsize = size;
    resize(size + bytes);
    ::memcpy(&(memory[oldsize]), contents, bytes);

}
void GrowableBuffer::add(char c) {
    size_t oldsize = size;
    resize(size + 1);
    memory[oldsize]=c;
    memory[size]='\0';
}
