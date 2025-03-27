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
 #pragma once

#include <cstring>
#include <stdlib.h>
#include <ostream>
#include <functional>

class PureMark {
public:
    virtual char *getStart() = 0;
    virtual ~PureMark(){}

};
class PureRange : public PureMark{
public:
    virtual char *getEnd() = 0;
    virtual size_t getSize() = 0;
    virtual ~PureRange(){

    }
};

class Buffer : public PureRange{
private:
    size_t max;  // max size before we need to realloc.
protected:
    char *memory;
    size_t size;  // size requested
public:
    Buffer();

    Buffer(size_t size);

    Buffer(char *mem, size_t size);

    Buffer(char *fileName);

    Buffer(std::string fileName);

    void resize(size_t size);

    void dump(std::ostream &s);

    void dump(std::ostream &s, std::function<void(std::ostream &)> prefix );

    size_t write(int fd);

    size_t read(int fd, size_t size);

    virtual ~Buffer();

   // char *getPtr();
    char *getStart() override;
    char *getEnd() override;
    size_t getSize() override;
    std::string str();
};

class GrowableBuffer : public Buffer {
public:
    GrowableBuffer();

    GrowableBuffer(size_t size);

    GrowableBuffer(char *mem, size_t size);

    GrowableBuffer(char *fileName);

    void add(void *contents, size_t bytes);

    void add(char c);

};