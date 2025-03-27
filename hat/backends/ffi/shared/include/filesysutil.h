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

#include <sys/stat.h>
#include <dirent.h>

#include <unistd.h>
#include <fcntl.h>

#include <functional>
#include "buffer_cursor.h"


class FileSysUtil {
public:
    static void visit(const std::string & dirName, bool recurse, std::function<void(bool dir, std::string name)> visitor);

    static size_t size(const std::string & fileName);

    static bool isDir(const std::string & dirName);

    static bool isFile(const std::string & dirName);

    static bool isFileOrLink(const std::string & dirName);

    static bool removeFile(const std::string & dirName);

    static bool isFile(const std::string & dirName, const std::string & fileName);

    static bool isFileOrLink(const std::string & dirName, const std::string & fileName);

    static bool hasFileSuffix(const std::string & fileName, const std::string & suffix);

    static std::string getFileNameEndingWith(const std::string & dir, const std::string & suffix);

    static void mkdir_p(char *path);

    static std::string getFile(const std::string &path);

    static BufferCursor *getFileBufferCursor(const std::string &path);

    static void putFile(const std::string &path, const std::string &content);

    static void putFileBufferCursor(const std::string &path, BufferCursor *buffer);

    static void forEachLine(const std::string &path, std::function<void(std::string name)> visitor);

    static void forEachFileName(const std::string &path, std::function<void(std::string name)> visitor);

    static void forEachDirName(const std::string &path, std::function<void(std::string name)> visitor);

    static void send(int from, size_t, int to);

    static void send(const std::string &fileName, int to);
    static void send(char *fileName, int to);

};



