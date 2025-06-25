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

#include "schema_cursor.h"
#include <iostream>
#include<cstring>


SchemaCursor::SchemaCursor(char *ptr): ptr(ptr) {
}

SchemaCursor::~SchemaCursor() = default;

void SchemaCursor::in(const char *location) {
    where.push(location);
}

void SchemaCursor::out() {
    where.pop();
}

SchemaCursor *SchemaCursor::skipWhiteSpace() {
    while (*ptr == ' ' || *ptr == '\n' || *ptr == '\t') {
        step(1);
    }
    return this;
}


SchemaCursor *SchemaCursor::skipIdentifier() {
    while (peekAlpha() || peekDigit()) {
        step(1);
    }
    return this;
}

void SchemaCursor::step(int count) {
    while (count--) {
        ptr++;
    }
}

bool SchemaCursor::peekAlpha() {
    skipWhiteSpace();
    return (::isalpha(*ptr));
}

bool SchemaCursor::peekDigit() {
    skipWhiteSpace();
    return (::isdigit(*ptr));
}

bool SchemaCursor::is(const char ch) {
    skipWhiteSpace();
    if (*ptr == ch) {
        step(1);
        return true;
    }
    return false;
}

bool SchemaCursor::isColon() {
    return is(':');
}

bool SchemaCursor::expect(char ch, const char *context, int line) {
    if (is(ch)) {
        return true;
    }
    if (!where.empty()) {
        std::cerr << where.top() << " ";
    }
    std::cerr << "@" << line << ": parse error expecting  '" << ch << "' " << context << " looking at " << ptr <<
            std::endl;
   // exit(1);
    return false;
}

bool SchemaCursor::expect(const char ch, const int line) {
    return expect(ch, "", line);
}

bool SchemaCursor::expectDigit(const char *context, const int line) {
    if (::isdigit(*ptr)) {
        return true;
    }
    if (!where.empty()) {
        std::cerr << where.top() << " ";
    }
    std::cerr << "@" << line << ": parse error expecting digit " << context << " looking at " << ptr << std::endl;
   // exit(1);
    return false;
}

bool SchemaCursor::expectAlpha(const char *context, const int line) {
    if (::isalpha(*ptr)) {
        return true;
    }
    if (!where.empty()) {
        std::cerr << where.top() << " ";
    }
    std::cerr << "@" << line << ": parse error expecting alpha " << context << " looking at " << ptr << std::endl;
 //   exit(1);
    return false;
}

bool SchemaCursor::isEither(const char ch1, const char ch2, char *actual) {
    skipWhiteSpace();
    if (*ptr == ch1 || *ptr == ch2) {
        step(1);
        *actual = *ptr;
        return true;
    }
    return false;
}

void SchemaCursor::expectEither(const char ch1, const char ch2, char *actual, const int line) {
    skipWhiteSpace();
    if (*ptr == ch1 || *ptr == ch2) {
        step(1);
        *actual = *ptr;
        return;
    }
    if (!where.empty()) {
        std::cerr << where.top() << " ";
    }
    std::cerr << "@" << line << ": parse error expecting  '" << ch1 << "' or '" << ch2 << "'  looking at " << ptr <<
            std::endl;
    exit(1);
}

int SchemaCursor::getInt() {
    const int value = *ptr - '0';
    step(1);
    if (peekDigit()) {
        return value * 10 + getInt();
    }
    return value;
}

long SchemaCursor::getLong() {
    const long value = *ptr - '0';
    step(1);
    if (peekDigit()) {
        return value * 10 + getLong();
    }
    return value;
}

char *SchemaCursor::getIdentifier() {
    const char *identifierStart = ptr;
    skipIdentifier();
    const size_t len = ptr - identifierStart;
    auto identifier = new char[len + 1];
    memcpy(identifier, identifierStart, len);
    identifier[len] = '\0';
    return identifier;
}

void SchemaCursor::error(std::ostream &ostream, const char *file,  const int line, const char *str) const {
    ostream << file << ":" << "@" << line << ": parse error " << str << " looking at " << ptr << std::endl;
    exit(1);
}
