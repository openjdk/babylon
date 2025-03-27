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
#include <vector>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <stack>

struct SchemaCursor {
private:
    std::stack<const char *> where;
public:
    char *ptr;
    SchemaCursor(char *ptr);
    virtual ~SchemaCursor();
    void in(const char * location);
    void out();
private:
    SchemaCursor *skipWhiteSpace();
    SchemaCursor *skipIdentifier();
public:
    void step(int count) ;

    bool peekAlpha();

    bool peekDigit() ;

    bool is(char ch);
    bool isColon();

    bool expect(char ch, const char *context,  int line ) ;
    bool expect(char ch,  int line ) ;
    bool expectDigit(const char *context,  int line );
    bool expectAlpha(const char *context,  int line );
    bool isEither(char ch1, char ch2, char*actual) ;
    void expectEither(char ch1, char ch2, char*actual, int line);

    int getInt() ;

    long getLong();

    char *getIdentifier();

    void error(std::ostream &ostream, const char *file, int line, const char *str);

};