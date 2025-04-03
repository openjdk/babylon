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

#include <string>
#include <vector>
#include "buffer.h"


class BufferCursor;

class Range;

class Mark : public PureRange {
private:
    BufferCursor *cursor;
    char *ptr;
    char *end;

private:
    friend BufferCursor;
    Mark(BufferCursor *);

public:
    char *getStart() override;
    char *getEnd() override;
    size_t getSize() override ;
    char *setEnd();
    std::string str(char *end);
    std::string str();
    std::string str(int delta);
};

class BufferCursor : public PureRange {
protected:
    char *startPtr, *ptr, *endPtr;
    std::vector<Mark *> marks;
public:

    char *get();

    char *getStart() override;
    char *getEnd() override;
    size_t getSize() override ;

    BufferCursor *moveToOffset(int offset);

    virtual bool isValid(char *p);

    bool end();

    BufferCursor *advance(int i);

    BufferCursor *advance();

    BufferCursor *backup(int i);

    BufferCursor *backup();

    char ch();

    int chAsDigit();

    int chAsHexDigit();

    char chAndAdvance();

    bool isLookingAt(const char c);

    BufferCursor *skipWhiteSpace();

    BufferCursor *skipWhiteSpaceOrNewLine();

    bool isLookingAt(const char *str);

    bool isLookingAtAndStepOver(const char *str);

    BufferCursor *skipUntilLookingAt(const char *str);

    BufferCursor *backupUntilLookingAt(const char *str);

    BufferCursor *skipUntilLookingAtOneOf(const char *str);

    BufferCursor *skipWhileLookingAt(const char *str);

    BufferCursor *skipWhileLookingAtOneOf(const char *str);

    bool isLookingAtOneOf(const char *str);

    bool isLookingAtCRNL();

    BufferCursor *stepOverCRNL();

    bool isLookingAtNL();

    BufferCursor *stepOverNL();

    Mark *mark();

    Mark *markUntil(const char *s);

    bool isLookingAtAlpha();

    bool isLookingAtDigit();

    bool isLookingAtHexDigit();

    bool isLookingAtOctalDigit();

    bool isLookingAtAlphaNum();

    bool isLookingAtAlphaNumOr(const char *s);

    BufferCursor *moveTo(Mark *mark);

    BufferCursor *stepOver(const char c);

    BufferCursor *stepOver(const char *s);

    BufferCursor *skipTill(const char *str);

    BufferCursor *reset();

    BufferCursor(PureRange *pureRange);
    BufferCursor(char *ptr, size_t len);


    BufferCursor(char *ptr);

    virtual ~BufferCursor();

    void show(std::ostream &o);
};


std::ostream &operator<<(std::ostream &o, BufferCursor &c);

std::ostream &operator<<(std::ostream &o, BufferCursor *c);
