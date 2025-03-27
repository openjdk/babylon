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
 #include <string.h>
#include <iostream>
#include "buffer_cursor.h"


Mark::Mark(BufferCursor *cursor)
        : cursor(cursor), ptr(nullptr), end(nullptr) {}

std::string Mark::str(char *end) {
    cursor->isValid(end);
    return std::string(ptr, end - ptr);
}

size_t Mark::getSize() {
    cursor->isValid((char *) (cursor->get() - ptr));
    return cursor->get() - ptr;
}

std::string Mark::str() {
    return std::string(ptr, getSize());
}

std::string Mark::str(int delta) {
    return std::string(ptr, getSize()+delta);
}


char *Mark::getStart(){
    return ptr;
}
char *Mark::getEnd(){
    return setEnd();
}

char *Mark::setEnd(){
    if (end == nullptr){
        end = cursor->get();
    }
    return end;
}

char *BufferCursor::get() {
    isValid(ptr);
    return ptr;
}

bool BufferCursor::isValid(char *p) {
  // if (p> endPtr){
     // std::cerr << "p beyond end " << endPtr-p << std::endl;
  // }
  // if (p< startPtr){
   //   std::cerr << "p before start " << p-startPtr << std::endl;
  // }
    return p >= startPtr && p <= endPtr;
}

bool BufferCursor::end() {
    // isValid(ptr);
    return ptr >= endPtr;
}

BufferCursor *BufferCursor::advance(int i) {
    if (isValid(ptr)) {
        ptr += i;
     //   if (isValid(ptr)) {
            return this;
        //} else {
           // std::cerr << "ptr after advance is invalid";
          //  return this;
         //   std::exit(1);

    } else {
        std::cerr << "ptr before advance is invalid";
        std::exit(1);
    }

}

BufferCursor *BufferCursor::backup(int i) {
    if (isValid(ptr)) {
        ptr -= i;
        if (isValid(ptr)) {
            return this;
        } else {
            std::cerr << "ptr after backup is invalid";
            std::exit(1);
        }
    } else {
        std::cerr << "ptr before backup is invalid";
        std::exit(1);
    }

}

BufferCursor *BufferCursor::advance() {
    return advance(1);
}

BufferCursor *BufferCursor::backup() {
    return backup(1);
}

char BufferCursor::ch() {
    if (!isValid(ptr)){
       std::cerr << "read past end!" << std::endl;
       std::exit(1);
    }
    return *ptr;
}

int BufferCursor::chAsDigit() {
    isValid(ptr);
    if (!isLookingAtDigit()) {
        std::cerr << "not a digit" << std::endl;
        std::exit(1);
    }
    return ch() - '0';
}

int BufferCursor::chAsHexDigit() {
    isValid(ptr);
    if (!isLookingAtHexDigit()) {
        std::cerr << "not a digit" << std::endl;
        std::exit(1);
    }
    if (isLookingAtDigit()) {
        return chAsDigit();
    } else {
        return tolower(ch()) - 'a' + 10;
    }
}

char BufferCursor::chAndAdvance() {
    char c = ch();
    advance();
    return c;
}

bool BufferCursor::isLookingAt(const char c) {
    return (ch() == c);
}

bool BufferCursor::isLookingAtAlpha() {
    return (::isalpha(ch()));
}

bool BufferCursor::isLookingAtDigit() {
    return (isLookingAtOneOf("0123456789"));
}
bool BufferCursor::isLookingAtOctalDigit() {
    return (isLookingAtOneOf("01234567"));
}
bool BufferCursor::isLookingAtHexDigit() {
    return (isLookingAtOneOf("0123456789abcdefABCDEF"));
}

bool BufferCursor::isLookingAtAlphaNum() {
    return (isLookingAtAlpha() || isLookingAtDigit());
}

bool BufferCursor::isLookingAtAlphaNumOr(const char *s) {
    return (isLookingAtAlphaNum() || isLookingAtOneOf(s));
}

BufferCursor *BufferCursor::skipWhiteSpace() {
    return skipWhileLookingAt(" ");
}

BufferCursor *BufferCursor::skipWhiteSpaceOrNewLine() {
    return skipWhileLookingAtOneOf(" \n\t\r");
}


bool BufferCursor::isLookingAt(const char *str) {
    char *p = ptr;
    isValid(p + strlen(str));
    while ((*p == *str) && (*str != '\0')) {
        p++;
        str++;
    }
    return (!*str);
}

bool BufferCursor::isLookingAtAndStepOver(const char *str) {
    if (isLookingAt(str)) {
        stepOver(str);
        return true;
    }
    return false;
}

BufferCursor *BufferCursor::skipUntilLookingAt(const char *str) {
    while (!isLookingAt(str)) {
        advance();
    }
    return this;
}

BufferCursor *BufferCursor::backupUntilLookingAt(const char *str) {
    while (!isLookingAt(str)) {
        backup();
    }
    return this;
}

BufferCursor *BufferCursor::skipWhileLookingAt(const char *str) {
    while (isLookingAt(str)) {
        advance();
    }
    return this;
}

BufferCursor *BufferCursor::skipWhileLookingAtOneOf(const char *str) {
    while (isLookingAtOneOf(str)) {
        advance();
    }
    return this;
}

BufferCursor *BufferCursor::skipUntilLookingAtOneOf(const char *str) {
    while (!isLookingAtOneOf(str)) {
        advance();
    }
    return this;
}

bool BufferCursor::isLookingAtOneOf(const char *str) {
    while (*str) {
        if (isLookingAt(*str++)) {
            return true;
        }
    }

    return false;
}


BufferCursor *BufferCursor::moveTo(Mark *mark) {
    ptr = mark->ptr;
    isValid(ptr);
    return this;
}

bool BufferCursor::isLookingAtCRNL() {
    return isLookingAt("\r\n");
}

BufferCursor *BufferCursor::stepOverCRNL() {
    return stepOver("\r\n");
}

bool BufferCursor::isLookingAtNL() {
    return isLookingAt("\n");
}

BufferCursor *BufferCursor::stepOverNL() {
    return stepOver("\n");
}

Mark *BufferCursor::mark() {
    Mark *mark = new Mark(this);
    mark->cursor = this;
    mark->ptr = ptr;
    marks.push_back(mark);
    return mark;
}

Mark *BufferCursor::markUntil(const char *s) {
    Mark *newMark = mark();
    skipTill(s);
    newMark->setEnd();
    return newMark;
}

BufferCursor *BufferCursor::stepOver(const char c) {
    if (isLookingAt(c)) {
        advance(1);
    } else {
        std::cerr << " expecting '" << c << "'" << std::endl;
        std::exit(0);
    }
    return this;
}

BufferCursor *BufferCursor::stepOver(const char *s) {
    int len = strlen(s);
    if (isLookingAt(s)) {
        advance(len);
    } else {
        std::cerr << " expecting to step over '" << s << "'" << std::endl;
        std::exit(0);
    }
    return this;
}


BufferCursor *BufferCursor::skipTill(const char *str) {
    while (!end() && *ptr) {
        if (isLookingAt(str)) {
            return this;
        }
        advance();
    }
    return this;
}

BufferCursor *BufferCursor::reset() {
    ptr = startPtr;
    return this;
}

BufferCursor::BufferCursor(PureRange *pureRange)
        : startPtr(pureRange->getStart()), ptr(pureRange->getStart()), endPtr(pureRange->getEnd()) {

}

BufferCursor::BufferCursor(char *ptr, size_t
len)
        : startPtr(ptr), ptr(ptr), endPtr(ptr + len) {

}

BufferCursor::BufferCursor(char *ptr)
        : startPtr(ptr), ptr(ptr), endPtr(ptr + ::strlen(ptr)) {

}
BufferCursor::~BufferCursor(){
   for (auto mark:marks){
      delete mark;
   }
   marks.clear();
}
void BufferCursor::show(std::ostream &o) {
    char *safe = ptr;
    while (!end()) {
        o << *ptr;
        advance();
    }
    ptr = safe;
}

std::ostream &operator<<(std::ostream &o, BufferCursor &c) {
    c.show(o);
    return o;
}

std::ostream &operator<<(std::ostream &o, BufferCursor *c) {
    c->show(o);
    return o;
}


char *BufferCursor::getStart() {
    return startPtr;
}

char *BufferCursor::getEnd() {
    return endPtr;
}

size_t BufferCursor::getSize(){
    return endPtr - startPtr;
}

BufferCursor *BufferCursor::moveToOffset(int offset) {
    if (offset < 0) {
        ptr = endPtr + offset;
    } else {
        ptr = startPtr + offset;
    }
    if (!isValid(ptr)) {
        std::cerr << "ptr after moveOffset is invalid";
        std::exit(1);
    }
    return this;
}
