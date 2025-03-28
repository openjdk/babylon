
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

#include <sstream>
#include <iomanip>


#include "hex.h"


void Hex::ascii(std::ostream &s, char c) {
    if (::iscntrl(c)) {
        if (c == '\a') {
            s << "\\a ";
        } else if (c == '\r') {
            s << "\\r ";
        } else if (c == '\n') {
            s << "\\n ";
        } else if (c == '\t') {
            s << "\\t ";
        } else {
            s << "?? ";
        }
    } else {
        s << c << "  ";
    }
}

void Hex::hex(std::ostream &s, char c) {
    s << std::hex << std::setw(2) << std::setfill('0') << std::uppercase << (c & 0xff) << " ";
}

void Hex::bytes(std::ostream &s, char *p, size_t len, std::function<void(std::ostream &)> prefix) {
    for (int i = 0; i < len; i++) {
        if ((i % 16) == 0) {
            if (i > 0) {
                s << "  ";
                for (int c = i - 16; c < i; c++) {
                    ascii(s, p[c]);
                }
            }
            s << std::endl;
            prefix(s);
            s << std::hex << std::setw(6) << std::setfill('0') << i << " ";
        }
        hex(s, p[i]);
    }

    if ((len % 16) == 0) {
        s << "  ";
        for (int c = len - 16; c < len; c++) {
            ascii(s, p[c]);
        }
    } else {
        for (int v = len % 16; v < 16; v++) {
            s << "   ";
        }
        s << "  ";
        for (int c = len - (len % 16); c < len; c++) {
            ascii(s, p[c]);
        }

    }


}




