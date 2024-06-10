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
#include "schema.h"

void Schema::show(std::ostream &out, char *argArray) {

}
void Schema::show(std::ostream &out, int depth, Node *schemaNode) {

}
void Schema::show(std::ostream &out, SchemaNode *schemaNode) {

}


Schema::Array *Schema::Array::parse(Cursor *cursor) {
    if (cursor->is('*')) {
        flexible = true;
    } else if (cursor->peekDigit()) {
        elementCount = cursor->getLong();
    }
    if (cursor->is(':')) {
        if (cursor->is('?')) {
        } else if (cursor->peekAlpha()) {
            elementName = cursor->getIdentifier();
        } else {
            cursor->error(std::cerr, __FILE__, __LINE__, "expecting '?' or identifier for element");
        }
        if (cursor->is(':')) {
            if (cursor->peekAlpha()) {
                elementType = addChild(cursor,new struct SimpleType(this));
            } else if (cursor->is('{')){
                elementType = addChild(cursor, new struct Struct(this));
            } else if (cursor->is('<')){
                elementType = addChild(cursor, new struct Union(this));
            }else{
                cursor->error(std::cerr, __FILE__, __LINE__, "expecting '?' or identifier for element");
            }
            if (cursor->is(']')) {
                return this;
            } else {
                cursor->error(std::cerr, __FILE__, __LINE__, "expecting ']'");
            }

        } else {
            cursor->error(std::cerr, __FILE__, __LINE__, "expecting ':'");
        }
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting ':'");
    }
    cursor->error(std::cerr, __FILE__, __LINE__, "Implement array parser");
    return this;
}

Schema::StructOrUnion *Schema::StructOrUnion::parse(Cursor *cursor) {
    do {
        if (cursor->is('?')) {

        }else if (cursor->peekAlpha()) {
            name = cursor->getIdentifier();
        }
        if (cursor->is(':')) {
            if (cursor->peekAlpha()) {
                typeNode = addChild(cursor, new struct SimpleType(this));
            } else if (cursor->is('[')) {
                typeNode = addChild(cursor, new Array( this));
            } else if (cursor->is('{')) {
                typeNode = addChild(cursor, new Struct( this));
            } else if (cursor->is('<')) {
                typeNode = addChild(cursor, new Union( this));
            } else {
                cursor->error(std::cerr, __FILE__, __LINE__, "expecting type");
            }
        } else {
            cursor->error(std::cerr, __FILE__, __LINE__, "expecting ':'");
        }
    } while (cursor->is(separator));
    if (cursor->is(terminator)) {
        return this;
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting '}' or '>'");
    }
    return this;
}

Schema::NamedStructOrUnion *Schema::NamedStructOrUnion::parse(Cursor *cursor) {
    if (cursor->peekDigit()) {
        bytes = cursor->getLong();
        if (cursor->isEither('#', '+')) {
            complete = ((*(cursor->ptr - 1)) == '#');
            if (cursor->peekAlpha()) {
                identifier = cursor->getIdentifier();
                if (cursor->is(':')) {
                    if (cursor->is('{')) {
                        addChild(cursor, new Struct( this));
                    } else if (cursor->is('<')) {
                        addChild(cursor, new Union(this));
                    } else {
                        cursor->error(std::cerr, __FILE__, __LINE__, "expecting '{' or '<'");
                    }
                } else {
                    cursor->error(std::cerr, __FILE__, __LINE__, "expecting ':'");
                }
            } else {
                cursor->error(std::cerr, __FILE__, __LINE__, "expecting identifier ");
            }
        } else if (cursor->peekAlpha()) {
            addChild(cursor, new Field( this));
        } else {
            cursor->error(std::cerr, __FILE__, __LINE__, "expecting '#' ");
        }
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting long byteCount of buffer  ");
    }
    if (cursor->is(')')) {
        return this;
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting ')'");
        return nullptr;
    }

}
Schema::Arg *Schema::Arg::parse(Cursor *cursor) {
    if (cursor->isEither('!', '?')) {
        if (cursor->is(':')) {
            addChild(cursor, new NamedStructOrUnion( this));
        } else {
            cursor->error(std::cerr, __FILE__, __LINE__, "expecting ':'");
        }
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting '?' or '!'");
    }
    return this;
}
Schema::SchemaNode *Schema::SchemaNode::parse(Cursor *cursor) {
    if (cursor->peekDigit()) {
        int argc = cursor->getInt();
        for (int i = 0; i < argc; i++) {
            if (cursor->is('(')) {
                addChild(cursor, new Arg( this));
                if (i < (argc - 1)) {
                    if (cursor->is(',')) {
                        //
                    } else {
                        cursor->error(std::cerr, __FILE__, __LINE__, "expecting ','");
                    }
                }
            } else {
                cursor->error(std::cerr, __FILE__, __LINE__, "expecting '('");
            }
        }
    }
    return this;
}
/*
char *Schema::strduprange(char *start, char *end) {
    char *s = new char[end - start + 1];
    std::memcpy(s, start, end - start);
    s[end - start] = '\0';
    return s;
}

std::ostream &Schema::dump(std::ostream &out, char *start, char *end) {
    while (start < end) {
        out << (char) *start;
        start++;
    }
    return out;
}

std::ostream &Schema::indent(std::ostream &out, int depth) {
    while (depth > 0) {
        out << "  ";
        depth++;
    }
    return out;
}

std::ostream &Schema::dump(std::ostream &out, char *label, char *start, char *end) {
    out << label << " '";
    dump(out, start, end) << "' " << std::endl;
    return out;
}
*/


