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

void indent(std::ostream &out, int depth, char ch) {
    while (depth-- > 0) {
        out << ch;
    }
}

void Schema::show(std::ostream &out, int depth, Node *node) {
    indent(out, depth, ' ');
    if (auto *schemaNode = dynamic_cast<SchemaNode *>(node)) {
        std::cout << schemaNode->type;
    } else if (auto *arg = dynamic_cast<ArgNode *>(node)) {
        std::cout << arg->idx;
    } else if (auto *structNode = dynamic_cast<StructNode *>(node)) {
        std::cout  <<  ((structNode->name== nullptr)?"?":structNode->name);
    } else if (auto *unionNode = dynamic_cast<UnionNode *>(node)) {
        std::cout <<  ((unionNode->name== nullptr)?"?":unionNode->name);
      } else if (auto *array = dynamic_cast<Array *>(node)) {
        if(array->flexible) {
            std::cout << "[*]";
        }else{
            std::cout << "[" << array->elementCount << "]";
        }
    } else if (auto *fieldNode = dynamic_cast<FieldNode *>(node)) {
        std::cout  << ((fieldNode->name== nullptr)?"?":fieldNode->name)<<":"<<fieldNode->typeName;
    } else {
        std::cout << "<node?>";
    }
    if (node->children.empty()) {
        std::cout << std::endl;
    }else{
        std::cout << "{" << std::endl;
        for (Node *n: node->children) {
            show(out, depth + 1, n);
        }
        indent(out, depth, ' ');
        std::cout << "}" << std::endl;
    }
}

void Schema::show(std::ostream &out, SchemaNode *schemaNode) {
    show(out, 0, schemaNode);
}

Schema::FieldNode *Schema::FieldNode::parse(Cursor *cursor) {
    typeName = cursor->getIdentifier();
    return this;
}

Schema::Array *Schema::Array::parse(Cursor *cursor) {
    cursor->in("Array::Parse");
    if (cursor->is('*')) {
        flexible = true;
    } else if (cursor->peekDigit()) {
        elementCount = cursor->getLong();
    }
    cursor->expect(':', "after element count in array", __LINE__);
    char *identifier= nullptr;
    if (cursor->is('?')) {
    } else if (cursor->peekAlpha()) {
        identifier = cursor->getIdentifier();
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting '?' or identifier for element");
    }
    cursor->expect(':', "after name in array", __LINE__);
    if (cursor->peekAlpha()) {
        elementType = addChild(cursor, new  FieldNode(this, identifier));
    } else if (cursor->is('{')) {
        elementType = addChild(cursor, new  StructNode(this,  identifier));
    } else if (cursor->is('<')) {
        elementType = addChild(cursor, new  UnionNode(this, identifier));
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting type  for element");
    }
    cursor->expect(']', "after array type", __LINE__);
    cursor->out();
    return this;
}

Schema::AbstractStructOrUnionNode *Schema::AbstractStructOrUnionNode::parse(Cursor *cursor) {
    cursor->in("StructUnion::parse");
    do {
        char *identifier = nullptr;
        if (cursor->is('?')) {
            // no name name = null!
        } else if (cursor->peekAlpha()) {
            identifier = cursor->getIdentifier();
        }
        cursor->expect(':', "after StrutOrUnion name", __LINE__);
        if (cursor->peekAlpha()) {
            typeNode = addChild(cursor, new  FieldNode(this, identifier));
        } else if (cursor->is('[')) {
            typeNode = addChild(cursor, new Array(this));
        } else if (cursor->is('{')) {
            typeNode = addChild(cursor, new StructNode(this, identifier ));
        } else if (cursor->is('<')) {
            typeNode = addChild(cursor, new UnionNode(this, identifier));
        } else {
            cursor->error(std::cerr, __FILE__, __LINE__, "expecting type");
        }
    } while (cursor->is(separator));
    cursor->expect(terminator, "at end of struct or union ", __LINE__);
    cursor->out();
    return this;
}

Schema::StructNode *Schema::StructNode::parse(Cursor *cursor) {
    return dynamic_cast<StructNode *>(AbstractStructOrUnionNode::parse(cursor));
}

Schema::UnionNode *Schema::UnionNode::parse(Cursor *cursor) {
    return dynamic_cast<UnionNode *>(AbstractStructOrUnionNode::parse(cursor));
}

Schema::ArgNode *Schema::ArgNode::parse(Cursor *cursor) {
    cursor->in("ArgNode::parse");
    char actual;
    cursor->expectEither('!', '?', &actual, __LINE__);

    cursor->expect(':', __LINE__);
    cursor->expectDigit("long byteCount of buffer", __LINE__);
    long bytes = cursor->getLong();
    if (cursor->isEither('#', '+', &actual)) {
        bool complete = (actual == '#');
        cursor->expectAlpha("identifier", __LINE__);
        char *identifier = cursor->getIdentifier();
        cursor->expect(':', "after identifier ", __LINE__);
        cursor->expect('{', "top level arg struct", __LINE__);
        addChild(cursor, new ArgStructNode(this, complete, identifier));
    } else if (cursor->peekAlpha()) {
        addChild(cursor, new FieldNode(this, cursor->getIdentifier()));
    } else {
        cursor->error(std::cerr, __FILE__, __LINE__, "expecting '#' ");
    }
    cursor->expect(')', "at end of NamedStructOrUnion", __LINE__);
    cursor->out();

    return this;
}

Schema::SchemaNode *Schema::SchemaNode::parse(Cursor *cursor) {
    cursor->in("SchemaNode::parse");
    cursor->expectDigit("arg count", __LINE__);
    int argc = cursor->getInt();
    for (int i = 0; i < argc; i++) {
        cursor->expect('(', __LINE__);
        addChild(cursor, new ArgNode(this, i));
        if (i < (argc - 1)) {
            cursor->expect(',', __LINE__);
        }
    }
    cursor->out();
    return this;
}


