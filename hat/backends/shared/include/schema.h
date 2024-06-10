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

struct Cursor {
    char *ptr;
    Cursor(char *ptr): ptr(ptr) {
    }

    virtual ~Cursor() {
    }

private:
    Cursor *skipWhiteSpace() {
        while (*ptr == ' ' || *ptr == '\n' || *ptr == '\t') {
            step(1);
        }
        return this;
    }


    Cursor *skipIdentifier() {
        while (peekAlpha() || peekDigit()) {
            step(1);
        }
        return this;
    }
public:
    void step(int count) {
        while (count--) {
            ptr++;
        }
    }

    bool peekAlpha() {
        skipWhiteSpace();
        return (::isalpha(*ptr));
    }

    bool peekDigit() {
        skipWhiteSpace();
        return (::isdigit(*ptr));
    }

    bool is(char ch) {
        skipWhiteSpace();
        if (*ptr == ch) {
            step(1);
            return true;
        }
        return false;
    }

    bool isEither(char ch1, char ch2) {
        skipWhiteSpace();
        if (*ptr == ch1 || *ptr == ch2) {
            step(1);
            return true;
        }
        return false;
    }

    bool isEither(char ch1, char ch2, char ch3) {
        skipWhiteSpace();
        if (*ptr == ch1 || *ptr == ch2 || *ptr == ch3) {
            step(1);
            return true;
        }
        return false;
    }

    /*bool is(char *str) {
        skipWhiteSpace();
        int count = 0;
        char *safePtr = ptr;
        while (*str && *ptr && *str == *ptr) {
            ptr++;
            str++;
            count++;
        }
        if (count > 0 && *str == '\0') {
            step(count);
            return true;
        }
        ptr = safePtr;
        return false;
    }*/

    int getInt() {
        int value = *ptr - '0';
        step(1);
        if (peekDigit()) {
            return value * 10 + getInt();
        }
        return value;
    }

    long getLong() {
        long value = *ptr - '0';
        step(1);
        if (peekDigit()) {
            return value * 10 + getLong();
        }
        return value;
    }

    char *getIdentifier() {
        char *identifierStart = ptr;
        skipIdentifier();
        size_t len = ptr - identifierStart;
        char *identifier = new char[len + 1];
        std::memcpy(identifier, identifierStart, len);
        identifier[len] = '\0';
        return identifier;
    }

    void error(std::ostream &ostream, const char *file, int line, const char *str) {
        ostream << file << ":" << "@" << line << ": parse error " << str << " looking at " << ptr << std::endl;
        exit(1);
    }

};

struct Schema {
    struct Node {
        enum Type {
            StructType, UnionType, ArrayType, NamedStructOrUnionType, ArgType, SchemaType, FieldType, SimpleType
        };
        Type type;
        Node *parent;
        char *start;
        char *end;
        std::vector<Node *> children;

        Node(Node *parent, Type type):parent(parent), type(type) {
        }

        Node *addChild(Cursor *cursor, Node *child) {
            children.push_back(child);
            return child->parse(cursor);
        }

        virtual Node *parse(Cursor *cursor) {
            cursor->error(std::cerr, __FILE__, __LINE__, "In Node virtual parse!");
            return nullptr;
        };

        virtual ~Node() = default;
    };

    struct SimpleType : public Node {
        char *name;

        SimpleType(Node *paren)
                : Node(paren, Node::Type::SimpleType), name(nullptr) {

        }

        virtual SimpleType *parse(Cursor *cursor) {
            name = cursor->getIdentifier();
            return this;
        }

        virtual ~SimpleType() {
            if (name) {
                delete[] name;
            }
        }
    };

    struct Array : public Node {
        bool flexible;
        long elementCount;
        char *elementName;
        Node *elementType;

        Array(Node *paren): Node(paren, Node::Type::ArrayType), flexible(false), elementCount(0), elementName(nullptr), elementType(nullptr) {
        }

        virtual Array *parse(Cursor *cursor);

        virtual ~Array() {
            if (elementType) {
                delete elementType;
            }
            if (elementName) {
                delete[] elementName;
            }
        }
    };

    struct NamedNode : public Node {
        char *name;
        Node *typeNode;

        NamedNode(Node *parent, Node::Type type): Node(parent, type), name(nullptr), typeNode(nullptr) {
        }

        ~NamedNode() {
            if (name) {
                delete[] name;
            }

            if (typeNode) {
                delete typeNode;
            }
        }
    };

    struct Field : public NamedNode {
        explicit Field(Node *parent)
                : NamedNode(parent, Type::FieldType) {
        }

        virtual Field *parse(Cursor *cursor) {
            cursor->error(std::cerr, __FILE__, __LINE__, "Implement field parser");
            return this;
        }

        ~Field() {
        }
    };

    struct StructOrUnion : public NamedNode {
        char separator;
        char terminator;

        StructOrUnion(Node *parent, Node::Type type, char separator, char terminator)
                : NamedNode(parent, type), separator(separator), terminator(terminator) {

        }

        virtual StructOrUnion *parse(Cursor *cursor);

        virtual ~StructOrUnion() {

        }
    };

    struct Union : public StructOrUnion {
        Union(Node *parent)
                : StructOrUnion(parent, Node::Type::UnionType, '|', '>') {
        }

        virtual Union *parse(Cursor *cursor) {
            return dynamic_cast<Union *>(StructOrUnion::parse(cursor));
        }

        virtual ~Union() {
        }
    };

    struct Struct : public StructOrUnion {

        explicit Struct(Node *parent)
                : StructOrUnion(parent, Node::Type::StructType, ',', '}') {
        }

        virtual Struct *parse(Cursor *cursor) {
            return dynamic_cast<Struct *>(StructOrUnion::parse(cursor));
        }

        virtual ~Struct() {

        }
    };


    struct NamedStructOrUnion : public Node {
        bool complete;
        long bytes;
        char *identifier;

        NamedStructOrUnion(Node *paren)
                : Node(paren, Node::Type::NamedStructOrUnionType), complete(false), bytes(0L), identifier(nullptr) {
        }

        virtual NamedStructOrUnion *parse(Cursor *cursor);

        virtual ~NamedStructOrUnion() {
            if (identifier) {
                delete[]identifier;
            }

        }
    };


    struct Arg : public Node {
        Arg(Node *parent)
                : Node(parent, Node::Type::ArgType) {
        }

        virtual Arg *parse(Cursor *cursor);

        virtual ~Arg() {

        }
    };

    struct SchemaNode : public Node {
        SchemaNode()
                : Node(nullptr, Node::Type::SchemaType) {
        }

        virtual SchemaNode *parse(Cursor *cursor);

        virtual ~SchemaNode() {
        }
    };

    static void show(std::ostream &out, char *schema);
    static void show(std::ostream &out, int depth, Node* node);
    static void show(std::ostream &out, SchemaNode* schemaNode);
};