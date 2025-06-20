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
#include <iostream>
#include <iomanip>
#include "schema_cursor.h"


struct Schema {
    struct Node {
        Node *parent;
        const char *type;

        std::vector<Node *> children;

        Node(Node *parent, const char *type): parent(parent), type(type) {
        }

        Node *addChild(SchemaCursor *cursor, Node *child) {
            children.push_back(child);
            return child->parse(cursor);
        }

        virtual Node *parse(SchemaCursor *cursor) {
            cursor->error(std::cerr, __FILE__, __LINE__, "In Node virtual parse!");
            return nullptr;
        };

        virtual ~Node() = default;
    };


    struct Array final : Node {
        bool flexible;
        long elementCount;
        char *elementName;
        Node *elementType;

        explicit Array(Node *paren): Node(paren, "Array"), flexible(false), elementCount(0), elementName(nullptr),
                                     elementType(nullptr) {
        }

        Array *parse(SchemaCursor *cursor) override;

        ~Array() override {
            delete elementType;
            delete[] elementName;
        }
    };

    struct AbstractNamedNode : Node {
        char *name;
        Node *typeNode;

        AbstractNamedNode(Node *parent, const char *type, char *name): Node(parent, type), name(name),
                                                                       typeNode(nullptr) {
        }

        ~AbstractNamedNode() override {
            if (name) {
                delete[] name;
            }

            if (typeNode) {
                delete typeNode;
            }
        }
    };

    struct FieldNode : AbstractNamedNode {
        char *typeName;

        FieldNode(Node *paren, char *name)
            : AbstractNamedNode(paren, "FieldNode", name), typeName(nullptr) {
        }

        FieldNode *parse(SchemaCursor *cursor) override;

        ~FieldNode() override {
            if (typeName) {
                delete[] name;
            }
        }
    };

    struct AbstractStructOrUnionNode : AbstractNamedNode {
        char separator;
        char terminator;


        AbstractStructOrUnionNode(Node *parent, const char *type, char separator, char terminator, char *name)
            : AbstractNamedNode(parent, type, name), separator(separator), terminator(terminator) {
        }

        AbstractStructOrUnionNode *parse(SchemaCursor *cursor) override;

        ~AbstractStructOrUnionNode() override = default;
    };

    struct UnionNode final : AbstractStructOrUnionNode {
        UnionNode(Node *parent, char *name)
            : AbstractStructOrUnionNode(parent, "UnionNode", '|', '>', name) {
        }

        UnionNode *parse(SchemaCursor *cursor) override;

        ~UnionNode() override = default;
    };

    struct StructNode : AbstractStructOrUnionNode {
        StructNode(Node *parent, const char *type, char *name)
            : AbstractStructOrUnionNode(parent, type, ',', '}', name) {
        }

        StructNode(Node *parent, char *name)
            : StructNode(parent, "StructNode", name) {
        }

        StructNode *parse(SchemaCursor *cursor) override;

        ~StructNode() override = default;
    };

    struct ArgStructNode final : StructNode {
        bool complete;

        explicit ArgStructNode(Node *parent, bool complete, char *name)
            : StructNode(parent, "ArgStructNode", name), complete(complete) {
        }

        ~ArgStructNode() override = default;
    };


    struct ArgNode : Node {
        int idx;

        ArgNode(Node *parent, int idx)
            : Node(parent, "ArgNode"), idx(idx) {
        }

        ArgNode *parse(SchemaCursor *cursor) override;

        ~ArgNode() override = default;
    };

    struct SchemaNode : Node {
        SchemaNode()
            : Node(nullptr, "Schema") {
        }

        SchemaNode *parse(SchemaCursor *cursor) override;

        ~SchemaNode() override = default;
    };

    static void show(std::ostream &out, char *schema);

    static void show(std::ostream &out, int depth, Node *node);

    static void show(std::ostream &out, SchemaNode *schemaNode);
};
