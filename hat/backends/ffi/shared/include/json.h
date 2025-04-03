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

#include <map>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include "buffer_cursor.h"
#include "fsutil.h"

//https://www.json.org/json-en.html
class JSonNode;

class JSonObjectNode;

class JSonValueNode;

class JSonListNode;

class JSonWriter {
   std::ostream &o;
   int indent;
   public:
   using Filter = std::function<bool(JSonNode *n)>;

   explicit JSonWriter(std::ostream &o);
     JSonWriter *put(std::string s);
     JSonWriter *name(std::string n);
     JSonWriter *comma();
     JSonWriter *colon();
     JSonWriter *oquote();
     JSonWriter *cquote();
     JSonWriter *obrace();
     JSonWriter *cbrace();
     JSonWriter *osbrace();
     JSonWriter *csbrace();
     JSonWriter *in();
     JSonWriter *out();
     JSonWriter *nl();
     JSonWriter *write(JSonNode *, Filter filter);
     JSonWriter *write(JSonNode *);
};

class JSonNode {
public:
    using JSonNodeVisitor = std::function<void(JSonNode *n)>;
    enum Type {
        LIST, VALUE, OBJECT
    };
    enum ValueType {
        STRING, INTEGER, NUMBER, BOOLEAN
    };
    Type type;
    JSonObjectNode *parent;
    std::string name;

    JSonNode(Type type, JSonObjectNode *parent, std::string name);

    virtual bool hasNode(std::string name);

   virtual JSonNode *getNode(std::string name);

    JSonValueNode *asValue();

    JSonListNode *asList();

    JSonObjectNode *asObject();

   bool isList();
   bool isObject();
   bool isValue();

    static JSonNode *parse(char *text);

    virtual JSonNode *clone(JSonObjectNode *newParent) = 0;

    virtual ~JSonNode() = 0;

    JSonNode *get(std::string s, JSonNodeVisitor visitor);
   JSonNode *collect(std::string s, std::vector<JSonNode *> &list);
    static std::string parseString(BufferCursor *c);

    JSonObjectNode * remove();

    bool write(std::ostream o);
    bool write(std::string filename);
};


class JSonObjectNode : public JSonNode {
public:
   friend JSonNode *JSonNode::get(std::string s, JSonNodeVisitor visitor);
   friend JSonNode *JSonNode::collect(std::string s, std::vector<JSonNode *> &list);
   friend JSonWriter *JSonWriter::write(JSonNode *node, JSonWriter::Filter filter);
    using JSonObjectNodeVisitor = std::function<void(JSonObjectNode *n)>;
    using JSonListNodeVisitor = std::function<void(JSonListNode *n)>;
   protected:
    std::map<std::string, JSonNode *> nameToChildMap;
    std::vector<JSonNode *> childArray;
   public:

    JSonObjectNode * remove(JSonNode *n);
   JSonObjectNode * remove(std::string name);
    JSonObjectNode(Type type, JSonObjectNode *parent, std::string name);

    JSonObjectNode(JSonObjectNode *parent, std::string name);

    ~JSonObjectNode() override;

    virtual JSonNode *parse(BufferCursor *cursor);
    void visit(JSonNodeVisitor visitor);
    JSonObjectNode *object(std::string name, JSonObjectNodeVisitor visitor);

    JSonObjectNode *list(std::string name, JSonListNodeVisitor visitor);

    JSonObjectNode *boolean(std::string name, std::string value);

    JSonObjectNode *boolean(std::string name, bool value);

    JSonObjectNode *number(std::string name, std::string value);

    JSonObjectNode *integer(std::string name, std::string value);
    JSonObjectNode *integer(std::string name, int value);
    JSonObjectNode *string(std::string name, std::string value);
    JSonNode * add( JSonNode *newOne);
    virtual bool hasNode(std::string name) override;

    virtual JSonNode *getNode(std::string name) override;

    virtual JSonNode *clone(JSonObjectNode *newParent) override{
       JSonObjectNode *copy = new JSonObjectNode(newParent, name);

       for (auto c:childArray){
          copy->childArray.push_back(copy->nameToChildMap[c->name] = c->clone(copy));
       }
       return copy;
    }
};


class JSonValueNode : public JSonNode {
public:
    std::string value;
    ValueType valueType;
    JSonValueNode(JSonObjectNode *parent, std::string name, ValueType valueType, std::string value);
    JSonNode *clone(JSonObjectNode *newParent) override{
      return new JSonValueNode(newParent, name, valueType, value);
   }
    ~JSonValueNode() override;
};

class JSonListNode : public JSonObjectNode {
public:


    JSonListNode(JSonObjectNode *parent, std::string name);

    JSonNode *parse(BufferCursor *cursor) override;

    ~JSonListNode() override;

    JSonObjectNode *item(JSonObjectNodeVisitor visitor);

    int size();

public:
   JSonNode *clone(JSonObjectNode *newParent) override{
      JSonListNode *copy = new JSonListNode(newParent, name);

      for (auto c:childArray){
         copy->childArray.push_back(copy->nameToChildMap[c->name] = c->clone(copy));
      }
      return copy;
   }
    JSonObjectNode *boolean(std::string value);

    JSonObjectNode *boolean(bool value);

    JSonObjectNode *string(std::string value);

    JSonObjectNode *integer(std::string value);

    JSonObjectNode *integer(int value);

    JSonObjectNode *number(std::string value);

    JSonObjectNode *list(JSonListNodeVisitor visitor);
};


class JSon{
   public:
   static JSonObjectNode *create(std::function<void(JSonObjectNode *)> builder);
   static JSonNode *parseFile(std::string filename);
};
