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
#include <vector>

#include <string.h>
#include <map>
#include <iostream>
#include <regex>
#include "json.h"
#include "buffer_cursor.h"


JSonNode::JSonNode(Type type, JSonObjectNode *parent, std::string name)
      : type(type), parent(parent), name(name) {

}

bool JSonNode::hasNode(std::string name) { return false; }

JSonNode *JSonNode::getNode(std::string name) { return nullptr; }




bool JSonNode::isList(){return  type==LIST;}
bool JSonNode::isObject(){return  type==OBJECT;}
bool JSonNode::isValue(){return  type==VALUE;}

JSonNode::~JSonNode() {}
JSonValueNode::JSonValueNode(JSonObjectNode *parent, std::string name, JSonValueNode::ValueType valueType, std::string value)
      : JSonNode(
      JSonNode::Type::VALUE, parent, name), valueType(valueType), value(value) {}

JSonValueNode::~JSonValueNode() {}

JSonObjectNode::JSonObjectNode(JSonNode::Type type, JSonObjectNode *parent, std::string name)
      : JSonNode(type, parent,
                 name) {}

JSonObjectNode::JSonObjectNode(JSonObjectNode *parent, std::string name)
      : JSonNode(JSonNode::Type::OBJECT, parent,
                 name) {}

JSonObjectNode::~JSonObjectNode() {
   for (auto c:childArray) {
      delete c;
   }
   childArray.clear();
   nameToChildMap.clear();
}

JSonNode *JSonObjectNode::add(JSonNode *newOne) {
   nameToChildMap[newOne->name] = newOne;
   childArray.push_back(newOne);
   return newOne;
}

int JSonListNode::size() {
   return nameToChildMap.size();
}

JSonListNode::JSonListNode(JSonObjectNode *parent, std::string name)
      : JSonObjectNode(JSonNode::Type::LIST, parent,
                       name) {}

JSonListNode::~JSonListNode() {
   // JsonObjectNode parent should deal with this
}


JSonValueNode *JSonNode::asValue() {
   return dynamic_cast<JSonValueNode *>(this);
}

JSonListNode *JSonNode::asList() {
   return dynamic_cast<JSonListNode *>(this);
}

JSonObjectNode *JSonNode::asObject() {
   return dynamic_cast<JSonObjectNode *>(this);
}

void JSonObjectNode::visit(JSonNodeVisitor visitor) {
   for (auto n:childArray) {
      visitor(n);
   }
}

bool JSonObjectNode::hasNode(std::string name) { return nameToChildMap.find(name) != nameToChildMap.end(); }

JSonNode *JSonObjectNode::getNode(std::string name) { return nameToChildMap[name]; }

std::string JSonNode::parseString(BufferCursor *cursor) {
   cursor->advance(); // step over "
   std::string content;
   //https://www.json.org/json-en.html
   while (!cursor->isLookingAt('"')) {
      if (cursor->isLookingAt('\\')){
         cursor->advance();
         char c = cursor->ch();
         switch (c){
            case 'n':content+='\n';break ;
            case 'r':content+='\r';break ;
            case 't':content+='\t';break ;
            case 'b':content+='\b';break ;
            case 'f':content+='\f';break ;
            case '"':content+='"';break ;
            case '/':content+='/';break ;
            case '\\':content+='\\';break ;
            case 'u':{
               cursor->advance();
               int value = 0;
               while (cursor->isLookingAtHexDigit()){
                  c = cursor->ch();
                  value =value *16 + (::isdigit(c) )?c-'0':(c>='a'&&c<='f')?c-'a'+10:c-'A'+10;
               }
               if (value < 127){
                  content+=(char)value;
               }else{
                  std::cerr << "skipping unicode "<< std::hex<< value << std::endl;std::exit(1);
               }
               break;
            }
            default:

            std::cerr << "skipping escape of char '"<< c << std::endl;
            content+=c;
         };
      }else{
         content+=cursor->ch();
      }
      cursor->advance();
   }

   cursor->advance(); // step over "
   return content;
}

JSonNode *JSonObjectNode::parse(BufferCursor *cursor) {
   // we are 'passed the open curly'
   Mark *objectStart = cursor->mark();
   cursor->skipWhiteSpaceOrNewLine();
   while (!cursor->isLookingAt("}")) {
      if (cursor->isLookingAt('\"')) {
         std::string parsedName = JSonNode::parseString(cursor);
         cursor->skipWhiteSpaceOrNewLine();
         if (cursor->isLookingAt(':')) {
            cursor->advance();
            cursor->skipWhiteSpaceOrNewLine();

            if (cursor->isLookingAt("{")) {
               cursor->advance();
               object(parsedName, [&](auto o) {
                  o->parse(cursor);
               });
            } else if (cursor->isLookingAt("[")) {
               cursor->advance();
               // std::cerr << "into Arr"<<std::endl;
               list(parsedName, [&](auto l) {
                  l->parse(cursor);
               });
               //  std::cerr << "outof Arr"<<std::endl;
            } else {
               if (cursor->isLookingAt('\"')) {
                  std::string parsedValue = JSonNode::parseString(cursor);
                  this->string(parsedName, parsedValue);
               } else if (cursor->isLookingAt("true")) {
                  cursor->stepOver("true");
                  this->boolean(parsedName, "true");
               } else if (cursor->isLookingAt("false")) {
                  cursor->stepOver("false");
                  this->boolean(parsedName, "false");
               } else {
                  Mark *start = cursor->mark();
                  bool number = true;
                  bool integer = true;
                  bool first = true;
                  bool hasDot = false;
                  while (cursor->isLookingAtAlphaNumOr("_-.")) {
                     number = number && ((first && cursor->isLookingAt('-')) || cursor->isLookingAtDigit() || (!hasDot && cursor->isLookingAt('.')));
                     integer = integer && ((first && cursor->isLookingAt('-')) || (cursor->isLookingAtDigit()));
                     hasDot = hasDot | cursor->isLookingAt('.');
                     cursor->advance();
                     first = false;

                  }
                  std::string parsedValue = start->str();
                  if (parsedValue == "x") {
                     std::cerr << "x!" << std::endl;
                  }
                  if (integer) {
                     this->integer(parsedName, parsedValue);
                  } else if (number) {
                     this->number(parsedName, parsedValue);
                  } else {
                     this->string(parsedName, parsedValue);
                  }
               }

            }
            cursor->skipWhiteSpaceOrNewLine();
            if (cursor->isLookingAt(",")) {
               cursor->advance();
               cursor->skipWhiteSpaceOrNewLine();
            } else if (!cursor->isLookingAt('}')) {
               std::cerr << "expecting , for }" << std::endl;
            }
         } else {
            std::cerr << "expecting colon name!" << std::endl;
         }

      } else {
         std::cerr << "expecting literal name!" << std::endl;
      }
   }
   cursor->advance();
   cursor->skipWhiteSpaceOrNewLine();
   return this;
}

JSonNode *JSonListNode::parse(BufferCursor *cursor) {
   // we are passed the open '['
   cursor->skipWhiteSpaceOrNewLine();

   Mark *listStart = cursor->mark();
   cursor->skipWhiteSpaceOrNewLine();
   while (!cursor->isLookingAt("]")) {
      if (cursor->isLookingAt("{")) {
         cursor->advance();
         item([&](auto n) {
            n->parse(cursor);
         });
      } else if (cursor->isLookingAt("[")) {
         cursor->advance();
         list([&](auto l) {
            l->parse(cursor);
         });
      } else {
         if (cursor->isLookingAt('\"')) {
            std::string parsedValue = JSonNode::parseString(cursor);
            this->string(parsedValue);
         } else if (cursor->isLookingAt("true")) {
            cursor->stepOver("true");
            this->boolean("true");
         } else if (cursor->isLookingAt("false")) {
            cursor->stepOver("false");
            this->boolean("false");
         } else {
            Mark *start = cursor->mark();
            bool number = true;
            bool integer = true;
            bool first = true;
            bool hasDot = false;
            while (cursor->isLookingAtAlphaNumOr("_-.")) {
               number = number && ((first && cursor->isLookingAt('-')) || cursor->isLookingAtDigit() || (!hasDot && cursor->isLookingAt('.')));
               integer = integer && ((first && cursor->isLookingAt('-')) || (cursor->isLookingAtDigit()));
               hasDot = hasDot | cursor->isLookingAt('.');
               cursor->advance();
               first = false;

            }
            std::string parsedValue = start->str();
            if (parsedValue == "x") {
               std::cerr << "x!" << std::endl;
            }
            if (integer) {
               this->integer(parsedValue);
            } else if (number) {
               this->number(parsedValue);
            } else {
               this->string(parsedValue);
            }

         }
      }

      cursor->skipWhiteSpaceOrNewLine();
      if (cursor->isLookingAt(",")) {
         cursor->advance();
         cursor->skipWhiteSpaceOrNewLine();
      } else if (!cursor->isLookingAt(']')) {
         std::cerr << "expecting , for [" << std::endl;
      }
   }
   cursor->advance();
   cursor->skipWhiteSpaceOrNewLine();
   return this;
}

JSonNode *JSonNode::parse(char *text) {
   BufferCursor *cursor = new BufferCursor((char *) text);
   cursor->skipWhiteSpace();
   if (cursor->isLookingAt("{")) {
      cursor->advance();
      JSonObjectNode *doc = new JSonObjectNode(nullptr, "");
      doc->parse(cursor);
      return doc;
   } else if (cursor->isLookingAt("[")) {
      cursor->advance();
      JSonObjectNode *doc = new JSonListNode(nullptr, "");
      doc->parse(cursor);
      return doc;
   }
   delete cursor;
   return nullptr;
}
JSonObjectNode * JSonNode::remove(){
   return parent->remove(this);
}

JSonNode *JSonNode::collect(std::string s, std::vector<JSonNode *> &list) {
   std::cout << "collecting "<< s << std::endl;
   if (s == "") {
      list.push_back(this);
   } else {
      auto slashpos = s.find_first_of('/');
      std::string head = (slashpos == std::string::npos) ? s : s.substr(0, slashpos);
      std::string tail = (slashpos == std::string::npos) ? "" : s.substr(slashpos + 1);
      if (head == "..") {
         parent->collect(tail, list);
      } else {
         if (head[0]=='{'){
            auto eqpos = head.find_first_of('=');
            auto tildepos = head.find_first_of('~');
            auto ccbracepos = head.find_last_of('}');
            auto notpos =  head.find_first_of('!');
            if (eqpos != std::string::npos && (tildepos==std::string::npos || tildepos>eqpos ) && ccbracepos != std::string::npos && ccbracepos>eqpos){
               bool invert = (notpos != std::string::npos && notpos+1 == eqpos);
               std::string listName = head.substr(1, eqpos-1 - (invert?1:0));
               std::string re = head.substr(eqpos+1, ccbracepos-eqpos-1);
               JSonObjectNode *node = asObject();
               if (node) {
                  auto n = node->nameToChildMap[listName];
                  if (n && n->isValue()) {
                     std::string svalue = n->asValue()->value;
                     if (invert && svalue != re) {
                        list.push_back(this);
                     } else if (!invert && svalue == re) {
                        list.push_back(this);
                     }
                  }
               }
            }else  if (tildepos != std::string::npos && (eqpos==std::string::npos || eqpos>tildepos ) && ccbracepos != std::string::npos && ccbracepos>tildepos){
               bool invert = (notpos != std::string::npos && notpos+1 ==tildepos);
               std::string listName = head.substr(1, tildepos-1 - (invert?1:0));
               std::string re = head.substr(tildepos+1, ccbracepos-tildepos-1);
               JSonObjectNode *node = asObject();
               if (node) {
                  auto n = node->nameToChildMap[listName];
                  if (n && n->isValue()){
                     std::string svalue = n->asValue()->value;
                     std::regex r(re);
                     bool matched = std::regex_match(svalue, r);
                     if (invert && !matched){
                        list.push_back(this);
                     }else if (!invert && matched) {
                        list.push_back(this);
                     }
                  }
               }
            }
         }else {
            auto osbracepos = head.find_first_of('[');
            auto csbracepos = head.find_last_of(']');
            if (osbracepos != std::string::npos && csbracepos != std::string::npos && csbracepos>osbracepos) {
               // we have something akin to map[...]
               std::string listName = s.substr(0, osbracepos);
               std::string listSuffix = s.substr(osbracepos + 1, csbracepos - osbracepos - 1);
               JSonObjectNode *node = asObject();
               if (node) {
                  std::regex r(listName);
                  for (auto spair: node->nameToChildMap) {
                     if (std::regex_match(spair.first, r)) {
                        if (tail == ""){
                           spair.second->collect(listSuffix, list);
                        }else {
                           spair.second->collect(listSuffix + "/" + tail, list);
                        }

                     }
                  }
               }
            } else {
               //  auto ocbracepos  = s.find_first_of('{');
               JSonObjectNode *node = asObject();
               if (node) {
                  std::regex r(head);
                  for (auto spair: node->nameToChildMap) {
                     if (std::regex_match(spair.first, r)) {
                        spair.second->collect(tail, list);
                     }
                  }
               } else {
                  list.push_back(this);
               }
            }
         }
      }
   }
   return this;
}


JSonNode *JSonNode::get(std::string s, JSonNodeVisitor visitor) {
   if (s == "") {
      visitor(this);
   } else {
      auto slashpos = s.find_first_of('/');
      std::string head = (slashpos == std::string::npos) ? s : s.substr(0, slashpos);
      std::string tail = (slashpos == std::string::npos) ? "" : s.substr(slashpos + 1);
      if (head == "..") {
         parent->get(tail, visitor);
      } else {
         if (head[0]=='{'){
            auto eqpos = head.find_first_of('=');
            auto tildepos = head.find_first_of('~');
            auto ccbracepos = head.find_last_of('}');
            auto notpos =  head.find_first_of('!');
            if (eqpos != std::string::npos && (tildepos==std::string::npos || tildepos>eqpos ) && ccbracepos != std::string::npos && ccbracepos>eqpos){
               bool invert = (notpos != std::string::npos && notpos+1 == eqpos);
               std::string listName = head.substr(1, eqpos-1 - (invert?1:0));
               std::string re = head.substr(eqpos+1, ccbracepos-eqpos-1);
               JSonObjectNode *node = asObject();
               if (node) {
                  auto n = node->nameToChildMap[listName];
                  if (n && n->isValue()) {
                     std::string svalue = n->asValue()->value;
                     if (invert && svalue != re) {
                        visitor(this);
                     } else if (!invert && svalue == re) {
                        visitor(this);
                     }
                  }
               }
            }else  if (tildepos != std::string::npos && (eqpos==std::string::npos || eqpos>tildepos ) && ccbracepos != std::string::npos && ccbracepos>tildepos){
               bool invert = (notpos != std::string::npos && notpos+1 == tildepos);
               std::string listName = head.substr(1, tildepos-1 - (invert?1:0));
               std::string re = head.substr(tildepos+1, ccbracepos-tildepos-1);
               JSonObjectNode *node = asObject();
               if (node) {
                  auto n = node->nameToChildMap[listName];
                  if (n && n->isValue()){
                     std::string svalue = n->asValue()->value;
                     std::regex r(re);
                     bool matched = std::regex_match(svalue, r);
                     if (invert && !matched){
                        visitor(this);
                     }else if (!invert && matched) {
                        visitor(this);
                     }
                  }
               }
            }
         }else {
            auto osbracepos = head.find_first_of('[');
            auto csbracepos = head.find_last_of(']');
            if (osbracepos != std::string::npos && csbracepos != std::string::npos && csbracepos>osbracepos) {
               // we have something akin to map[...]
               std::string listName = s.substr(0, osbracepos);
               std::string listSuffix = s.substr(osbracepos + 1, csbracepos - osbracepos - 1);
               JSonObjectNode *node = asObject();
               if (node) {
                  std::regex r(listName);
                  for (auto spair: node->nameToChildMap) {
                     if (std::regex_match(spair.first, r)) {
                        if (tail == ""){
                           spair.second->get(listSuffix, visitor);
                        }else {
                           spair.second->get(listSuffix + "/" + tail, visitor);
                        }
                     }
                  }
               }
            } else {
               //  auto ocbracepos  = s.find_first_of('{');
               JSonObjectNode *node = asObject();
               if (node) {
                  std::regex r(head);
                  for (auto spair: node->nameToChildMap) {
                     if (std::regex_match(spair.first, r)) {
                        spair.second->get(tail, visitor);
                     }
                  }
               } else {
                  visitor(this);
               }
            }
         }
      }
   }
   return this;
}
JSonObjectNode* JSonObjectNode::remove(JSonNode *n){
   nameToChildMap.erase(n->name);
   for (auto i=childArray.begin(); i!=childArray.end(); i++){
      if (*i == n){
         childArray.erase(i);
         break;
      }
   }
   return this;
}
JSonObjectNode* JSonObjectNode::remove(std::string name){
   return remove(getNode(name));
}
JSonObjectNode *JSonObjectNode::object(std::string name, JSonObjectNodeVisitor visitor) {
   JSonObjectNode *newOne = new JSonObjectNode(this, name);
   visitor(newOne);
   add(newOne);
   return this;
}

JSonObjectNode *JSonObjectNode::list(std::string name, JSonListNodeVisitor visitor) {

   JSonListNode *newOne = new JSonListNode(this, name);
   visitor(newOne);
   add(newOne);
   return this;
}

JSonObjectNode *JSonListNode::item(JSonObjectNodeVisitor visitor) {
   JSonObjectNode *newOne = new JSonObjectNode(this, std::to_string(childArray.size()));
   visitor(newOne);
   add(newOne);
   return this;
}

JSonObjectNode *JSonObjectNode::boolean(std::string name, std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, name, BOOLEAN, value);
   add(newOne);
   return this;
}

JSonObjectNode *JSonObjectNode::boolean(std::string name, bool value) {
   JSonValueNode *newOne = new JSonValueNode(this, name, BOOLEAN, std::to_string(value));
   add(newOne);
   return this;
}

JSonObjectNode *JSonObjectNode::string(std::string name, std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, name, STRING, value);
   add(newOne);
   return this;
}

JSonObjectNode *JSonObjectNode::integer(std::string name, std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, name, INTEGER, value);
   add(newOne);
   return this;
}

JSonObjectNode *JSonObjectNode::integer(std::string name, int value) {
   JSonValueNode *newOne = new JSonValueNode(this, name, INTEGER, std::to_string(value));
   add(newOne);
   return this;
}

JSonObjectNode *JSonObjectNode::number(std::string name, std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, name, NUMBER, value);
   add(newOne);
   return this;
}

JSonObjectNode *JSonListNode::boolean(std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, std::to_string(childArray.size()), ValueType::BOOLEAN, value);
   add(newOne);
   return this;
}

JSonObjectNode *JSonListNode::boolean(bool value) {
   JSonValueNode *newOne = new JSonValueNode(this, std::to_string(childArray.size()), ValueType::BOOLEAN, std::to_string(value));
   add(newOne);
   return this;
}

JSonObjectNode *JSonListNode::integer(std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, std::to_string(childArray.size()), ValueType::INTEGER, value);
   add(newOne);
   return this;
}

JSonObjectNode *JSonListNode::integer(int value) {
   JSonValueNode *newOne = new JSonValueNode(this, std::to_string(childArray.size()), ValueType::INTEGER, std::to_string(value));
   add(newOne);
   return this;
}

JSonObjectNode *JSonListNode::number(std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, std::to_string(childArray.size()), ValueType::NUMBER, value);
   add(newOne);
   return this;
}
bool JSonNode::write(std::ostream o){
   JSonWriter w(o);
   w.write(this, nullptr);
   return true;
}
bool JSonNode::write(std::string filename){
   std::ofstream all(filename, std::ios::trunc);
   JSonWriter w(all);
   w.write(this, nullptr);
   all.close();
   return true;
}


JSonObjectNode *JSonListNode::string(std::string value) {
   JSonValueNode *newOne = new JSonValueNode(this, std::to_string(childArray.size()), ValueType::STRING, value);
   add(newOne);
   return this;
}

JSonObjectNode *JSonListNode::list(JSonListNodeVisitor visitor) {
   JSonListNode *newOne = new JSonListNode(this, std::to_string(childArray.size()));
   add(newOne);
   visitor(newOne);
   return this;
}


JSonWriter *JSonWriter::write(JSonNode *n){
   return write(n, nullptr);
}
JSonWriter *JSonWriter::write(JSonNode *n, Filter filter){
   if (filter == nullptr || filter(n)) {
      if (n->isObject()) {
         JSonObjectNode *object = n->asObject();
         obrace();
         in();
         nl();
         bool first = true;
         for (auto c: object->childArray) {
            if (filter== nullptr || filter(c)) {
            if (first) {
               first = false;
            } else {
               comma();
               nl();
            }
               name(c->name);
               write(c, filter);
            }
         }
         out();
         nl();
         cbrace();
      } else if (n->isList()) {
         JSonListNode *list = n->asList();
         osbrace();
         in();
         nl();
         bool first = true;
         for (auto c: list->childArray) {
            if (first) {
               first = false;
            } else {
               comma();
               if (!c->isObject()) {
                  nl();
               }
            }
            write(c, filter);
         }
         out();
         nl();
         csbrace();
      } else if (n->isValue()) {
         JSonValueNode *value = n->asValue();
         if (value->valueType == JSonNode::ValueType::STRING) {
            oquote();
         }
         std::size_t n = value->value.length();
         std::string escaped;
         escaped.reserve(n * 2);        // pessimistic preallocation

         for (std::size_t i = 0; i < n; ++i) {
            switch (value->value[i]) {
               case '\n':
                  escaped += "\\n";
                  break;
               case '"':
                  escaped += "\\\"";
                  break;
               case '\\':
                  escaped += "\\\\";
                  break;
               case '\r':
                  escaped += "\\r";
                  break;
               case '\t':
                  escaped += "\\t";
                  break;
               default:
                  escaped += value->value[i];
                  break;
            }
         }
         put(escaped);
         if (value->valueType == JSonNode::ValueType::STRING) {
            cquote();
         }
      } else {
         std::cerr << "what type is this!" << std::endl;
      }
   }
   return this;
}

JSonWriter::JSonWriter(std::ostream &o)
      : o(o) , indent(0){
}

JSonWriter *JSonWriter::put(std::string s) {
   o << s;
   return this;
}

JSonWriter *JSonWriter::comma() {
   return put(",");
}
JSonWriter *JSonWriter::nl(){
   o<<std::endl;
   std::fill_n(std::ostream_iterator<char>(o), indent, ' ');
   return this;
}
JSonWriter *JSonWriter::in(){
   indent++;
   return this;
}
JSonWriter *JSonWriter::out(){
   indent--;
   return this;
}
JSonWriter *JSonWriter::colon() {
   return put(":");
}
JSonWriter *JSonWriter::oquote() {
   return put("\"");
}
JSonWriter *JSonWriter::cquote() {
   return put("\"");
}
JSonWriter *JSonWriter::obrace() {
   return put("{");
}
JSonWriter *JSonWriter::cbrace() {
   return put("}");
}
JSonWriter *JSonWriter::osbrace() {
   return put("[");
}
JSonWriter *JSonWriter::csbrace() {
   return put("]");
}
JSonWriter *JSonWriter::name(std::string n){
   return oquote()->put(n)->cquote()->colon();
}

JSonObjectNode *JSon::create(std::function<void(JSonObjectNode *)> builder){
   JSonObjectNode * root = new JSonObjectNode(nullptr, "");
   builder(root);
   return root;
}


 JSonNode *JSon::parseFile(std::string filename){
   if (fsutil::isFile(filename)) {

      struct stat st;
      stat(filename.c_str(), &st);
      if (S_ISREG(st.st_mode)) {
         int fd = ::open(filename.c_str(), O_RDONLY);
         char *memory= new char[st.st_size];
         size_t bytesRead = 0;
         size_t bytes = 0;
         while (bytesRead < st.st_size && (bytes = ::read(fd, memory + bytesRead, st.st_size - bytesRead)) >= 0) {
            bytesRead -= bytes;
         }
         ::close(fd);
         JSonNode *json = JSonNode::parse(memory);
         delete []memory;
         return json;
      }else{
         std::cout << "not reg file!"<< std::endl;
      }
   }

   return nullptr;
}
