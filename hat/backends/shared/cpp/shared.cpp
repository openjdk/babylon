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
#include "shared.h"


extern void dump(FILE *file, size_t len, void *ptr) {
    for (int i = 0; i < len; i++) {
        if (i % 16 == 0) {
            if (i != 0) {
                fprintf(file, "\n");
            }
            fprintf(file, "%lx :", ((unsigned long) ptr) + i);
        }
        fprintf(file, " %02x", ((int) (((unsigned char *) ptr)[i]) & 0xff));
    }
}


extern "C" void dumpArgArray(void *ptr) {
    ArgSled argSled((ArgArray_t *) ptr);
    std::cout << "ArgArray->argc = " << argSled.argc() << std::endl;
    for (int i = 0; i < argSled.argc(); i++) {
        argSled.dumpArg(i);
    }
    std::cout << "schema = " << argSled.schema() << std::endl;


    //#define YEP
#ifdef YEP
    std::cout << std::endl << "spec->" << spec << std::endl;

    char *data = (char *) ptr;
    char *p = spec;
    std::stack<State *> stack;

    while (*p) {
       if (*p == '[' && ::isdigit(*(p+1))) {
          p++;
          int count = 0;
          while (::isdigit(*p)) {
             count = (count * 10) + (*p - '0');
             p++;
          }
          if (*p == ':') {
             p++;
          }
          stack.push(State::sequence(p, data, count));
          for (int i = 0; i < stack.size(); i++) {
             std::cout << " ";
          }
          std::cout << (stack.top()->count) << " of " << (stack.top()->of) <<std::endl;
       } else if (*p == '[') {
          p++;
          stack.push(State::structOrUnion(p, data));
          for (int i = 0; i < stack.size(); i++) {
             std::cout << " ";
          }
          std::cout << "{" << std::endl;
       } else if (*p == ']' && !stack.empty() && stack.top()->isMidSequence()) {
          stack.top()->count++;
          p = stack.top()->start;
          for (int i = 0; i < stack.size(); i++) {
             std::cout << " ";
          }
          //I V vi iii IV I IV V
          std::cout << (stack.top()->count) << " of " << (stack.top()->of) << std::endl ;
       } else if (*p == ']' && !stack.empty() && stack.top()->isSequence()) {
          p++;
          State *state = stack.top();
          if (*p == '(') {
             p++;
             char *start = p;
             while (*p != ')') {
                state->name[p - start] = *p;
                p++;
             }
             state->name[p - start] = '\0';
          }
          p++;
          stack.pop();
          for (int i = 0; i < stack.size(); i++) {
             std::cout << " ";
          }
          std::cout << "]" << state->name << std::endl;
          delete state;

       } else if (*p == ']' && !stack.empty() && stack.top()->isStructOrUnion()) {
          p++;
          State *state = stack.top();
          if (*p == '(') {
             p++;
             char *start = p;
             while (*p != ')') {
                state->name[p - start] = *p;
                p++;
             }
             state->name[p - start] = '\0';
          }
          p++;
          stack.pop();
          for (int i = 0; i < stack.size(); i++) {
             std::cout << " ";
          }
          std::cout << "}" << state->name<< std::endl;
          delete state;
       } else if ( (*p == '|') && !stack.empty() && stack.top()->isStructOrUnion() ) {
          p++;
          // we refetch data from the dataStart of the enclosing union
          data = (char*)stack.top()->dataStart;
       } else if ( (*p == 'i' || *p == 'b' || *p == 's'|| *p == 'f') && !stack.empty() && stack.top()->isStructOrUnion() ) {
          char *start = p;
          p++;
          int sz = 0;
          while (::isdigit(*p)) {
             sz = sz * 10 + *p - '0';
             p++;
          }
          State *state =  State::member(start,data, sz);
          stack.push(state);
          if (*p == '(') {
             p++;
             char *start = p;
             while (*p != ')') {
                state->name[p - start] = *p;
                p++;
             }
             state->name[p - start] = '\0';
          }
          for (int i = 0; i < stack.size(); i++) {
             std::cout << " ";
          }
          state->value(std::cout, data);
          data += (state->sz / 8);
          stack.pop();
          delete state;
          p++;
       } else if (stack.empty()){
          std::cout << "empty stack and  unhandled -> "<< p<< std::endl;
          p++;
       } else {
          std::cout <<stack.top()->state()<< " unhandled -> "<< p<< std::endl;
          p++;
       }
    }
#endif
}

void hexdump(void *ptr, int buflen) {
    unsigned char *buf = (unsigned char *) ptr;
    int i, j;
    for (i = 0; i < buflen; i += 16) {
        printf("%06x: ", i);
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%02x ", buf[i + j]);
            else
                printf("   ");
        printf(" ");
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%c", isprint(buf[i + j]) ? buf[i + j] : '.');
        printf("\n");
    }
}

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


static const int AWAITING_STATE = 0x00001;
static const int IN_STATE = 0x00002;
static const int HAVE_STATE = 0x00004;

static const int NAME_STATE = 0x00010;
static const int CAN_BE_ANON_STATE = 0x00020;
static const int NUMERIC_STATE = 0x00040;
static const int CAN_BE_FLEX_STATE = 0x00080;

static const int ARGS_STATE = 0x00100;
static const int ARRAY_STATE = 0x00200;
static const int BUFFER_STATE = 0x00400;
static const int FIELD_STATE = 0x01000;
static const int TYPE_STATE = 0x02000;
static const int STRUCT_STATE = 0x04000;
static const int UNION_STATE = 0x08000;

static const int awaitingArgsName = (ARGS_STATE | NAME_STATE | AWAITING_STATE);
static const int awaitingArrayName = (ARRAY_STATE | NAME_STATE | AWAITING_STATE);
static const int awaitingBufferName = (BUFFER_STATE | NAME_STATE | AWAITING_STATE);
static const int awaitingArrayLen = (ARRAY_STATE | NUMERIC_STATE | AWAITING_STATE);
static const int awaitingFieldName = (FIELD_STATE | NAME_STATE | AWAITING_STATE);
static const int awaitingTypeName = (TYPE_STATE | NAME_STATE | AWAITING_STATE);

static const int inArgsName = (ARGS_STATE | NAME_STATE | IN_STATE);
static const int inArrayName = (ARRAY_STATE | CAN_BE_ANON_STATE | NAME_STATE | IN_STATE);
static const int inBufferName = (BUFFER_STATE | NAME_STATE | IN_STATE);
static const int inArrayLen = (ARRAY_STATE | CAN_BE_FLEX_STATE | NUMERIC_STATE | IN_STATE);
static const int inBufferSize = (BUFFER_STATE | NUMERIC_STATE | IN_STATE);
static const int inFieldName = (FIELD_STATE | CAN_BE_ANON_STATE | NAME_STATE | IN_STATE);
static const int inTypeName = (TYPE_STATE | NAME_STATE | IN_STATE);

static const int haveArrayName = (ARRAY_STATE | NAME_STATE | HAVE_STATE);
static const int haveBufferName = (BUFFER_STATE | NAME_STATE | HAVE_STATE);
static const int haveArgsName = (ARGS_STATE | NAME_STATE | HAVE_STATE);
static const int haveFieldName = (FIELD_STATE | NAME_STATE | HAVE_STATE);
static const int haveTypeName = (TYPE_STATE | NAME_STATE | HAVE_STATE);

#define nameit(s) {s, #s}
std::map<int, std::string> Schema::stateNameMap = {
        nameit(awaitingArrayLen),
        nameit(awaitingArrayName),
        nameit(awaitingArgsName),
        nameit(inArgsName),
        nameit(inTypeName),
        nameit(inArrayLen),
        nameit(inArrayName),
        nameit(inBufferName),
        nameit(inBufferSize),
        nameit(haveArrayName),
        nameit(haveArgsName),
        nameit(haveTypeName),
        nameit(haveFieldName)
};

int Schema::replaceStateBit(int state, int remove, int set) {
    state |= set;
    state &= ~remove;
    return state;
}

int Schema::newState(int state, int to) {
    return to;
}

std::ostream &Schema::stateDescribe(std::ostream &out, int state) {
    out << "(";
    if (state & AWAITING_STATE) {
        out << "Awaiting";
    } else if (state & IN_STATE) {
        out << "In";
    } else if (state & HAVE_STATE) {
        out << "Have";
    }

    if (state & FIELD_STATE) {
        out << "Field";
    } else if (state & TYPE_STATE) {
        out << "Type";
    } else if (state & BUFFER_STATE) {
        out << "Buffer";
    } else if (state & ARRAY_STATE) {
        out << "Array";
    } else if (state & ARGS_STATE) {
        out << "Args";
    }

    if (state & NAME_STATE) {
        out << "Name";
    } else if (state & NUMERIC_STATE) {
        out << "Numeric";
    }

    if (state & CAN_BE_ANON_STATE) {
        out << "(? ok)";
    } else if (state & CAN_BE_FLEX_STATE) {
        out << "(* ok)";
    }
    out << ")";
    return out;
}

std::ostream &Schema::stateType(std::ostream &out, int state) {
    if (state & FIELD_STATE) {
        out << "field";
    } else if (state & TYPE_STATE) {
        out << "type";
    } else if (state & BUFFER_STATE) {
        out << "buffer";
    } else if (state & ARRAY_STATE) {
        out << "array";
    } else if (state & ARGS_STATE) {
        out << "args";
    } else {
        out << "WTF";
    }
    return out;
}

char *Schema::dumpSchema(std::ostream &out, int state, int depth, char *ptr, void *data) {
    char *start = nullptr;
    indent(out, depth);
    while (ptr != nullptr && *ptr != '\0') {
        if (stateNameMap.count(state) == 0) {
            std::cerr << "no key" << std::endl;
            exit(1);
        }
        stateDescribe(out, state) << "> with '" << ((char) *ptr) << "'";
        out.flush();
        if (state & AWAITING_STATE) {
            start = ptr;
            if (
                    (((state == awaitingArrayName) || (state == awaitingFieldName)) && (*ptr == '?'))
                    || ((state & NAME_STATE) && (std::isalpha(*ptr)))
                    || ((state & NUMERIC_STATE) && (std::isdigit(*ptr)))
                    || ((state == awaitingArrayLen) && (*ptr == '*'))
                    ) {
                state = replaceStateBit(state, AWAITING_STATE, IN_STATE);
                ptr++;
            } else if (((*ptr == ',') || (*ptr == '}') || (*ptr == ']') || (*ptr == '>'))) {
                ptr++;
            } else {
                std::cerr << "err " << "<" << stateNameMap[state] << "> with '" << ((char) *ptr) << "'" << ptr
                          << std::endl;
                exit(1);
            }
        } else if (state & IN_STATE) {
            if (
                    ((state & NAME_STATE) && (std::isalnum(*ptr) || *ptr == '_'))
                    ||  ((state & NUMERIC_STATE) && std::isdigit(*ptr))
                    || ((state == inBufferSize) && ((*ptr == '+') || (*ptr == '!')))
                    ){
                ptr++;
            } else if (*ptr == ':') {
                stateType(out, state);
                dump(out, start, ptr++);
                indent(out, depth);
                state = replaceStateBit(state, IN_STATE, HAVE_STATE);
            } else {
                std::cerr << "err " << "<" << stateNameMap[state] << "> with '" << ((char) *ptr) << "'" << ptr
                          << std::endl;
                exit(1);
            }
        } else if ((state & HAVE_STATE)) {
            switch (state) {
                case haveArrayName: {
                    // we expect a type
                    if (std::isdigit(*ptr)) {
                        start = ptr;
                        ptr++;
                        state = newState(state, inBufferSize);
                    } else if (std::isalpha(*ptr)) {
                        ptr++;
                        state = newState(state, inTypeName);
                    } else {
                        std::cerr << "err " << "<" << stateNameMap[state] << "> with '" << ((char) *ptr) << "'" << ptr
                                  << std::endl;
                        exit(1);
                    }
                    break;
                }
                case haveArgsName: {
                    // we expect a type name  or array, struct, union
                    if (*ptr == '[') {
                        ptr++;
                        state = newState(state, awaitingArrayLen);
                        // we expect a type
                    } else if ((*ptr == '{') || (*ptr == '<')) {
                        ptr++;
                        state = newState(state, awaitingFieldName);
                    } else if (std::isalnum(*ptr)) {
                        ptr++;
                        state =newState(state, inTypeName);
                    } else {
                        std::cerr << "err " << "<" << stateNameMap[state] << "> with '" << ((char) *ptr) << "'" << ptr
                                  << std::endl;
                        exit(1);
                    }
                    break;
                }
                default: {
                    std::cerr << "err " << "<" << stateNameMap[state] << "> with '" << ((char) *ptr) << "'" << ptr
                              << std::endl;
                    exit(1);
                }
            }
        } else {
            std::cerr << "err " << "<" << stateNameMap[state] << "> with '" << ((char) *ptr) << "'" << ptr
                      << std::endl;
            exit(1);
        }

    }
    return ptr;
}

char *Schema::dumpSchema(std::ostream &out, char *ptr, void *data) {
    return dumpSchema(out, awaitingArgsName, 0, ptr, data);
}

char *Schema::dumpSchema(std::ostream &out, char *ptr) {
    return dumpSchema(out, ptr, nullptr);
}

void Schema::dumpSled(std::ostream &out, void *argArray) {
    ArgSled argSled(static_cast<ArgArray_t *>(argArray));
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
                out << "Buf: of " << arg->value.buffer.sizeInBytes << " bytes " << std::endl;
                break;
            }
            case 'B': {
                out << "S8:" << arg->value.s8 << std::endl;
                break;
            }
            case 'Z': {
                out << "Z:" << arg->value.z1 << std::endl;
                break;
            }
            case 'C': {
                out << "U16:" << arg->value.u16 << std::endl;
                break;
            }
            case 'S': {
                out << "S16:" << arg->value.s16 << std::endl;
                break;
            }
            case 'I': {
                out << "S32:" << arg->value.s32 << std::endl;
                break;
            }
            case 'F': {
                out << "F32:" << arg->value.f32 << std::endl;
                break;
            }
            case 'J': {
                out << "S64:" << arg->value.s64 << std::endl;
                break;
            }
            case 'D': {
                out << "F64:" << arg->value.f64 << std::endl;
                break;
            }
            default: {
                std::cerr << "unexpected variant '" << (char) arg->variant << "'" << std::endl;
                exit(1);
            }
        }
    }
    out << "schema len = " << argSled.schemaLen() << std::endl;

    out << "schema = " << argSled.schema() << std::endl;

    // dumpSchema(out, argSled.schema()); not stable yet
}

// We need to trampoline through the real backend

extern "C" int getMaxComputeUnits(long backendHandle) {
   // std::cout << "trampolining through backendHandle to backend.getMaxComputeUnits()" << std::endl;
    Backend *backend = (Backend *) backendHandle;
    return backend->getMaxComputeUnits();
}

extern "C" void info(long backendHandle) {
  //  std::cout << "trampolining through backendHandle to backend.info()" << std::endl;
    Backend *backend = (Backend *) backendHandle;
    backend->info();
}
extern "C" void releaseBackend(long backendHandle) {
    Backend *backend = (Backend *) backendHandle;
    delete backend;
}
extern "C" long compileProgram(long backendHandle, int len, char *source) {
    std::cout << "trampolining through backendHandle to backend.compileProgram()" << std::endl;
    Backend *backend = (Backend *) backendHandle;
    return backend->compileProgram(len, source);
}
extern "C" long getKernel(long programHandle, int nameLen, char *name) {
  //  std::cout << "trampolining through programHandle to program.getKernel()" << std::endl;
    Backend::Program *program = (Backend::Program *) programHandle;
    return program->getKernel(nameLen, name);
}

extern "C" long ndrange(long kernelHandle, void *argArray) {
    std::cout << "trampolining through kernelHandle to kernel.ndrange(...) " << std::endl;

    Backend::Program::Kernel *kernel = (Backend::Program::Kernel *) kernelHandle;
    kernel->ndrange( argArray);
    return (long) 0;
}
extern "C" void releaseKernel(long kernelHandle) {
    Backend::Program::Kernel *kernel = (Backend::Program::Kernel *) kernelHandle;
    delete kernel;
}

extern "C" void releaseProgram(long programHandle) {
    Backend::Program *program = (Backend::Program *) programHandle;
    delete program;
}
extern "C" bool programOK(long programHandle) {
    Backend::Program *program = (Backend::Program *) programHandle;
    return program->programOK();
}


