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
   ArgSled argSled((ArgArray_t*)ptr);
   std::cout  << "ArgArray->argc = " << argSled.argc()  << std::endl;
   for (int i=0; i<argSled.argc(); i++){
      argSled.dumpArg(i);
   }
   std::cout<< "schema = "<< argSled.schema() << std::endl;


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
   unsigned char *buf = (unsigned char*)ptr;
   int i, j;
   for (i=0; i<buflen; i+=16) {
      printf("%06x: ", i);
      for (j=0; j<16; j++)
         if (i+j < buflen)
            printf("%02x ", buf[i+j]);
         else
            printf("   ");
      printf(" ");
      for (j=0; j<16; j++)
         if (i+j < buflen)
            printf("%c", isprint(buf[i+j]) ? buf[i+j] : '.');
      printf("\n");
   }
}

// We need to trampoline through the real backend

extern "C"  int getMaxComputeUnits(long backendHandle) {
   Backend *backend = (Backend*)backendHandle;
   return backend->getMaxComputeUnits();
}

extern "C" void info(long backendHandle) {
   Backend *backend = (Backend*)backendHandle;
   backend->info();
}
extern "C" void releaseBackend(long backendHandle) {
   Backend* backend = (Backend*)backendHandle;
   delete backend;
}
extern "C" long compileProgram(long backendHandle, int len, char *source) {
   //std::cout << "trampolining through backendHandle to compileProgram" <<std::endl;
   Backend *backend = (Backend*)backendHandle;
   return backend->compileProgram(len, source);
}
extern "C" long getKernel(long programHandle, int nameLen, char *name) {
   //std::cout << "trampolining through programHandle to get kernel" <<std::endl;
   Backend::Program *program = (Backend::Program *)programHandle;
   return program->getKernel(nameLen, name);
}

extern "C" long ndrange(long kernelHandle, int range, void* argArray) {
   //std::cout << "trampolining through kernelHandle to dispatch " <<std::endl;
   Backend::Program::Kernel *kernel = (Backend::Program::Kernel *)kernelHandle;
   kernel->ndrange(range, argArray);
   return (long)0;
}
extern "C" void releaseKernel(long kernelHandle) {
   Backend::Program::Kernel *kernel = (Backend::Program::Kernel *)kernelHandle;
   delete kernel;
}

extern "C" void releaseProgram(long programHandle) {
   Backend::Program *program = (Backend::Program *)programHandle;
   delete program;
}
extern "C" bool programOK(long programHandle) {
   Backend::Program *program = (Backend::Program *)programHandle;
   return program->programOK();
}


