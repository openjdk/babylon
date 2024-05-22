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

class MockBackend: public Backend{
   public:
      class MockConfig: public Backend::Config{
         public :
      };
      class MockProgram : public Backend::Program{
         class MockKernel : public Backend::Program::Kernel{
            public:
               MockKernel(Backend::Program *program):Backend::Program::Kernel(program){
               }
               ~MockKernel(){
               }
               long ndrange( int range, void *argArray) {
                  std::cout<<"mock ndrange("<<range<<") "<< std::endl;
                  return 0;
               }
         };
         public:
         MockProgram(Backend *backend, BuildInfo *buildInfo ):Backend::Program(backend, buildInfo){
         }
         ~MockProgram(){
         }
         long getKernel(int nameLen, char *name){
            return (long) new MockKernel(this);
         }
         bool programOK(){
            return true;
         }
      };

   public:

      MockBackend(MockConfig *mockConfig, int mockConfigSchemeLen, char *mockBackendSchema):Backend(mockConfig,mockConfigSchemeLen,mockBackendSchema) {
         if (mockConfig == nullptr){
            std::cout << "mockConfig == null"<< std::endl;
         }else{
            std::cout << "mockConfig != null" <<std::endl;
         }
      }

      ~MockBackend() {
      }

      int getMaxComputeUnits(){
         std::cout << "mock getMaxComputeUnits()"<<std::endl;
         return 0;
      }

      void info(){
         std::cout << "mock info()"<<std::endl;
      }

      long compileProgram(int len, char *source){
         std::cout << "mock compileProgram()"<<std::endl;
         size_t srcLen = ::strlen(source);
         char *src = new char[srcLen + 1];
         ::strncpy(src, source, srcLen);
         src[srcLen] = '\0';
         std::cout << "native compiling " << src << std::endl;
         return (long) new MockProgram(this, new BuildInfo(src, nullptr,false));
      }
};

long getBackend(void *config, int configSchemaLen, char *configSchema){
   MockBackend::MockConfig *mockConfig = (MockBackend::MockConfig*)config;
   return (long)new MockBackend(mockConfig,configSchemaLen,configSchema);
}
