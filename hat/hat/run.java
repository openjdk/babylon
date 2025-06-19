/* vim: set ft=java:
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

import static java.lang.IO.print;
import static java.lang.IO.println;

class Config{
     boolean headless=false;
     boolean verbose = false;
     boolean startOnFirstThread = false;
     boolean justShowCommandline = false;
     String backendName = null;
     Script.JarFile backendJar= null;
     String exampleName = null;
     String examplePackageName = null;
     String exampleClassName =  null;
     Script.JarFile exampleJar= null;
     List<Script.ClassPathEntryProvider> classpath = new ArrayList<>();
     List<String> vmargs = new ArrayList<>();
     List<String> appargs = new ArrayList<>();
     Config(Script.BuildDir buildDir,  String[] args){

        classpath.add(buildDir.jarFile("hat-core-1.0.jar"));
        for (int arg=0;arg<args.length;arg++){
            if (args[arg].startsWith("ffi-")) {
                backendName = args[arg];
                backendJar = buildDir.jarFile("hat-backend-" + backendName + "-1.0.jar");
                classpath.add(buildDir.jarFile("hat-backend-ffi-shared-1.0.jar"));
                classpath.add(backendJar);
                if (verbose){
                    println("backend "+backendName);
                }
            }else if (args[arg].startsWith("ext--")){
                backendName = args[arg];
                backendJar = buildDir.jarFile("hat-backend-" + backendName + "-1.0.jar");
                classpath.add(buildDir.jarFile("hat-backend-jextracted-shared-1.0.jar"));
                classpath.add(backendJar);
                if (verbose){
                    println("backend "+backendName);
                }
            }else if (args[arg].startsWith("java-")){
                backendName = args[arg];
                backendJar = buildDir.jarFile("hat-backend-"+backendName+"-1.0.jar");
                classpath.add(backendJar);
                if (verbose){
                    println("backend "+backendName);
                }
            }else{
                switch (args[arg]) {
                   case "headless" -> headless = true;
                   case "verbose" -> verbose = true;
                   case "justShowCommandLine" -> justShowCommandline = true;
                   case "startOnFirstThread" -> startOnFirstThread = true;
                   default ->{
                       var proposedExampleName = args[arg];
                       int lastDot = proposedExampleName.lastIndexOf('.');
                       var proposedExampleClass="Main";
                       var proposedPackageName=args[arg];
                       if (lastDot != -1){
                           proposedExampleClass = proposedExampleName.substring(lastDot + 1);
                           proposedPackageName = proposedExampleName.substring(0,lastDot);
                       }
                       var proposedJar = buildDir.jarFile("hat-example-"+proposedPackageName+"-1.0.jar");
                       if (proposedJar.exists()) {
                           exampleClassName = proposedExampleClass;
                           examplePackageName =exampleName = proposedPackageName;
                           exampleJar = proposedJar;
                           if (exampleJar.exists()){
                               classpath.add(exampleJar);
                               if (verbose){
                                   println("example "+examplePackageName+"."+exampleClassName);
                               }
                           }else{
                               if (exampleClassName == null) {
                                   this.vmargs.add(args[arg]);
                               }else{
                                   this.appargs.add(args[arg]);
                               }
                           }
                       }else{
                           if (exampleClassName == null) {
                               this.vmargs.add(args[arg]);
                           }else{
                               this.appargs.add(args[arg]);
                           }
                       }
                   }
                }
            }
        }
    }
}


void main(String[] argv) {
  var usage ="""
    usage:
      java @bldr/args hatrun [headless] backend package args ...
         [headless] : Optional passes -Dheadless=true to app
          backend   : opencl|cuda|spirv|ptx|mock
          package   : the examples package (and dirname under hat/examples)

      class name is assumed to be package.Main  (i.e. mandel.main)

      examples:
         java @bldr/args ffi-opencl mandel
         java @bldr/args java-opencl mandel
         java @bldr/args headless ffi-opencl mandel
         java @bldr/args ffi-opencl life
         java @bldr/args java-opencl life
  """;

      var hatDir = Script.DirEntry.current();
      var buildDir = hatDir.existingBuildDir("build");

      Config config = new Config(buildDir,argv);
      if (config.classpath.isEmpty() ) {
          println("Classpath is empty!");
      }else if (config.backendJar == null || !config.backendJar.exists()) {
          println("No backend !");
      }else if (!config.exampleJar.exists()){
          println("No example !");
      }else{
          var jextractedOpenCLJar = buildDir.jarFile("hat-jextracted-opencl-1.0.jar");
          var jextractedOpenGLJar = buildDir.jarFile("hat-jextracted-opengl-1.0.jar");
          var wrapJar = buildDir.jarFile("hat-wrap-1.0.jar");
          var clwrapJar = buildDir.jarFile("hat-clwrap-1.0.jar");
          var glwrapJar = buildDir.jarFile("hat-glwrap-1.0.jar");
          switch (config.backendName){
             default -> {}
          }
          switch (config.exampleName){
              case "nbody" -> {
                  if (Script.os instanceof Script.OS.Mac){
                     println("For MAC we added  --startOnFirstThread");
                     config.startOnFirstThread = true;
                  }
                  config.classpath.addAll(List.of(
                          wrapJar,
                          clwrapJar, jextractedOpenCLJar,
                          glwrapJar, jextractedOpenGLJar,
                          clwrapJar, jextractedOpenCLJar)
                  );
              }
              default -> {}
          }
      }
      Script.java(java -> java
              .enable_preview()
              .verbose(true)
              .enable_native_access("ALL-UNNAMED")
              .library_path(buildDir)
              .when(config.headless, Script.JavaBuilder::headless)
              .when(config.startOnFirstThread, Script.JavaBuilder::start_on_first_thread)
              .class_path(config.classpath)
              .vmargs(config.vmargs)
              .main_class(config.examplePackageName + "."+config.exampleClassName)
              .args(config.appargs)
              .justShowCommandline(config.justShowCommandline)
      );

}
