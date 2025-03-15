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

  var hatDir =  Script.DirEntry.current();
  var backends = hatDir.existingDir("backends");
  var examples = hatDir.dir("examples");
  var buildDir = hatDir.existingBuildDir("build");
  var jextractedOpenCLJar = buildDir.jarFile("hat-jextracted-opencl-1.0.jar");
  var jextractedOpenGLJar = buildDir.jarFile("hat-jextracted-opengl-1.0.jar");
  var wrapJar = buildDir.jarFile("hat-wrap-1.0.jar");
  var clwrapJar = buildDir.jarFile("hat-clwrap-1.0.jar");
  var glwrapJar = buildDir.jarFile("hat-glwrap-1.0.jar");
  var verbose = false;

  var args = new ArrayList<>(List.of(argv));
  Script.java(java -> java
     .enable_preview()
     .verbose(verbose)
     //.add_exports("java.base", "jdk.internal", "ALL-UNNAMED")
     .enable_native_access("ALL-UNNAMED")
     .library_path(buildDir)
     .class_path(buildDir.jarFile("hat-core-1.0.jar"))
     .when((!args.isEmpty() && args.getFirst().equals("headless")), ifHeadless -> {
        ifHeadless.headless();
        args.removeFirst();
     })
     .either(!args.isEmpty(), haveBackend -> {
        var backendName = args.removeFirst();
        if (backends.dir(backendName.replace('-','/')) instanceof Script.DirEntry backend && backend.exists()) {
           haveBackend.class_path(buildDir.jarFile("hat-backend-" + backendName + "-1.0.jar"));
           if (backendName.equals("ffi-opencl")){
               haveBackend.class_path(wrapJar, clwrapJar, jextractedOpenCLJar);
           }
        } else {
           throw new RuntimeException("No such backend " + backendName);
        }
        if (!args.isEmpty() && args.removeFirst() instanceof String exampleName) {
           if (examples.dir(exampleName) instanceof Script.DirEntry example && example.exists()) { haveBackend
              .class_path(buildDir.jarFile("hat-example-" + exampleName + "-1.0.jar"))
              .when(jextractedOpenCLJar.exists() && jextractedOpenGLJar.exists() && exampleName.equals("nbody"), _->{ haveBackend
                  .class_path(jextractedOpenCLJar,jextractedOpenGLJar, wrapJar, clwrapJar, glwrapJar )
                  .start_on_first_thread();
              })
              .when(jextractedOpenCLJar.exists()  && exampleName.equals("life"), _->{ haveBackend
                  .class_path(jextractedOpenCLJar, wrapJar, clwrapJar);
              })
              .main_class(exampleName + ".Main")
              .args(args);
           } else {
              throw new RuntimeException("no such example " + exampleName);
           }
        }
     }, _ -> {
        throw new RuntimeException("No backend provided...\n" + usage);
     })
  );
}
