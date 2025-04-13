/*
 *
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


import static java.lang.System.out;

void main(String[] args) {
    var layout = """     
            └──./
                ├──hat                         All build scripts in each case 'foo' has java options for (and points to) 'foo.java'
                │    ├──bld                    --enable-preview --source 24 hat/bld.java
                │    ├──bld.java
                │    ├──run                    --enable-preview --source 24 hat/run.java
                │    ├──run.java
                │    └──Script                 Contains all the tools for building
                │
                ├──build/                            All jars, native libs and executables
                │    └──cmake-build-debug/           All intermediate cmake artifacts
                │        ├── hat-*wrap-1.0.jar              Wrapper jars around extracted * (clwrap, glwrap, cuwrap)
                │        ├── hat-core-1.0.jar               Base hat jar
                │        ├── hat-example-*-1.0.jar          Example jars (hat-example-nbody-1.0.jar, hat-example-life-1.0.jar)
                │        ├── hat-jextracted-opencl-1.0.jar  Raw jextracted jars (hat-jextracted-opencl-1.0.jar ....)
                │        ├── lib*_backend.[dylib|so]        ffi library backends
                │        └── *(no suffix)                   various generated executables (opencl_info, cuda_info, cuda_squares)
                ├──stage/
                │   ├── repo/
                │   │   └── *                      Maven artifacts (poms and jars)
                │   ├── opencl_jextracted/         All jextracted files (created using java @hat/extract
                │   │   ├── compile_flags.txt
                │   │   └── opencl
                │   ├── cuda_jextracted/
                │   │   ├── compile_flags.txt
                │   │   └── cuda
                │   └── opengl_jextracted/
                │       ├── compile_flags.txt
                │       └── opengl
                ├──wrap/
                │    └──wrap/                 All downloaded maven assets
                │         ├──wrap/                (*)
                │         ├──clwrap/              (*)
                │         ├──glwrap/              (*)
                │         └──cuwrap/              (*)
                │
                ├──hat-core                       Maven style layout
                │    ├──src/main/java
                │    │    └──hat/
                │    │
                │    └──src/main/test
                │         └──hat/
                │
                ├──backends
                │    ├──java
                │    │    ├──mt                    Maven style layout
                │    │    └──seq                   Maven style layout
                │    ├──jextracted
                │    │    └──opencl                Maven style layout
                │    └──ffi
                │         ├──opencl                Maven style layout with cmake
                │         ├──ptx                   Maven style layout with cmake
                │         ├──mock                  Maven style layout with cmake
                │         ├──spirv                 Maven style layout with cmake
                │         ├──cuda                  Maven style layout with cmake
                │         └──hip                   Maven style layout with cmake
                │
                └──examples
                     ├──mandel                     Maven style layout
                     ├──squares                    Maven style layout
                     ├──heal                       Maven style layout
                     ├──life                       Maven style layout
                     ├──nbody                      Maven style layout
                     ├──experiments                Maven style layout
                     └──violajones                 Maven style layout
            """;

    var dir = Script.DirEntry.current();
    var buildDir = Script.BuildDir.of(dir.path("build")).create();

    var jextractCapability = Script.Capabilities.JExtract.required();
    var cmakeCapability = Script.Capabilities.CMake.required();


    var openclCapability = Script.Capabilities.OpenCL.of();
    var openglCapability = Script.Capabilities.OpenGL.of();
    var cudaCapability = Script.Capabilities.CUDA.of();
    var hipCapability = Script.Capabilities.HIP.of();

    Script.Capabilities capabilities = Script.Capabilities.of(openclCapability, openglCapability, cudaCapability, hipCapability, jextractCapability, cmakeCapability);

    cmakeCapability.probe(buildDir, capabilities);

    var stageDir = dir.buildDir("stage").create();

    println(capabilities.tickOrCheck());

    Stream.of(openglCapability, openclCapability, cudaCapability,hipCapability)
            .filter(capability -> {
                          out.println(capability.tickOrCheck());
                if (!capability.available()) {
                    out.println("This platform does not have " + capability.name);
                }
                return capability.available();
            })
            .forEach(capability -> {
                var extractedDir = stageDir.buildDir(capability.packageName() + "_jextracted");
                if (!extractedDir.exists()) {
                    Script.jextract(jextractCapability.executable, $ -> $.capability(capability, extractedDir));
                } else {
                    out.println("Using previously extracted  " + extractedDir.fileName());
                }
                var extractedJar = buildDir.jarFile("hat-jextracted-" + capability.packageName() + "-1.0.jar");
                Script.jar(jar -> jar
                        .jarFile(extractedJar)
                        .javac(javac -> javac
                                .current_source()
                                .source_path(Script.SourceDir.of(extractedDir))
                        )
                );
            });

}

