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

import static java.lang.IO.print;
import static java.lang.IO.println;

void main(String[] args) {
    var layout = """
       └──./
           ├──hat                                  //  All build scripts in each case 'foo' has java options for (and points to) 'foo.java'
           │    ├──bld                             //  --enable-preview --source 26 hat/bld.java
           │    ├──bld.java
           │    ├──run                             //  --enable-preview --source 26 hat/run.java
           │    ├──run.java
           │    └──Script                          //  Contains all the tools for building
           ├──build/                               // All jars, native libs and executables
           │    ├── cmake-build-debug/             // All intermediate cmake artifacts
           │    ├── hat-*wrap-1.0.jar              // Wrapper jars around extracted * (clwrap, glwrap, cuwrap)
           │    ├── core-1.0.jar                   // Base hat jar
           │    ├── hat-example-*-1.0.jar          // Example jars (hat-example-nbody-1.0.jar, hat-example-life-1.0.jar)
           │    ├── hat-jextracted-opencl-1.0.jar  // Raw jextracted jars (hat-jextracted-opencl-1.0.jar ....)
           │    ├── lib*_backend.[dylib|so]        // ffi library backends
           │    └── *(no suffix)                   // various generated executables (opencl_info, cuda_info, cuda_squares)
           ├──extractions/
           │   ├──CMakeFiles.txt
           │   ├── opencl/
           │   │   └──CMakeFiles.txt
           │   ├── cuda/
           │   │   └──CMakeFiles.txt
           │   └── opengl/
           │       └──CMakeFiles.txt
           ├──wraps/
           │   ├──wrap/
           │   │   └──src/main/java
           │   ├──cuwrap/
           │   │   └──src/main/java
           │   ├──clwrap/
           │   │   └──src/main/java
           │   ├──glwrap/
           │   │   └──src/main/java
           │   └──cuwrap/
           │
           ├──core
           │    ├──src/main/java
           │    └──src/main/test
           │
           ├──backends
           │    ├──java
           │    │    ├──mt
           │    │    └──seq
           │    ├──jextracted
           │    │    └──opencl
           │    └──ffi
           │         ├──CMakeFiles.txt
           │         ├──opencl
           │         │    ├──CMakeFiles.txt
           │         │    ├──src/main/java
           │         │    ├──src/main/native
           │         │    └──src/main/resources
           │         ├──cuda
           │         │    ├──CMakeFiles.txt
           │         │    ├──src/main/java
           │         │    ├──src/main/native
           │         │    └──src/main/resources
           │         ├──mock
           │         │    ├──CMakeFiles.txt
           │         │    ├──src/main/java
           │         │    ├──src/main/native
           │         │    └──src/main/resources
           │         ├──spirv
           │         │    ├──CMakeFiles.txt
           │         │    ├──src/main/java
           │         │    ├──src/main/native
           │         │    └──src/main/resources
           │         └──hip
           │              ├──CMakeFiles.txt
           │              ├──src/main/java
           │              ├──src/main/native
           │              └──src/main/resources
           │
           └──examples
                ├──mandel
                │    ├──src/main/java
                │    └──src/main/resources
                ├──squares
                │    ├──src/main/java
                │    └──src/main/resources
                ├──heal
                │    ├──src/main/java
                │    └──src/main/resources
                ├──life
                │    ├──src/main/java
                │    └──src/main/resources
                ├──nbody
                │    ├──src/main/java
                │    └──src/main/resources
                ├──experiments
                │    ├──src/main/java
                │    └──src/main/resources
                └──violajones
                     ├──src/main/java
                     └──src/main/resources
       """;
    class Artifacts{
        static Script.MavenStyleProject javaSeqBackend;
        static Script.MavenStyleProject javaMTBackend;
        static Script.MavenStyleProject exampleNbody;
        static Script.MavenStyleProject ffiBackendCuda;
        static Script.MavenStyleProject ffiBackendMock;
        static Script.MavenStyleProject ffiBackendOpenCL;
        static Script.MavenStyleProject jextractedBackendCuda;
        static Script.MavenStyleProject jextractedBackendOpenCL;
        static Script.MavenStyleProject jextractedBackendShared;
        static Script.MavenStyleProject ffiBackendShared;
        static Script.MavenStyleProject cuWrap;
        static Script.MavenStyleProject glWrap;
        static Script.MavenStyleProject clWrap;
        static Script.MavenStyleProject jextractedCuda;
        static Script.MavenStyleProject jextractedOpenGL;
        static Script.MavenStyleProject jextractedOpenCL;
        static Script.MavenStyleProject wrap;
        static Script.MavenStyleProject hatCore;
    }
    var dir = Script.DirEntry.current();
    var buildDir = Script.BuildDir.of(dir.path("build")).create();

    Artifacts.hatCore = buildDir.mavenStyleBuild(
            dir.existingDir("core"),
            "core-1.0.jar"
    );


    var extractionsDir = dir.existingDir("extractions");

    var extractionsCmakeBuildDir = extractionsDir.buildDir("cmake-build-debug");
    if (!extractionsCmakeBuildDir.exists()) {
        Script.cmake($ -> $
                .verbose(false)
                .source_dir(extractionsDir)
                .build_dir(extractionsCmakeBuildDir)
        );
    }
    Script.cmake($ -> $
            .build(extractionsCmakeBuildDir)
            .target("extract")
    );

    var jextractedOpenCLDir = extractionsDir.dir("opencl");
    if (jextractedOpenCLDir.dir("src").exists()) {
        Artifacts.jextractedOpenCL = buildDir.mavenStyleBuild(
                jextractedOpenCLDir,
                "hat-jextracted-opencl-1.0.jar"
        );
    }else{
        print("no src for jextractedOpenCL");
    }

    var jextractedOpenGLDir = extractionsDir.dir("opengl");
    if (jextractedOpenGLDir.dir("src").exists()) {
        Artifacts.jextractedOpenGL = buildDir.mavenStyleBuild(
                jextractedOpenGLDir,
                "hat-jextracted-opengl-1.0.jar");
    }else{
        print("no src for jextractedOpenGL");
    }

    var jextractedCudaDir = extractionsDir.dir("cuda");
    if (jextractedCudaDir.dir("src").exists()) {
        Artifacts.jextractedCuda = buildDir.mavenStyleBuild(
                jextractedCudaDir,
                "hat-jextracted-cuda-1.0.jar"
        );
    }


    var wrapsDir = dir.existingDir("wraps");

    Artifacts.wrap = buildDir.mavenStyleBuild(
            wrapsDir.existingDir("wrap"),
            "hat-wrap-1.0.jar"
    );

    if (Artifacts.jextractedOpenCL != null){
    Artifacts.clWrap = buildDir.mavenStyleBuild(
            wrapsDir.dir("clwrap"), "hat-clwrap-1.0.jar",
            Artifacts.wrap, Artifacts.hatCore, Artifacts.jextractedOpenCL
    );
}
// on jetson
// ls extractions/opengl/src/main/java/opengl/glutKeyboardFunc*
//  -> extractions/opengl/src/main/java/opengl/glutKeyboardFunc$callback.java
//  so we exclude "^.*/wrap/glwrap/GLFuncEventHandler\\.java$"
// on mac
//    ls extractions/opengl/src/main/java/opengl/glutKeyboardFunc*
//  -> extractions/opengl/src/main/java/opengl/glutKeyboardFunc$func.java
//  So we exclude  "^.*/wrap/glwrap/GLCallbackEventHandler\\.java$"
//

if (Artifacts.jextractedOpenGL != null
        && Artifacts.jextractedOpenGL.jarFile.exists()) {
    String exclude = null;
    if (!Artifacts.jextractedOpenGL.jarFile.select(Script.Regex.of("^.*glutKeyboardFunc\\$func.class$")).isEmpty()) {
        exclude = "Callback";
    }else if (!Artifacts.jextractedOpenGL.jarFile.select(Script.Regex.of("^.*glutKeyboardFunc\\$callback.class$")).isEmpty()) {
        exclude = "Func";
    }else {
        println("We can't build glwrap unless exclude one of GLFuncEventHandler or GLCallbackEventHandler something");
    }
    if (exclude != null) {
        final var excludeMeSigh = "^.*/GL"+exclude+"EventHandler\\.java$";
        println("exclude ="+exclude+" "+excludeMeSigh);
        Artifacts.glWrap = Script.mavenStyleProject(buildDir,
                wrapsDir.dir("glwrap"),
                buildDir.jarFile("hat-glwrap-1.0.jar"),
                Artifacts.wrap, Artifacts.hatCore, Artifacts.jextractedOpenGL
        ).buildExcluding(javaSrc -> javaSrc.matches(excludeMeSigh));

    }
}

    if (false && Artifacts.jextractedCuda != null ) {
        Artifacts.cuWrap = buildDir.mavenStyleBuild(
                wrapsDir.dir("cuwrap"), "hat-cuwrap-1.0.jar",
                Artifacts.jextractedCuda
        );
    }

    var backendsDir = dir.existingDir("backends");

    var ffiBackendsDir = backendsDir.existingDir("ffi");
    Artifacts.ffiBackendShared = buildDir.mavenStyleBuild(
            ffiBackendsDir.existingDir("shared"), "hat-backend-ffi-shared-1.0.jar",
            Artifacts.hatCore
    );

    if (ffiBackendsDir.optionalDir("opencl") instanceof Script.DirEntry ffiBackendDir ) {
        Artifacts.ffiBackendOpenCL = buildDir.mavenStyleBuild(
                ffiBackendDir,
                "hat-backend-ffi-"+ffiBackendDir.fileName()+ "-1.0.jar",
                Artifacts.hatCore, Artifacts.ffiBackendShared
        );
    }
    if (ffiBackendsDir.optionalDir("mock") instanceof Script.DirEntry ffiBackendDir) {
        Artifacts.ffiBackendMock = buildDir.mavenStyleBuild(
                ffiBackendDir,
                "hat-backend-ffi-"+ffiBackendDir.fileName()+ "-1.0.jar",
                Artifacts.hatCore, Artifacts.ffiBackendShared
        );
    }

    if (ffiBackendsDir.optionalDir("cuda") instanceof Script.DirEntry ffiBackendDir) {
        Artifacts.ffiBackendCuda = buildDir.mavenStyleBuild(
                ffiBackendDir,
                "hat-backend-ffi-"+ffiBackendDir.fileName()+ "-1.0.jar",
                Artifacts.hatCore, Artifacts.ffiBackendShared
        );
    }

    var jextractedBackendsDir = backendsDir.existingDir("jextracted");

    Artifacts.jextractedBackendShared = buildDir.mavenStyleBuild(
            jextractedBackendsDir.existingDir("shared"),
            "hat-backend-jextracted-shared-1.0.jar",
            Artifacts.hatCore
    );

    if (Artifacts.jextractedOpenCL != null && jextractedBackendsDir.optionalDir("opencl") instanceof Script.DirEntry jextractedBackendDir) {
        Artifacts.jextractedBackendOpenCL = buildDir.mavenStyleBuild(
                jextractedBackendDir,
                "hat-backend-jextracted-" + jextractedBackendDir.fileName() + "-1.0.jar",
                Artifacts.hatCore, Artifacts.jextractedOpenCL, Artifacts.jextractedBackendShared
        );
    }

    if (Artifacts.jextractedCuda != null && jextractedBackendsDir.optionalDir("cuda") instanceof Script.DirEntry jextractedBackendDir) {
        Artifacts.jextractedBackendCuda = buildDir.mavenStyleBuild(
                jextractedBackendDir,
                "hat-backend-jextracted-" + jextractedBackendDir.fileName() + "-1.0.jar",
                Artifacts.hatCore, Artifacts.jextractedCuda, Artifacts.jextractedBackendShared
        );
    }

    var javaBackendsDir = backendsDir.existingDir("java");
    Artifacts.javaMTBackend =  buildDir.mavenStyleBuild(javaBackendsDir.existingDir("mt"),
            "hat-backend-java-mt-1.0.jar",
            Artifacts.hatCore
    );
    Artifacts.javaSeqBackend =  buildDir.mavenStyleBuild(javaBackendsDir.existingDir("mt"),
            "hat-backend-java-seq-1.0.jar",
            Artifacts.hatCore
    );

    var examplesDir = dir.existingDir("examples");
    Stream.of(
            "blackscholes",
                    "heal",
                    "life",
                    "mandel",
                    "squares",
                    "violajones"
            )
            .parallel()
            .map(examplesDir::existingDir)
            .forEach(exampleDir->buildDir.mavenStyleBuild(exampleDir,
                    "hat-example-"+exampleDir.fileName()+"-1.0.jar",
                    Artifacts.hatCore
                    )
            );

    var nbodyDependencies = new Script.MavenStyleProject[]{
            Artifacts.hatCore,
            Artifacts.wrap,
            Artifacts.clWrap,
            Artifacts.jextractedOpenCL,
            Artifacts.ffiBackendOpenCL,
            Artifacts.glWrap,
            Artifacts.jextractedOpenGL
    };
    boolean foundNull = false;

    for (var o:nbodyDependencies){
       if (o == null){
          foundNull = true;
       }
    }
    if (foundNull){
        print("incomplete nbody dependencies ");
    }else {
        Artifacts.exampleNbody = buildDir.mavenStyleBuild(examplesDir.existingDir("nbody"),
                "hat-example-nbody-1.0.jar",
                nbodyDependencies
        );
    }

        var cmakeBuildDir = buildDir.buildDir("cmake-build-debug");
        if (!cmakeBuildDir.exists()) {
            Script.cmake($ -> $
                    .verbose(false)
                    .source_dir(ffiBackendsDir)
                    .build_dir(cmakeBuildDir)
                    .copy_to(buildDir)
            );
        }
        Script.cmake($ -> $
                .build(cmakeBuildDir)
        );


}

