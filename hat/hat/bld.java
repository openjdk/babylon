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
           │    ├── hat-wrap-*-1.0.jar             // Wrapper jars around extracted * (opencl, glwrap, opencl)
           │    ├── core-1.0.jar                   // Base hat jar
           │    ├── hat-example-*-1.0.jar          // Example jars (hat-example-nbody-1.0.jar, hat-example-life-1.0.jar)
           │    ├── hat-extracted-opencl-1.0.jar   // Raw extraction jars (hat-extracted-opencl-1.0.jar ....)
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
           │   ├──shared/
           │   │   └──src/main/java
           │   ├──cuda/
           │   │   └──src/main/java
           │   ├──opencl/
           │   │   └──src/main/java
           │   └──opengl/
           │       └──src/main/java
           │
           ├──core
           │    ├──src/main/java
           │    └──src/main/test
           │
           ├──tools  : core
           │    ├──src/main/java
           │    └──src/main/test
           │
           ├──backends
           │    ├──java
           │    │    ├──mt
           │    │    │    ├──src/main/java
           │    │    │    └──src/main/resources
           │    │    └──seq
           │    │         ├──src/main/java
           │    │         └──src/main/resources
           │    ├──jextracted
           │    │    └──opencl
           │    │         ├──src/main/java
           │    │         ├──src/main/native
           │    │         └──src/main/resources
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
                ├──shared
                │    ├──src/main/java
                │    └──src/main/resources
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
                ├──violajones
                │    ├──src/main/java
                │    └──src/main/resources
                └──matmul
                     ├──src/main/java
                     └──src/main/resources
       """;
    class Artifacts{
        static Script.MavenStyleProject core;
        static Script.MavenStyleProject tools;
        static Script.MavenStyleProject tests;
        static Script.MavenStyleProject example_shared;
        static Script.MavenStyleProject example_nbody;
        static Script.MavenStyleProject backend_ffi_shared;
        static Script.MavenStyleProject backend_ffi_cuda;
        static Script.MavenStyleProject backend_ffi_mock;
        static Script.MavenStyleProject backend_ffi_opencl;
        static Script.MavenStyleProject backend_java_seq;
        static Script.MavenStyleProject backend_java_mt;
        static Script.MavenStyleProject extraction_cuda;
        static Script.MavenStyleProject extraction_opengl;
        static Script.MavenStyleProject extraction_opencl;
        static Script.MavenStyleProject backend_jextracted_cuda;
        static Script.MavenStyleProject backend_jextracted_opencl;
        static Script.MavenStyleProject backend_jextracted_shared;
        static Script.MavenStyleProject wrap_shared;
        static Script.MavenStyleProject wrap_cuda;
        static Script.MavenStyleProject wrap_opengl;
        static Script.MavenStyleProject wrap_opencl;
    }
    var dir = Script.DirEntry.current();
    var buildDir = Script.BuildDir.of(dir.path("build")).create();

    Artifacts.core = buildDir.mavenStyleBuild(
            dir.existingDir("core"), "hat-core-1.0.jar"
    );

    Artifacts.tools = buildDir.mavenStyleBuild(
            dir.existingDir("tools"), "hat-tools-1.0.jar", Artifacts.core
    );

    Artifacts.tests = buildDir.mavenStyleBuild(
            dir.existingDir("tests"), "hat-tests-1.0.jar", Artifacts.core
    );

    var extractionsDir = dir.existingDir("extractions");

    var extractionsCmakeBuildDir = extractionsDir.buildDir("cmake-build-debug");
    if (!extractionsCmakeBuildDir.exists()) {
        Script.cmake($ -> $ .verbose(false) .source_dir(extractionsDir) .build_dir(extractionsCmakeBuildDir));
    }
    Script.cmake($ -> $ .build(extractionsCmakeBuildDir) .target("extract"));

    var extraction_opencl_dir = extractionsDir.dir("opencl");
    if (extraction_opencl_dir.dir("src").exists()) {
        Artifacts.extraction_opencl = buildDir.mavenStyleBuild(
                extraction_opencl_dir, "hat-extracted-opencl-1.0.jar"
        );
    }else{
        print("no src for extraction_opencl");
    }

    var extraction_opengl_dir = extractionsDir.dir("opengl");
    if (extraction_opengl_dir.dir("src").exists()) {
        Artifacts.extraction_opengl = buildDir.mavenStyleBuild(
                extraction_opengl_dir, "hat-extracted-opengl-1.0.jar"
        );
    }else{
        print("no src for extraction_opengl");
    }

    var extraction_cuda_dir = extractionsDir.dir("cuda");
    if (extraction_cuda_dir.dir("src").exists()) {
        Artifacts.extraction_cuda = buildDir.mavenStyleBuild(
                extraction_cuda_dir, "hat-extracted-cuda-1.0.jar"
        );
    }


    var wrapsDir = dir.existingDir("wraps");

    Artifacts.wrap_shared = buildDir.mavenStyleBuild( wrapsDir.existingDir("shared"), "hat-wrap-shared-1.0.jar");

    if (Artifacts.extraction_opencl != null){
        Artifacts.wrap_opencl = buildDir.mavenStyleBuild( wrapsDir.dir("opencl"), "hat-wrap-opencl-1.0.jar", Artifacts.wrap_shared, Artifacts.core, Artifacts.extraction_opencl);
    }
// on jetson
// ls extractions/opengl/src/main/java/opengl/glutKeyboardFunc*
//  -> extractions/opengl/src/main/java/opengl/glutKeyboardFunc$callback.java
//  so we exclude "^.*/wrap/opengl/GLFuncEventHandler\\.java$"
// on mac
//    ls extractions/opengl/src/main/java/opengl/glutKeyboardFunc*
//  -> extractions/opengl/src/main/java/opengl/glutKeyboardFunc$func.java
//  So we exclude  "^.*/wrap/opengl/GLCallbackEventHandler\\.java$"
//

    if (Artifacts.extraction_opengl != null
            && Artifacts.extraction_opengl.jarFile.exists()) {
        String exclude = null;
        if (!Artifacts.extraction_opengl.jarFile.select(Script.Regex.of("^.*glutKeyboardFunc\\$func.class$")).isEmpty()) {
            exclude = "Callback";
        }else if (!Artifacts.extraction_opengl.jarFile.select(Script.Regex.of("^.*glutKeyboardFunc\\$callback.class$")).isEmpty()) {
            exclude = "Func";
        }else {
            println("We can't build wrap_opengl unless exclude one of GLFuncEventHandler or GLCallbackEventHandler something");
        }
        if (exclude != null) {
            final var excludeMeSigh = "^.*/GL"+exclude+"EventHandler\\.java$";
            println("exclude ="+exclude+" "+excludeMeSigh);
            Artifacts.wrap_opengl = Script.mavenStyleProject(
                    buildDir, wrapsDir.dir("opengl"), buildDir.jarFile("hat-wrap-opengl-1.0.jar"), Artifacts.wrap_shared, Artifacts.core, Artifacts.extraction_opengl
            ).buildExcluding(javaSrc -> javaSrc.matches(excludeMeSigh));
        }
    }

    if (false && Artifacts.extraction_cuda != null ) {
        Artifacts.wrap_cuda = buildDir.mavenStyleBuild(
                wrapsDir.dir("cuda"), "hat-wrap-cuda-1.0.jar", Artifacts.extraction_cuda
        );
    }

    var backendsDir = dir.existingDir("backends");

    var ffiBackendsDir = backendsDir.existingDir("ffi");
    Artifacts.backend_ffi_shared = buildDir.mavenStyleBuild(
            ffiBackendsDir.existingDir("shared"), "hat-backend-ffi-shared-1.0.jar", Artifacts.core
    );

    if (ffiBackendsDir.optionalDir("opencl") instanceof Script.DirEntry ffiBackendDir ) {
        Artifacts.backend_ffi_opencl = buildDir.mavenStyleBuild(
                ffiBackendDir, "hat-backend-ffi-"+ffiBackendDir.fileName()+ "-1.0.jar", Artifacts.core, Artifacts.backend_ffi_shared
        );
    }
    if (ffiBackendsDir.optionalDir("mock") instanceof Script.DirEntry ffiBackendDir) {
        Artifacts.backend_ffi_mock = buildDir.mavenStyleBuild(
                ffiBackendDir, "hat-backend-ffi-"+ffiBackendDir.fileName()+ "-1.0.jar", Artifacts.core, Artifacts.backend_ffi_shared
        );
    }

    if (ffiBackendsDir.optionalDir("cuda") instanceof Script.DirEntry ffiBackendDir) {
        Artifacts.backend_ffi_cuda = buildDir.mavenStyleBuild(
                ffiBackendDir, "hat-backend-ffi-"+ffiBackendDir.fileName()+ "-1.0.jar", Artifacts.core, Artifacts.backend_ffi_shared
        );
    }

    var jextractedBackendsDir = backendsDir.existingDir("jextracted");

    Artifacts.backend_jextracted_shared = buildDir.mavenStyleBuild(
            jextractedBackendsDir.existingDir("shared"), "hat-backend-jextracted-shared-1.0.jar", Artifacts.core
    );

    if (Artifacts.extraction_opencl != null && jextractedBackendsDir.optionalDir("opencl") instanceof Script.DirEntry jextractedBackendDir) {
        Artifacts.backend_jextracted_opencl = buildDir.mavenStyleBuild(
                jextractedBackendDir, "hat-backend-jextracted-" + jextractedBackendDir.fileName() + "-1.0.jar",
                Artifacts.core, Artifacts.extraction_opencl, Artifacts.backend_jextracted_shared
        );
    }

    if (Artifacts.extraction_cuda != null && jextractedBackendsDir.optionalDir("cuda") instanceof Script.DirEntry jextractedBackendDir) {
        Artifacts.backend_jextracted_cuda = buildDir.mavenStyleBuild(
                jextractedBackendDir, "hat-backend-jextracted-" + jextractedBackendDir.fileName() + "-1.0.jar",
                Artifacts.core, Artifacts.extraction_cuda, Artifacts.backend_jextracted_shared
        );
    }

    var javaBackendsDir = backendsDir.existingDir("java");
    Artifacts.backend_java_mt =  buildDir.mavenStyleBuild(javaBackendsDir.existingDir("mt"),
            "hat-backend-java-mt-1.0.jar", Artifacts.core
    );
    Artifacts.backend_java_seq =  buildDir.mavenStyleBuild(javaBackendsDir.existingDir("seq"),
            "hat-backend-java-seq-1.0.jar", Artifacts.core
    );

    var examplesDir = dir.existingDir("examples");


    Stream.of( "blackscholes", "squares", "matmul")
            .parallel()
            .map(examplesDir::existingDir)
            .forEach(exampleDir->buildDir.mavenStyleBuild(
                    exampleDir, "hat-example-"+exampleDir.fileName()+"-1.0.jar", Artifacts.core
            ));

    Stream.of( "experiments")   // this has hardcoded references to opencl backend
            .parallel()
            .map(examplesDir::existingDir)
            .forEach(exampleDir->buildDir.mavenStyleBuild(
                    exampleDir, "hat-example-"+exampleDir.fileName()+"-1.0.jar",
                    Artifacts.core, Artifacts.backend_ffi_shared, Artifacts.backend_ffi_opencl
            ));

    Artifacts.example_shared = buildDir.mavenStyleBuild(
            examplesDir.existingDir("shared"), "hat-example-shared-1.0.jar", Artifacts.core
    );

    Stream.of( "heal", "life", "mandel", "violajones")   // these require example_shared ui stuff
            .parallel()
            .map(examplesDir::existingDir)
            .forEach(exampleDir->buildDir.mavenStyleBuild(
                    exampleDir, "hat-example-"+exampleDir.fileName()+"-1.0.jar", Artifacts.core, Artifacts.example_shared
            ));

    var nbodyDependencies = new Script.MavenStyleProject[]{
            Artifacts.core,
            Artifacts.wrap_shared,
            Artifacts.wrap_opencl,
            Artifacts.extraction_opencl,
            Artifacts.backend_ffi_opencl,
            Artifacts.wrap_opengl,
            Artifacts.extraction_opengl
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
        Artifacts.example_nbody = buildDir.mavenStyleBuild(
                examplesDir.existingDir("nbody"), "hat-example-nbody-1.0.jar", nbodyDependencies
        );
    }

    var cmakeBuildDir = buildDir.buildDir("cmake-build-debug");
    if (!cmakeBuildDir.exists()) {
        Script.cmake($ -> $ .verbose(false) .source_dir(ffiBackendsDir) .build_dir(cmakeBuildDir) .copy_to(buildDir));
    }
    Script.cmake($ -> $ .build(cmakeBuildDir));

}

