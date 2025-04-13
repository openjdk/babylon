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


static class MavenStyleProject {

    Script.JarFile jarFile;
    Script.DirEntry dir;
    String name;

    MavenStyleProject(Script.JarFile jarFile, Script.DirEntry dir, String name) {
        this.jarFile = jarFile;
        this.dir = dir;
        this.name = name;
    }

    static MavenStyleProject example(Script.BuildDir buildDir, Script.DirEntry dir) {
        return new MavenStyleProject(buildDir.jarFile("hat-example-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName());
    }
    static MavenStyleProject javaBackend(Script.BuildDir buildDir, Script.DirEntry dir) {
        return new MavenStyleProject(buildDir.jarFile("hat-backend-java-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName());
    }
    static MavenStyleProject ffiBackend(Script.BuildDir buildDir, Script.DirEntry dir) {
        return new MavenStyleProject(buildDir.jarFile("hat-backend-ffi-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName());
    }


}

void main(String[] args) {
    var layout = """     
       └──./
           ├──hat                                      //  All build scripts in each case 'foo' has java options for (and points to) 'foo.java'
           │    ├──bld                                 //  --enable-preview --source 24 hat/bld.java
           │    ├──bld.java
           │    ├──run                                 //  --enable-preview --source 24 hat/run.java
           │    ├──run.java
           │    └──Script                              //  Contains all the tools for building
           │
           ├──build/                                   // All jars, native libs and executables
           │    └──cmake-build-debug/                  // All intermediate cmake artifacts
           │        ├── hat-*wrap-1.0.jar              // Wrapper jars around extracted * (clwrap, glwrap, cuwrap)
           │        ├── hat-core-1.0.jar               // Base hat jar
           │        ├── hat-example-*-1.0.jar          // Example jars (hat-example-nbody-1.0.jar, hat-example-life-1.0.jar)
           │        ├── hat-jextracted-opencl-1.0.jar  // Raw jextracted jars (hat-jextracted-opencl-1.0.jar ....)
           │        ├── lib*_backend.[dylib|so]        // ffi library backends
           │        └── *(no suffix)                   // various generated executables (opencl_info, cuda_info, cuda_squares)
           ├──stage/
           │   ├── repo/
           │   │   └── *                               // Maven artifacts (poms and jars)
           │   ├── opencl_jextracted/                  // All jextracted files (created using java @hat/extract
           │   │   ├── compile_flags.txt
           │   │   └── opencl
           │   ├── cuda_jextracted/
           │   │   ├── compile_flags.txt
           │   │   └── cuda
           │   └── opengl_jextracted/
           │       ├── compile_flags.txt
           │       └── opengl
           ├──wrap/
           │    └──wrap/
           │         ├──wrap/                          // Maven style layout
           │         ├──clwrap/                        // Maven style layout
           │         ├──glwrap/                        // Maven style layout
           │         └──cuwrap/                        // Maven style layout
           │
           ├──hat-core                                 // Maven style layout
           │    ├──src/main/java
           │    │    └──hat/
           │    │
           │    └──src/main/test
           │         └──hat/
           │
           ├──backends
           │    ├──java
           │    │    ├──mt                             // Maven style layout
           │    │    └──seq                            // Maven style layout
           │    ├──jextracted
           │    │    └──opencl                         // Maven style layout
           │    └──ffi
           │         ├──opencl                         // Maven style layout with cmake
           │         ├──ptx                            // Maven style layout with cmake
           │         ├──mock                           // Maven style layout with cmake
           │         ├──spirv                          // Maven style layout with cmake
           │         ├──cuda                           // Maven style layout with cmake
           │         └──hip                            // Maven style layout with cmake
           │
           └──examples
                ├──mandel                              // Maven style layout
                ├──squares                             // Maven style layout
                ├──heal                                // Maven style layout
                ├──life                                // Maven style layout
                ├──nbody                               // Maven style layout
                ├──experiments                         // Maven style layout
                └──violajones                          // Maven style layout
       """;

    var dir = Script.DirEntry.current();
    var buildDir = Script.BuildDir.of(dir.path("build")).create();
    var cmakeCapability = Script.Capabilities.CMake.required();
    Script.Capabilities capabilities = Script.Capabilities.of( cmakeCapability);
    cmakeCapability.probe(buildDir, capabilities);
    out.println(capabilities.tickOrCheck());

    var hatJavacOpts = Script.javacBuilder($ -> $
            .enable_preview()
            .add_modules("jdk.incubator.code")
            .current_source()
    );

    var hatJarOptions = Script.jarBuilder($ -> $
            .verbose(false)
    );

    var hatCoreDir = dir.existingDir("hat-core");
    var hatCoreJar = buildDir.jarFile("hat-core-1.0.jar");

    var cmakeBuildDir = buildDir.cMakeBuildDir("cmake-build-debug");

    Script.jar(hatJarOptions, jar -> jar
            .jarFile(hatCoreJar)
            .maven_style_root(hatCoreDir)
            .javac(hatJavacOpts, javac -> {})
    );

    var wrapsDir = dir.existingDir("wrap");

    var wrapDir = wrapsDir.existingDir("wrap");
    var wrapJar = buildDir.jarFile("hat-wrap-1.0.jar");

    Script.jar(jar -> jar
            .jarFile(wrapJar)
            .maven_style_root(wrapDir)
            .javac(javac -> javac.current_source())
    );


    var clWrapJar = buildDir.jarFile("hat-clwrap-1.0.jar");
    var clwrapDir = wrapsDir.dir("clwrap");
    var extractedOpenCLJar = buildDir.jarFile("hat-jextracted-opencl-1.0.jar");

    if (clwrapDir.exists() && extractedOpenCLJar.exists()) {
        Script.jar(jar -> jar
                        .jarFile(clWrapJar)
                        .maven_style_root(clwrapDir)
                        .javac(javac ->
                                javac.current_source()
                                        .class_path(wrapJar, hatCoreJar, extractedOpenCLJar))
                );
    }else{
        println("Skipping OPENCL and dependencies ");
    }

    var glWrapJar = buildDir.jarFile("hat-glwrap-1.0.jar");
    var glwrapDir = wrapsDir.dir("glwrap");
    var extractedOpenGLJar = buildDir.jarFile("hat-jextracted-opengl-1.0.jar");
    if (glwrapDir.exists() && extractedOpenGLJar.exists()) {
        Script.jar(jar -> jar
                .jarFile(glWrapJar)
                .maven_style_root(glwrapDir)
                .javac(javac -> javac
                        .current_source()
                        .exclude(javaSrc -> javaSrc.matches("^.*/wrap/glwrap/GLCallbackEventHandler\\.java$"))
                        //.exclude(javaSrc -> javaSrc.matches("^.*/wrap/glwrap/GLFuncEventHandler\\.java$"))
                        .class_path(wrapJar,extractedOpenGLJar)
                )
        );
    }else{
        println("Skipping OPENGL and dependencies ");
    }


    var cuWrapJar = buildDir.jarFile("hat-cuwrap-1.0.jar");
    var cuwrapDir = wrapsDir.dir("cuwrap");
    var extractedCudaJar = buildDir.jarFile("hat-jextracted-cuda-1.0.jar");
    if (cuwrapDir.exists() && extractedCudaJar.exists()) {

    }else{
        println("Skipping CUDA and dependencies");
    }


    var backends = dir.existingDir("backends");

    // Here we create all ffi-backend jars.
    var ffiBackends = backends.existingDir("ffi");
    var ffiBackendSharedJar = buildDir.jarFile("hat-backend-ffi-shared-1.0.jar");
    ffiBackends.subDirs()
            .filter(backend -> backend.matches("^.*(shared|mock|opencl)$"))
            .sorted(new Comparator<Script.DirEntry>() {
                @Override
                public int compare(Script.DirEntry lhs, Script.DirEntry rhs) {
                    return rhs.fileName().compareTo(lhs.fileName()); // reverse sorted we want shared first ;)
                }
            })
            .map(javaBackend->MavenStyleProject.ffiBackend(buildDir,javaBackend))
            .forEach(msp -> {
                Script.jar(hatJarOptions, jar -> jar
                        .jarFile(msp.jarFile)
                        .maven_style_root(msp.dir)
                        .javac(hatJavacOpts, javac ->
                                javac.either(msp.name.equals("shared"),
                                        _-> javac.class_path(hatCoreJar),
                                        _-> javac.class_path(hatCoreJar, ffiBackendSharedJar)
                                )
                        )
                );
                out.println(msp.jarFile.fileName()+ " OK");
            });

    var jextractedBackends = backends.existingDir("jextracted");
    var jextractedBackendSharedDir = jextractedBackends.dir("shared");
    var jextractedSharedBackendJar = buildDir.jarFile("hat-backend-jextracted-shared-1.0.jar");

    Script.jar(hatJarOptions, jar -> jar
            .jarFile(jextractedSharedBackendJar)
            .maven_style_root(jextractedBackendSharedDir)
            .javac(hatJavacOpts, javac -> javac.class_path(hatCoreJar))
    );
    out.println(jextractedSharedBackendJar.fileName()+ " OK");

    var jextractedBackendOpenCLDir = jextractedBackends.dir("opencl");
    if ( jextractedBackendOpenCLDir.exists() && extractedOpenCLJar.exists()) {
        var jextractedOpenCLBackendJar = buildDir.jarFile("hat-backend-jextracted-opencl-1.0.jar");
        Script.jar(hatJarOptions, jar -> jar
                .jarFile(jextractedOpenCLBackendJar)
                .maven_style_root(jextractedBackendOpenCLDir)
                .javac(hatJavacOpts, javac -> javac
                        .class_path(hatCoreJar, extractedOpenCLJar, jextractedSharedBackendJar)
                )
        );
        out.println(jextractedOpenCLBackendJar.fileName()+ " OK");
    }


    // Here we create all java backend jars.
    backends.existingDir("java")
            .subDirs()
            .filter(backend -> backend.matches("^.*(mt|seq)$"))
            .map(javaBackend->MavenStyleProject.javaBackend(buildDir,javaBackend))
            .forEach(msp -> {
                Script.jar(hatJarOptions, jar -> jar
                        .maven_style_root(msp.dir)
                        .jarFile(msp.jarFile)
                        .javac(javac->javac.class_path(hatCoreJar))
                );
                out.println(msp.jarFile.fileName()+" OK");
            });

    var examplesDir = dir.existingDir("examples");

    examplesDir
            .subDirs()
            .filter(exampleDir -> exampleDir.failsToMatch("^.*(experiments|target|.idea)$"))
            .map(exampleDir -> MavenStyleProject.example(buildDir, exampleDir))
            .forEach(msp -> {
                out.println(msp.jarFile.fileName());
                Script.jar(hatJarOptions, jar -> jar
                        .jarFile(msp.jarFile)
                        .maven_style_root(msp.dir)
                        .javac(hatJavacOpts, javac ->
                                javac.class_path(hatCoreJar)
                                        .when(msp.dir.matches("^.*(nbody)$")
                                                && extractedOpenCLJar.exists() && clWrapJar.exists()
                                                && extractedOpenGLJar.exists() && glWrapJar.exists()
                                                , _ ->
                                                javac.class_path(wrapJar,
                                                        clWrapJar, extractedOpenCLJar,
                                                        buildDir.jarFile("hat-backend-ffi-opencl-1.0.jar"),
                                                        glWrapJar, extractedOpenGLJar

                                                )
                                        )
                        )
                        .manifest(manifest -> manifest.main_class(msp.name + ".Main"))
                );
            });


    if (cmakeCapability.available()) {
        if (!cmakeBuildDir.exists()) {
            Script.cmake($ -> $
                    .verbose(false)
                    .source_dir(ffiBackends)
                    .build_dir(cmakeBuildDir)
                    .copy_to(buildDir)
            );
        }
        Script.cmake($ -> $
                .build(cmakeBuildDir)
        );
    } else {
        out.println("No cmake available so we did not build ffi backend shared libs");
    }

}

