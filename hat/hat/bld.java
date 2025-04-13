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


import static java.io.IO.println;
import static java.lang.System.out;


static class MavenStyleProject implements Script.ClassPathEntryProvider {
    final Script.JarFile jarFile;
    final Script.DirEntry dir;
    final String name;

    final boolean hasJavaSources;

    final List<Script.ClassPathEntryProvider> classPath = new ArrayList<>();
    final List<Script.ClassPathEntryProvider> failedDependencies = new ArrayList<>();
    MavenStyleProject(Script.JarFile jarFile, Script.DirEntry dir, String name,  Script.ClassPathEntryProvider ...classPathEntryProviders) {
        this.jarFile = jarFile;
        this.dir = dir;
        this.name = name;

        this.classPath.addAll(List.of(classPathEntryProviders ));

        for (Script.ClassPathEntryProvider classPathEntryProvider : classPathEntryProviders) {
            classPathEntryProvider.classPathEntries().forEach(classPathEntry -> {
                if (!classPathEntry.exists()){
                    failedDependencies.add(classPathEntry);
                }
            });
        }
        this.hasJavaSources = dir.sourceDir("src/main/java").javaFiles().findAny().isPresent();
      //  println(name+" failedDependencies.isEmpty()="+failedDependencies.isEmpty()+ " hasJavaSources="+hasJavaSources);
    }

    static MavenStyleProject example(Script.BuildDir buildDir, Script.DirEntry dir,  Script.ClassPathEntryProvider ...classPathEntryProviders) {
        return new MavenStyleProject(buildDir.jarFile("hat-example-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName(), classPathEntryProviders);
    }
    static MavenStyleProject javaBackend(Script.BuildDir buildDir, Script.DirEntry dir,  Script.ClassPathEntryProvider ...classPathEntryProviders) {
        return new MavenStyleProject(buildDir.jarFile("hat-backend-java-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName(),classPathEntryProviders);
    }
    static MavenStyleProject ffiBackend(Script.BuildDir buildDir, Script.DirEntry dir,  Script.ClassPathEntryProvider ...classPathEntryProviders) {
        return new MavenStyleProject(buildDir.jarFile("hat-backend-ffi-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName(),classPathEntryProviders);
    }
    static MavenStyleProject jextractedBackend(Script.BuildDir buildDir, Script.DirEntry dir,  Script.ClassPathEntryProvider ...classPathEntryProviders) {
        return new MavenStyleProject(buildDir.jarFile("hat-backend-jextracted-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName(),classPathEntryProviders);
    }

    static MavenStyleProject of(  Script.DirEntry dir, Script.JarFile jarFile, Script.ClassPathEntryProvider ...classPathEntryProviders) {
        return new MavenStyleProject(jarFile, dir, dir.fileName(),classPathEntryProviders);
    }

    boolean canWeBuild(){
        if (hasJavaSources && failedDependencies.isEmpty()) {
           return true;
        }else if (!hasJavaSources) {
            println("Skipping " + jarFile.fileName() + " no java sources");
        }else  if (!failedDependencies.isEmpty()){
            print("Skipping "+jarFile.fileName()+" failed dependencies ");
            for (Script.ClassPathEntryProvider classPathEntryProvider : failedDependencies) {
                classPathEntryProvider.classPathEntries().forEach(classPathEntry ->
                   print(classPathEntry.fileName())
                );
            }
            println("");
        }else{
            println("Skipping "+jarFile.fileName()+"why!");
        }
        return false;
    }

    public MavenStyleProject build() {
        if (canWeBuild()) {
            Script.jar(jar -> jar
                    .verbose(false)
                    .jarFile(jarFile)
                    .maven_style_root(dir)
                    .javac(javac -> javac
                            .enable_preview()
                            .add_modules("jdk.incubator.code")
                            .current_source()
                            .class_path(classPath)
                    )
            );
            println(jarFile.fileName() + " OK");
        }
        return this;
    }



    @Override
    public List<Script.ClassPathEntry> classPathEntries() {
        return List.of(jarFile);
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

    var hatCore = MavenStyleProject.of(
            dir.existingDir("hat-core"),
            buildDir.jarFile("hat-core-1.0.jar")
    ).build();

    var wrapsDir = dir.existingDir("wrap");

    var wrap = MavenStyleProject.of(wrapsDir.existingDir("wrap"), buildDir.jarFile("hat-wrap-1.0.jar")
    ).build();


    var extractedOpenCLJar = buildDir.jarFile("hat-jextracted-opencl-1.0.jar");

    var clWrap = MavenStyleProject.of(wrapsDir.dir("clwrap"), buildDir.jarFile("hat-clwrap-1.0.jar"),
            wrap, hatCore, extractedOpenCLJar
    ).build();

    var extractedOpenGLJar = buildDir.jarFile("hat-jextracted-opengl-1.0.jar");
    var glWrap = MavenStyleProject.of(wrapsDir.dir("glwrap"), buildDir.jarFile("hat-glwrap-1.0.jar"),
            wrap, hatCore, extractedOpenGLJar
    ); // we can't use build, we need a custom build
    if (glWrap.canWeBuild()) {
        Script.jar(jar -> jar
                .jarFile(glWrap.jarFile)
                .maven_style_root(glWrap.dir)
                .javac(javac -> javac
                        .current_source()
                        .exclude(javaSrc -> javaSrc.matches("^.*/wrap/glwrap/GLCallbackEventHandler\\.java$"))
                        //.exclude(javaSrc -> javaSrc.matches("^.*/wrap/glwrap/GLFuncEventHandler\\.java$"))
                        .class_path(wrap, extractedOpenGLJar)
                )
        );
        println(glWrap.jarFile.fileName()+" OK");
    }

    var extractedCudaJar = buildDir.jarFile("hat-jextracted-cuda-1.0.jar");
    var cuWrap = MavenStyleProject.of(wrapsDir.dir("cuwrap"), buildDir.jarFile("hat-cuwrap-1.0.jar"),
            extractedCudaJar
    ).build();


    var backendsDir = dir.existingDir("backends");

    var ffiBackendsDir = backendsDir.existingDir("ffi");

    var ffiBackendShared = MavenStyleProject.of(ffiBackendsDir.existingDir("shared"),
            buildDir.jarFile("hat-backend-ffi-shared-1.0.jar"),
            hatCore
    ).build();

    ffiBackendsDir.subDirs()
            .filter(ffiBackend -> ffiBackend.matches("^.*(mock|opencl)$"))
            .map(ffiBackend->MavenStyleProject.ffiBackend(buildDir,ffiBackend,hatCore,ffiBackendShared))
            .forEach(MavenStyleProject::build);

    var jextractedBackends = backendsDir.existingDir("jextracted");
    var jextractedBackendShared = MavenStyleProject.jextractedBackend(buildDir,
            jextractedBackends.existingDir("shared"),
            hatCore
    ).build();

    var jextractedBackendOpenCL = MavenStyleProject.jextractedBackend(buildDir,
            jextractedBackends.dir("opencl"),
            buildDir.jarFile("hat-backend-jextracted-opencl-1.0.jar"),
            hatCore, extractedOpenCLJar, jextractedBackendShared
            ).build();

    var javaBackendsDir = backendsDir.existingDir("java");
    javaBackendsDir
            .subDirs()
            .filter(backend -> backend.matches("^.*(mt|seq)$"))
            .map(javaBackend->MavenStyleProject.javaBackend(buildDir,javaBackend,hatCore))
            .forEach(MavenStyleProject::build);

    var examplesDir = dir.existingDir("examples");
    examplesDir
            .subDirs()
            .filter(exampleDir -> exampleDir.failsToMatch("^.*(experiments|nbody|target|.idea)$"))
            .map(exampleDir ->  MavenStyleProject.example(buildDir, exampleDir, hatCore))
            .forEach(MavenStyleProject::build);

    var ffiOpenCLBackendJar = buildDir.jarFile("hat-backend-ffi-opencl-1.0.jar");
    examplesDir
            .subDirs()
            .filter(exampleDir -> exampleDir.matches("^.*(nbody)$"))
            .map(exampleDir ->  MavenStyleProject.example(buildDir,exampleDir,
                            hatCore,
                            wrap,
                            clWrap, extractedOpenCLJar,
                            ffiOpenCLBackendJar,
                            glWrap, extractedOpenGLJar
                            )
            )
            .forEach(MavenStyleProject::build);

    if (cmakeCapability.available()) {
        var cmakeBuildDir = buildDir.cMakeBuildDir("cmake-build-debug");
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
    } else {
        out.println("No cmake available so we did not build ffi backend shared libs");
    }

}

