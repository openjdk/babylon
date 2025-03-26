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

static class Targets {
    final Script.BuildDir buildDir;
    final public Script.JarFile wrapJar;
    final Script.JarFile clWrapJar;
    final public Script.JarFile glWrapJar;
    final public Script.JarFile cuWrapJar;
    final public Script.JarFile hatJar;
    final public Script.CMakeBuildDir cmakeBuildDir;

    Targets(Script.BuildDir buildDir) {
        this.buildDir = buildDir;
        this.wrapJar = buildDir.jarFile("hat-wrap-1.0.jar");
        this.clWrapJar = buildDir.jarFile("hat-clwrap-1.0.jar");
        this.glWrapJar = buildDir.jarFile("hat-glwrap-1.0.jar");
        this.cuWrapJar = buildDir.jarFile("hat-cuwrap-1.0.jar");
        this.hatJar = buildDir.jarFile("hat-core-1.0.jar");

        cmakeBuildDir = buildDir.cMakeBuildDir("cmake-build-debug");
    }
}

static class Example {

    Script.JarFile jarFile;
    Script.DirEntry dir;
    String name;

    Example(Script.JarFile jarFile, Script.DirEntry dir, String name) {
        this.jarFile = jarFile;
        this.dir = dir;
        this.name = name;
    }

    static Example of(Script.BuildDir buildDir, Script.DirEntry dir) {
        return new Example(buildDir.jarFile("hat-example-" + dir.fileName() + "-1.0.jar"), dir, dir.fileName());
    }
}

void main(String[] args) {

    /*
     *  ./
     *    +--build/                     All jars, native libs and executables
     *    |    +--cmake-build-debug/    All intermediate cmake artifacts
     *    |
     *    +--stage/
     *    |    +--repo/                 All downloaded maven assets (if any)
     *    |    |
     *    |    +--jextract/             All jextracted files
     *    |    |    +--opencl
     *    |    |    +--opengl
     *    |    |    +--cuda
     *    |
     *    +--wrap/
     *    |    +--wrap/                 All downloaded maven assets
     *    |    |    +--wrap/                (*)
     *    |    |    +--clwrap/              (*)
     *    |    |    +--glwrap/              (*)
     *    |    |    +--cuwrap/              (*)
     *    |    |
     *    |
     *    +--extractions/
     *    |    +--opencl/
     *    |    +--opengl/
     *    |    +--cuda/
     *    |
     *    +--hat-core                        * Note maven style layout
     *    |    +--src/main/java
     *    |    |    +--hat/
     *    |    |
     *    |    +--src/main/test
     *    |         +--hat/
     *    |
     *    +--backends
     *    |    +--java
     *    |    |    +--mt                    (*)
     *    |    |    +--seq                   (*)
     *    |    +--jextracted
     *    |    |    +--opencl                (*)
     *    |    +--ffi
     *    |         +--opencl                (*)
     *    |         +--ptx                   (*)
     *    |         +--mock                  (*)
     *    |         +--spirv                 (*)
     *    |         +--cuda                  (*)
     *    |         +--hip                   (*)
     *    |
     *    +--examples
     *    |    +--mandel                     (*)
     *    |    +--squares                    (*)
     *    |    +--heal                       (*)
     *    |    +--life                       (*)
     *    |    +--nbody                      (*)
     *    |    +--experiments                (*)
     *    |    +--violajones                 (*)
     *
     */

    var dir = Script.DirEntry.current();
    var hatCoreDir = dir.existingDir("hat-core");
    var backends = dir.existingDir("backends");
    var examplesDir = dir.existingDir("examples");
    var wrapsDir = dir.existingDir("wrap");
    var stageDir = dir.buildDir("stage").create();

    var buildDir = Script.BuildDir.of(dir.path("build")).create();
    var targets = new Targets(buildDir);


    var openclCapability = Script.Capabilities.OpenCL.of();
    var openglCapability = Script.Capabilities.OpenGL.of();
    var cudaCapability = Script.Capabilities.CUDA.of();
    var hipCapability = Script.Capabilities.HIP.of();
    var jextractCapability = Script.Capabilities.JExtract.of();// or Capability.JExtract.of(Path.of("/my/jextract-22/bin/jextract"));
    var cmakeCapability = Script.Capabilities.CMake.of();

    Script.Capabilities capabilities = Script.Capabilities.of(openclCapability, openglCapability, cudaCapability, hipCapability, jextractCapability, cmakeCapability);

    if (cmakeCapability.available()) {
        cmakeCapability.probe(buildDir, capabilities);
    }

    capabilities.capabilities().forEach(fw -> out.print("[" + fw.name + (fw.available() ? "\u2714" : "\u2715") + "]"));
    out.println();

    var verbose = false;


    var hatJavacOpts = Script.javacBuilder($ -> $
            .enable_preview()
            .add_modules("jdk.incubator.code")
            .current_source()
    );

    var hatJarOptions = Script.jarBuilder($ -> $
            .verbose(verbose)
    );
    Script.jar(hatJarOptions, jar -> jar
            .jarFile(targets.hatJar)
            .maven_style_root(hatCoreDir)
            .javac(hatJavacOpts, javac -> {
            })
    );

    Script.jar(jar -> jar
            .jarFile(targets.wrapJar)
            .maven_style_root(wrapsDir.dir("wrap"))
            .javac(javac -> javac.current_source())
    );

    if (jextractCapability.available()) {
        if (openclCapability.available()) {
            if (!openclCapability.jarFile(buildDir).exists()) {
                if (!openclCapability.stage(stageDir).exists()) {
                    Script.jextract(jextractCapability.executable, $ -> $.verbose(verbose).capability(openclCapability, stageDir));
                } else {
                    out.println("Using previously extracted  " + openclCapability.stage(buildDir).fileName());
                }
                Script.jar(jar -> jar
                        .jarFile(openclCapability.jarFile(buildDir))
                        .javac(javac -> javac.current_source().source_path(Script.SourceDir.of(openclCapability.stage(stageDir).path())))

                );
            } else {
                out.println("Using existing extracted " + openclCapability.jarFile(buildDir).fileName());
            }
            Script.jar(jar -> jar
                    .jarFile(targets.clWrapJar)
                    .maven_style_root(wrapsDir.dir("clwrap"))
                    .javac(javac -> javac.current_source().class_path(targets.wrapJar, targets.hatJar, openclCapability.jarFile(buildDir)))
            );
        } else {
            out.println("This platform does not have OpenCL");
        }

        if (openglCapability.available()) {
            if (!openglCapability.jarFile(buildDir).exists()) {
                if (!openglCapability.stage(stageDir).exists()) {
                    Script.jextract(jextractCapability, $ -> $.verbose(verbose).capability(openglCapability, stageDir));
                } else {
                    out.println("Using previously extracted  " + openglCapability.stage(buildDir).fileName());
                }
                Script.jar(jar -> jar
                        .jarFile(openglCapability.jarFile(buildDir))
                        .javac(javac -> javac.current_source().source_path(Script.SourceDir.of(openglCapability.stage(stageDir).path())))
                );
            } else {
                out.println("Using existing extracted " + openglCapability.jarFile(buildDir).fileName());
            }
            Script.jar(jar -> jar
                    .jarFile(targets.glWrapJar)
                    .maven_style_root(wrapsDir.dir("glwrap"))
                    .javac(javac -> javac
                            .current_source()
                            .exclude(javaSrc -> javaSrc.matches("^.*/wrap/glwrap/GLCallbackEventHandler\\.java$"))
                            //.exclude(javaSrc -> javaSrc.matches("^.*/wrap/glwrap/GLFuncEventHandler\\.java$"))
                            .class_path(targets.wrapJar, openglCapability.jarFile(buildDir))
                    )
            );
        } else {
            out.println("This platform does not have OpenGL");
        }


        if (cudaCapability.available()) {

        } else {
            out.println("This platform does not have CUDA");
        }
    }

    var backendJars = new ArrayList<Script.JarFile>();


    // Here we create all ffi-backend jars.
    var ffiBackends = backends.existingDir("ffi");
    var ffiBackendSharedDir = ffiBackends.dir("shared");
    out.println("Shared ffi " + ffiBackendSharedDir.path());
    var ffiSharedBackendJar = buildDir.jarFile("hat-backend-ffi-shared-1.0.jar");
    backendJars.add(ffiSharedBackendJar);
    var ffiBackendSharedResult = Script.jar(hatJarOptions, jar -> jar
            .jarFile(ffiSharedBackendJar)
            .maven_style_root(ffiBackendSharedDir)
            .javac(hatJavacOpts, javac -> javac.verbose(verbose)
                    .class_path(targets.hatJar)
            )
    );

    ffiBackends.subDirs()
            .filter(backend -> backend.failsToMatch("^.*(spirv|hip|shared|target|.idea)$"))
            .forEach(backend -> {
                var ffiBackendJarFile = buildDir.jarFile("hat-backend-ffi-" + backend.fileName() + "-1.0.jar");
                backendJars.add(ffiBackendJarFile);
                out.println(ffiBackendJarFile.fileName());
                Script.jar(hatJarOptions, jar -> jar
                        .jarFile(ffiBackendJarFile)
                        .maven_style_root(backend)
                        .javac(hatJavacOpts, javac -> javac.class_path(targets.hatJar, ffiSharedBackendJar))
                );
            });

    // Here we create jextracted-backend jars.
    var jextractedBackends = backends.existingDir("jextracted");
    var jextractedBackendSharedDir = jextractedBackends.dir("shared");
    out.println("Shared jextracted " + jextractedBackendSharedDir.path());
    var jextractedSharedBackendJar = buildDir.jarFile("hat-backend-jextracted-shared-1.0.jar");
    backendJars.add(jextractedSharedBackendJar);
    var jextractedBackendSharedResult = Script.jar(hatJarOptions, jar -> jar
            .jarFile(jextractedSharedBackendJar)
            .maven_style_root(jextractedBackendSharedDir)
            .javac(hatJavacOpts, javac -> javac.verbose(verbose)
                    .class_path(targets.hatJar)
            )
    );

    if (openclCapability.available()) {
        var jextractedBackendOpenCLDir = jextractedBackends.dir("opencl");
        out.println("OpenCL jextracted " + jextractedBackendOpenCLDir.path());
        var jextractedOpenCLBackendJar = buildDir.jarFile("hat-backend-jextracted-opencl-1.0.jar");
        backendJars.add(jextractedOpenCLBackendJar);
        Script.jar(hatJarOptions, jar -> jar
                .jarFile(jextractedOpenCLBackendJar)
                .maven_style_root(jextractedBackendOpenCLDir)
                .javac(hatJavacOpts, javac -> javac.verbose(verbose)
                        .class_path(targets.hatJar, openclCapability.jarFile(buildDir), jextractedBackendSharedResult)
                )
        );
    }


    // Here we create all java backend jars.
    backends.existingDir("java")
            .subDirs()
            .filter(backend -> backend.failsToMatch("^.*(target|.idea)$"))
            .forEach(backend -> {
                var backendJavaJar = buildDir.jarFile("hat-backend-java-" + backend.fileName() + "-1.0.jar");
                out.println(backendJavaJar.fileName());
                backendJars.add(backendJavaJar);
                Script.jar(hatJarOptions, jar -> jar
                        .jarFile(backendJavaJar)
                        .dir_list(backend.dir("src/main/resources"))
                );
            });

    backendJars.forEach(j -> out.println(" backend " + j.path()));

    examplesDir.subDirs()
            .filter(exampleDir -> exampleDir.failsToMatch("^.*(experiments|target|.idea)$"))
            .map(exampleDir -> Example.of(targets.buildDir, exampleDir))
            .forEach(example -> {
                out.println(example.jarFile.fileName());
                Script.jar(hatJarOptions, jar -> jar
                        .jarFile(example.jarFile)
                        .maven_style_root(example.dir)
                        .javac(hatJavacOpts, javac ->
                                javac.class_path(targets.hatJar)
                                        .when(example.dir.matches("^.*(life|nbody)$") && jextractCapability.available() && openclCapability.available(), _ ->
                                                javac.class_path(targets.wrapJar,
                                                        targets.clWrapJar,
                                                        openclCapability.jarFile(buildDir),
                                                        buildDir.jarFile("hat-backend-ffi-opencl-1.0.jar")
                                                )
                                        )
                                        .when(example.dir.matches("^.*(nbody)$") && jextractCapability.available() && openclCapability.available() && openglCapability.available(), _ ->
                                                javac.class_path(targets.wrapJar,
                                                        targets.glWrapJar,
                                                        openglCapability.jarFile(buildDir)

                                                )
                                        )
                        )
                        .manifest(manifest -> manifest.main_class(example.name + ".Main"))
                );
            });


    if (cmakeCapability.available()) {
        if (!targets.cmakeBuildDir.exists()) {
            Script.cmake($ -> $
                    .verbose(verbose)
                    .source_dir(ffiBackends)
                    .build_dir(targets.cmakeBuildDir)
                    .copy_to(buildDir)
            );
        }
        Script.cmake($ -> $
                .build(targets.cmakeBuildDir)
        );
    } else {
        out.println("No cmake available so we did not build ffi backend shared libs");
    }

}

