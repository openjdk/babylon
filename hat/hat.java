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

import job.*;

static void logo(){
    System.out.println("""
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣤⣀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀ ⠙⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⣷⣶⣤⣄⣈⣉⣉⣉⣉⣉⣉⣉⣁⣤⡄⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀ ⠀⣿⣿⣿⣿⣿ HAT ⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⢀⣠⣶⣾⡏⢀⡈⠛⠻⠿⢿⣿⣿⣿⣿⣿⠿⠿⠟⠛⢁⠀⢶⣤⣀⠀⠀⠀
        ⠀⢠⣿⣿⣿⣿⡇⠸⣿⣿⣶⣶⣤⣤⣤⣤⣤⣤⣤⣶⣶⣿⡿⠂⣸⣿⣿⣷⡄⠀
        ⠀⢸⣿⣿⣿⣿⣿⣦⣄⡉⠛⠛⠛⠿⠿⠿⠿⠛⠛⠛⢉⣁⣤⣾⣿⣿⣿⣿⡷⠀
        ⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀
        ⠀⠀⠀⠀⠈⠙⠛⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠿⠛⠛⠉⠁⠀⠀⠀⠀
        """);
}
static void help(){
    System.out.println("""
        Usage  bld|clean|run ...
             bld:
                   Compile all buildable (based on capabilities) available jars and native code.

             dot:
                   Create dot graph  (bld.dot) of buildable dependencies (based on capabilities)
                      dot bld.dot -Tsvg > bld.svg  && chrome bld.svg
           clean:
                   Removes build directory entirely
                   conf dir and jextracted artifacts (opencl/cuda/opengl) remain


             run:  [ffi|my|seq]-[opencl|java|cuda|mock|hip] [-DXXX ...] runnable  args
                      run ffi-opencl mandel
                      run ffi-opencl nbody 4096
                      run ffi-opencl -DHAT=SHOW_CODE nbody 4096
                      run ffi-opencl -DHAT=SHOW_KERNEL_MODEL heal
                      run ffi-opencl -DHAT=MINIMIZE_BUFFERS life

             exp:  [ffi|my|seq]-[opencl|java|cuda|mock|hip] [-DXXX ... ] experimentClassName  args
                      exp ffi-opencl QuotedConstantArgs

             test-suite:  [ffi|my|seq]-[opencl|java|cuda|mock|hip]
                      test-suite ffi-opencl

             test:  [ffi|my|seq]-[opencl|java|cuda|mock|hip] classToTest (also classToTest#method)
                      test ffi-opencl  hat.test.TestMatMul

          sanity:  Check source files for copyright and WS issues (tabs and trailing EOL WS)
        """);
}

static  void logoAndHelp(){
    logo();
    help();
}

public static void main(String[] argArr) throws IOException, InterruptedException {
    var args = new ArrayList<>(List.of(argArr));
    if (args.isEmpty()) {
        help();
    } else {
        var hat = new Project(
                Util.currentDirAsPath(),
                // These opts will be applied to all javac, cmake and jextract tools
                Jar.JavacConfig.of(o -> o.progress(true)
                        .debug().enablePreview().source(26).addModules("jdk.incubator.code")
                ),
                CMake.Config.of(o -> o.progress(true)),
                JExtract.Config.of(o -> o.progress(true))
        );

        var commonJavaOpts = Jar.JavaConfig.of(o -> o
                .verbose(true)
                .command(true)
                .enablePreview()
                .addModules("jdk.incubator.code")
                .enableNativeAccess("ALL-UNNAMED"));

        var cmake = hat.isAvailable("cmake", "--version");
        if (!cmake.isAvailable()) {
            System.out.println("We need cmake, to check the availability of opencl, cuda etc so we wont be able to build much  ");
        }
        var jextract = hat.isAvailable("jextract", "--version");
        if (!jextract.isAvailable()) {
            System.out.println("We will need jextract to create jextracted backends and for examples requiring opengl ");
        }

        // This is an example of a user defined optional dependency.
        // Anything depending on this will only build if (In this case) the property is true
        // So in our case if the headless system property
        var ui = new Opt(hat.id("ui"), !Boolean.getBoolean("headless")); //-Dheadless=true|false

        // These dependencies are  'true' on the appropriate platform.
        // So any target depending on one these, will only build on that platform
        var mac = new Mac(hat.id("os-mac"));
        var linux = new Linux(hat.id("os-linux"));

        // These next three 'optional' dependencies use cmake to determine availability.  We delegate to cmake which
        //    a) determines if capability is available,
        //    b) if they are, they extract from cmake vars (see conf/cmake-info/OpenCL/properties for example) information export headers and libs needed by JExtract
        var jextractOpts = JExtract.Config.of(o -> o.command(true));
        var cmakeOpts = CMake.Config.of(o -> o.command(true));
        var openclCmakeInfo = new OpenCL(hat.id("cmake-info-opencl"), cmake);
        var openglCmakeInfo = new OpenGL(hat.id("cmake-info-opengl"), cmake);
        var cudaCmakeInfo = new Cuda(hat.id("cmake-info-cuda"), cmake);

        // Now we just create jars and shared libs and declare dependencies
        var optkl = hat.jar("optkl");
        var core = hat.jar("core", optkl);
        var tools = hat.jar("tools", core);
        var tests = hat.jar("tests", core, tools);

        var backend_ffi_native = hat.cmakeAndJar("backend{s}-ffi", core, cmake);
        var ffiSharedBackend = hat.jar("backend{s}-ffi-shared", backend_ffi_native);
        var backend_ffi_cuda = hat.jar("backend{s}-ffi-cuda", ffiSharedBackend);
        var backend_ffi_opencl = hat.jar("backend{s}-ffi-opencl", ffiSharedBackend);
        var backend_ffi_mock = hat.jar("backend{s}-ffi-mock", ffiSharedBackend);

        // These examples just rely on core
        var backend_mt_java = hat.jar("backend{s}-java-mt", core);
        var backend_seq_java = hat.jar("backend{s}-java-seq", core);
        var example_squares = hat.jar("example{s}-squares", core);
        var example_matmul = hat.jar("example{s}-matmul", core);
        var example_blackscholes = hat.jar("example{s}-blackscholes", core);
        var example_view = hat.jar("example{s}-view", core);
        var example_normmap = hat.jar("example{s}-normmap", core); // will probabvly need shared when we hatify

        // example_shared allows us to break out common UI functions, views, even loops etc
        var example_shared = hat.jar("example{s}-shared", ui, core);

        // These examples use example_shared, so they are UI based
        var example_mandel = hat.jar("example{s}-mandel", example_shared);
        var example_life = hat.jar("example{s}-life", example_shared);
        var example_heal = hat.jar("example{s}-heal", example_shared);
        var example_violajones = hat.jar("example{s}-violajones", example_shared);

        // experiments include code that expects an opencl backend, this is not idea, but we can accomodate
        var example_experiments = hat.jar("example{s}-experiments", core);

        // Now we have the more complex nonsense for nbody (which needs opengl and opencl extracted)
        var wrapped_shared = hat.jar("wrap{s}-shared");
        var jextracted_opencl = hat.jextract("extract{ions|ed}-opencl", jextract, openclCmakeInfo, core);
        var wrapped_jextracted_opencl = hat.jar("wrap{s}-opencl", jextracted_opencl, wrapped_shared);
        var backend_jextracted_shared = hat.jar("backend{s}-jextracted-shared", core);
        var backend_jextracted_opencl = hat.jar("backend{s}-jextracted-opencl", wrapped_jextracted_opencl, backend_jextracted_shared);
        var jextracted_opengl = hat.jextract("extract{ions|ed}-opengl", jextract, ui, openglCmakeInfo, core);

        // Sigh... We have different src exclusions for wrapped opengl depending on the OS
        var excludedOpenGLWrapSrc = hat.rootPath().resolve(
                "wraps/opengl/src/main/java/wrap/opengl/GL" + (mac.isAvailable() ? "Callback" : "Func") + "EventHandler.java");

        var wrapped_jextracted_opengl = hat.jar("wrap{s}-opengl", Set.of(excludedOpenGLWrapSrc), jextracted_opengl, wrapped_shared);

        // Finally we have everything needed for nbody
        var example_nbody = hat.jar("example{s}-nbody", ui, wrapped_jextracted_opengl, wrapped_jextracted_opencl);

        var testEnginePackage = "hat.test.engine";
        var testEngineClassName = "HATTestEngine";
        while (!args.isEmpty()) {
            var arg = args.removeFirst();
            switch (arg) {
                case "help" -> logoAndHelp();
                case "clean" -> hat.clean(true);
                case "dot" -> {
                    Files.writeString(Path.of("bld.dot"), hat.all().available().toDot());
                    System.out.println("Consider...\n    dot bld.dot -Tsvg > bld.svg");
                }
                case "bld" -> {
                    hat.build(hat.all().available());
                }
                case "sanity" -> {
                    final var copyrightPattern = Pattern.compile("^.*Copyright.*202[0-9].*(Intel|Oracle).*$");
                    final var copyrightExemptPattern = Pattern.compile("^(robertograham|CMakeFiles|hip)");
                    final var tabOrEolWsPattern = Pattern.compile("^(.*\\t.*|.* )$");
                    final var textSuffix = Pattern.compile("^(.*\\.(java|cpp|h|hpp|md)|pom.xml)$");
                    final var sourceSuffix = Pattern.compile("^(.*\\.(java|cpp|h|hpp)|pom.xml)$");

                    Stream.of("hat", "tests", "optkl", "core", "tools", "examples", "backends", "docs", "wraps")
                            .map(hat.rootPath()::resolve)
                            .forEach(dir -> {
                                System.out.println("Checking " + dir);
                                Util.recurse(dir,
                                        (d) -> true, // we do this for all subdirs
                                        (f) -> textSuffix.matcher(f.getFileName().toString()).matches() && Util.grepLines(tabOrEolWsPattern, f),
                                        (c) -> System.out.println("File contains WS issue (TAB or EOLWs) " + c)
                                );
                                Util.recurse(dir,
                                        (d) -> !copyrightExemptPattern.matcher(d.getFileName().toString()).matches(), // we skip these subdirs
                                        (f) -> sourceSuffix.matcher(f.getFileName().toString()).matches() && !Util.grepLines(copyrightPattern, f),
                                        (c) -> System.out.println("File does not contain copyright " + c)
                                );
                            });
                    args.clear();
                }
                case "run" -> {
                    if (args.size() > 1) {
                        var backendName = args.removeFirst();
                        if (hat.get(backendName) instanceof Jar backend) {
                            var javaOpts = commonJavaOpts.with(o -> o
                                    .collectVmOpts(args).mainClass(args.removeFirst(), "Main")
                                    .startOnFirstThreadIf(o.packageName().equals("nbody") && mac.isAvailable()).collectArgs(args)
                            );
                            if (hat.get(javaOpts.packageName()) instanceof Jar runnable) {
                                runnable.run(javaOpts, runnable, backend);
                            } else {
                                System.err.println("Found backend " + backendName + " but failed to find runnable/example " + javaOpts.packageName());
                            }
                        } else {
                            System.err.println("Failed to find backend " + backendName);
                        }
                    } else {
                        System.err.println("For run we expect 'run backend runnable' ");
                    }
                }
                case "test-suite" -> {
                    if (!args.isEmpty()) {
                        var backendName = args.removeFirst();
                        if (hat.get(backendName) instanceof Jar backend) {
                            System.out.println("""
                                    *****************************************************************
                                    HAT Test Report
                                    *****************************************************************
                                    """);
                            var test_reports_txt = Paths.get("test_report.txt");
                            Files.deleteIfExists(test_reports_txt); // because we will append to it in the next loop
                            var commonTestSuiteJavaOpts = commonJavaOpts.with(o -> o
                                    .command(false).collectVmOpts(args).mainClass(testEnginePackage, testEngineClassName) //  note no app args as add them below
                            );

                            tests.forEachMatchingEntry("(hat/test/Test[a-zA-Z0-9]*).class", (_, matcher) ->
                                    tests.run(Jar.JavaConfig.of(commonTestSuiteJavaOpts, o -> o.arg(matcher.group(1).replace('/', '.'))), tests, backend)
                            );
                            args.clear();
                            var pattern = Pattern.compile("passed: (\\d+), failed: (\\d+), unsupported: (\\d+), precision-errors: (\\d+)");
                            class Stats {
                                int passed = 0;
                                int failed = 0;
                                int unsupported = 0;
                                int precisionError = 0;

                                @Override
                                public String toString() {
                                    return String.format("Global passed: %d, failed: %d, unsupported: %d, precision-errors: %d, pass-rate: %.2f%%\\n",
                                            passed, failed, unsupported, precisionError, ((float) (passed * 100 / (passed + failed + unsupported + precisionError))));
                                }
                            }
                            var stats = new Stats();
                            Files.readAllLines(test_reports_txt).forEach(line -> {
                                if (!commonTestSuiteJavaOpts.verbose()) { //We already dumped this info to stdout above
                                    System.out.println(line);
                                }
                                if (pattern.matcher(line) instanceof Matcher matcher && matcher.find()) {
                                    stats.passed += Integer.parseInt(matcher.group(1));
                                    stats.failed += Integer.parseInt(matcher.group(2));
                                    stats.unsupported += Integer.parseInt(matcher.group(3));
                                    stats.precisionError += Integer.parseInt(matcher.group(4));
                                }
                            });
                            System.out.println(stats);
                            if (stats.failed > 0) {
                                System.exit(-1);
                            } else {
                                System.exit(0);
                            }
                        } else {
                            System.err.println("Failed to find backend   " + backendName);
                        }
                    } else {
                        System.err.println("For test-suite we require a backend ");
                    }
                }
                case "test" -> {
                    if (args.size() >= 2) {
                        var backendName = args.removeFirst();
                        if (hat.get(backendName) instanceof Jar backend) {
                            var javaOpts = commonJavaOpts.with(o -> o
                                    .verbose(true).command(true).collectVmOpts(args).mainClass(testEnginePackage, testEngineClassName).collectArgs(args)
                            );
                            tests.run(javaOpts, tests, backend);
                        } else {
                            System.err.println("Failed to find backend   " + backendName);
                        }
                    } else {
                        System.err.println("""
                                For test we require a backend and a TestClass.");
                                Examples:
                                    $ test ffi-opencl hat.test.TestMatMul
                                    $ test ffi-opencl hat.test.TestMatMul#method
                                """);
                    }
                    args.clear(); //!! :)
                }
                case "exp" -> {
                    if (args.size() > 1) {
                        var backendName = args.removeFirst();
                        if (hat.get(backendName) instanceof Jar backend) {
                            var javaOpts = Jar.JavaConfig.of(commonJavaOpts, o -> o
                                    .collectVmOpts(args).mainClass("experiments", args.removeFirst()).collectArgs(args)
                            );
                            if (hat.get(javaOpts.packageName()) instanceof Jar runnable) {
                                runnable.run(javaOpts, runnable, backend);
                            } else {
                                System.err.println("Failed to find runnable " + javaOpts.mainClassName());
                            }
                        } else {
                            System.err.println("Failed to find " + backendName);
                        }
                    } else {
                        System.err.println("For exp we expect 'exp backend [-DXXX ...] testclass' ");
                    }
                    args.clear(); //!! :)
                }
                default -> {
                    System.err.println("'" + arg + "' was unexpected ");
                    help();
                    args.clear();
                }
            }
        }
    }
}
