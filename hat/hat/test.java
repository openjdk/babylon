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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class Config {
    boolean headless = false;
    boolean noModuleOp = false;
    boolean verbose = false;
    boolean startOnFirstThread = false;
    boolean justShowCommandline = false;
    boolean runSuite = false;
    String backendName = null;
    Script.JarFile backendJar = null;
    String testName = null;
    String examplePackageName = null;
    String testMainClassName = null;
    Script.JarFile exampleJar = null;
    List<Script.ClassPathEntryProvider> classpath = new ArrayList<>();
    List<String> vmargs = new ArrayList<>();
    List<String> appargs = new ArrayList<>();

    Config(Script.BuildDir buildDir, String[] args) {

        var testJARFile = buildDir.jarFile("hat-tests-1.0.jar");
        if (testJARFile.exists()) {
            testMainClassName = "oracle.code.hat.engine.HatTestEngine";
            exampleJar = testJARFile;
            if (exampleJar.exists()) {
                classpath.add(exampleJar);
            } else {
                System.err.println("Cannot find example jar at " + testMainClassName);
            }
        }

        classpath.add(buildDir.jarFile("hat-tests-1.0.jar"));
        classpath.add(buildDir.jarFile("hat-core-1.0.jar"));
        classpath.add(buildDir.jarFile("hat-example-shared-1.0.jar"));

        for (int arg = 0; arg < args.length; arg++) {

            if (args[arg].equals("suite")) {
                runSuite = true;
            } else if (args[arg].startsWith("ffi-")) {
                backendName = args[arg];
                backendJar = buildDir.jarFile("hat-backend-" + backendName + "-1.0.jar");
                classpath.add(buildDir.jarFile("hat-backend-ffi-shared-1.0.jar"));
                classpath.add(backendJar);
            } else if (args[arg].startsWith("java-")) {
                backendName = args[arg];
                backendJar = buildDir.jarFile("hat-backend-" + backendName + "-1.0.jar");
                classpath.add(backendJar);
            } else {
                switch (args[arg]) {
                    case "headless" -> headless = true;
                    case "noModuleOp" -> noModuleOp = true;
                    case "verbose" -> verbose = true;
                    case "justShowCommandLine" -> justShowCommandline = true;
                    case "startOnFirstThread" -> startOnFirstThread = true;
                    default -> {
                        this.appargs.add(args[arg]);
                    }
                }
            }
        }
    }
}

class Stats {
    int passed = 0;
    int failed = 0;
    public void incrementPassed(int val) {
        passed += val;
    }
    public void incrementFailed(int fail) {
        failed += fail;
    }

    public int getPassed() {
        return passed;
    }
    public int getFailed() {
        return failed;
    }

    @Override
    public String toString() {
        return String.format("Global passed: %d, failed: %d, pass-rate: %.2f%%", passed, failed, ((float)(passed * 100 / (passed + failed))));
    }
}

void main(String[] argv) {
    var usage = """
              usage:
                java @hat/test [backend] class
                    backend   : opencl|cuda

                examples:
                   java @hat/test suite ffi-opencl
                   java @hat/test ffi-opencl class#method
                   java @hat/test suite ffi-cuda
                   java @hat/test ffi-cuda class#method
            """;

    var hatDir = Script.DirEntry.current();
    var buildDir = hatDir.existingBuildDir("build");

    Config config = new Config(buildDir, argv);

    if (config.classpath.isEmpty()) {
        println("Classpath is empty!");
    } else if (config.backendJar == null || !config.backendJar.exists()) {
        println("No backend !");
    } else if (!config.exampleJar.exists()) {
        println("No example !");
    } else {
        var extraction_opencl_jar = buildDir.jarFile("hat-extraction-opencl-1.0.jar");
        var extraction_opengl_jar = buildDir.jarFile("hat-extraction-opengl-1.0.jar");
        var wrap_shared_jar = buildDir.jarFile("hat-wrap-shared-1.0.jar");
        var wrap_opencl_jar = buildDir.jarFile("hat-wrap-opencl-1.0.jar");
        var wrap_opengl_jar = buildDir.jarFile("hat-wrap-opengl-1.0.jar");
        switch (config.backendName) {
            default -> {
            }
        }
        if (config.noModuleOp) {
            System.out.println("NOT using ModuleOp for CallGraphs");
        }
    }

    // Remove the previous report file:
    Path file = Paths.get("test_report.txt");
    try {
        Files.deleteIfExists(file);
    } catch (IOException e) {
        e.printStackTrace();
    }

    if (config.runSuite) {

        String[] suite = new String[] {
                "oracle.code.hat.TestArrays",
                "oracle.code.hat.TestMatMul",
                "oracle.code.hat.TestMandel",
                "oracle.code.hat.TestLocal",
                "oracle.code.hat.TestReductions",
                "oracle.code.hat.TestPrivate"
        };

        // Test the whole suite
        for (String testClass : suite) {
            Script.java(java -> java
                    .enable_preview()
                    .verbose(true)
                    .enable_native_access("ALL-UNNAMED")
                    .library_path(buildDir)
                    .when(config.headless, Script.JavaBuilder::headless)
                    .when(config.noModuleOp, Script.JavaBuilder::noModuleOp)
                    .when(config.startOnFirstThread, Script.JavaBuilder::start_on_first_thread)
                    .class_path(config.classpath)
                    .vmargs(config.vmargs)
                    .main_class(config.testMainClassName)
                    .args(testClass)
                    .justShowCommandline(config.justShowCommandline));
        }

        // Final report
        String regex = "passed: (\\d+), failed: (\\d+)";
        Pattern pattern = Pattern.compile(regex);
        Stats stats = new Stats();

        System.out.println("\n\n************************************************");
        System.out.println("                 HAT Test Report ");
        System.out.println("************************************************");
        try {
            List<String> lines = Files.readAllLines(file);
            for (String line : lines) {
                System.out.println(line);

                Matcher matcher = pattern.matcher(line);
                if (matcher.find()) {
                    int passed = Integer.parseInt(matcher.group(1));
                    int fail = Integer.parseInt(matcher.group(2));
                    stats.incrementPassed(passed);
                    stats.incrementFailed(fail);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(stats);

    } else {
        // A single command for a specific class/method
        Script.java(java -> java
                .enable_preview()
                .verbose(true)
                .enable_native_access("ALL-UNNAMED")
                .library_path(buildDir)
                .when(config.headless, Script.JavaBuilder::headless)
                .when(config.noModuleOp, Script.JavaBuilder::noModuleOp)
                .when(config.startOnFirstThread, Script.JavaBuilder::start_on_first_thread)
                .class_path(config.classpath)
                .vmargs(config.vmargs)
                .main_class(config.testMainClassName)
                .args(config.appargs)
                .justShowCommandline(config.justShowCommandline));
    }

}
