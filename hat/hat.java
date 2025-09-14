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

static String logo = """
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
        """;
static String help = """
        Usage  bld|clean|run ...
             bld:
                   Compile all buildable (based on capabilities) available jars and native code.

             dot:
                   Create dot graph  (bld.dot) of buildable dependencies (based on capabilities)
                      dot bld.dot -Tsvg > bld.svg  && chrome bld.svg
           clean:
                   Removes build directory entirely
                   conf dir and jextracted artifacts (opencl/cuda/opengl) remain


             run:  [ffi|my|seq]-[opencl|java|cuda|mock|hip] runnable  args
                      run ffi-opencl mandel
                      run ffi-opencl nbody 4096
                      run ffi-opencl heal 4096

             exp:  [ffi|my|seq]-[opencl|java|cuda|mock|hip] experimentClassName  args
                      exp ffi-opencl QuotedConstantArgs

             test:  [ffi|my|seq]-[opencl|java|cuda|mock|hip]
                      test ffi-opencl

          sanity:  Check source files for copyright and WS issues (tabs and trailing EOL WS)
        """;


public static void main(String[] argArr) throws IOException, InterruptedException {
    var args = new ArrayList<>(List.of(argArr));
    if (args.isEmpty()) {
        System.out.println(help);
    } else {
        Path hatDir = Path.of(System.getProperty("user.dir"));
        var project = new Project(hatDir, Reporter.progressAndErrors);
        var cmake = project.isAvailable("cmake", "--version");
        if (!cmake.isAvailable()) {
            System.out.println("We need cmake, to check the availability of opencl, cuda etc so we wont be able to build much  ");
        }
        var jextract = project.isAvailable("jextract", "--version");
        if (!jextract.isAvailable()) {
            System.out.println("We will need jextract to create jextracted backends and for examples requiring opengl ");
        }

        // A user defined optional dependency.  Anything depending on this will only build if it is true
        // In our case we pull the value from the headless system property
        var ui = new Opt(project.id("ui"), !Boolean.getBoolean("headless"));

        // These dependencies are only 'true' on the appropriate platform.
        // So any target that depends on one of these, will only build on that platform
        var mac = new Mac(project.id("os-mac"));
        var linux = new Linux(project.id("os-linux"));
        // var windows = new Windows(project.id("os-windows")); maybe one day

        // These next three 'optional' dependencies use cmake to determine availability.  We delegate to cmake which
        //    a) determines if capability is available,
        //    b) if they are, they extract from cmake vars (see conf/cmake-info/OpenCL/properties for example) information export headers and libs needed by JExtract
        var openclCmakeInfo = new OpenCL(project.id("cmake-info-opencl"), cmake);
        var openglCmakeInfo = new OpenGL(project.id("cmake-info-opengl"), cmake);
        var cudaCmakeInfo = new Cuda(project.id("cmake-info-cuda"), cmake);

        // Now we just create jars and shared libs and declare dependencies
        var core = Jar.of(project.id("core"));
        var tools = Jar.of(project.id("tools"), core);
        var tests = Jar.of(project.id("tests"), core, tools);
        var backend_ffi_native = CMake.of(project.id("backend{s}-ffi"), core, cmake);
        var ffiSharedBackend = Jar.of(project.id("backend{s}-ffi-shared"), backend_ffi_native);
        var backend_ffi_cuda = Jar.of(project.id("backend{s}-ffi-cuda"), ffiSharedBackend);
        var backend_ffi_opencl = Jar.of(project.id("backend{s}-ffi-opencl"), ffiSharedBackend);
        var backend_ffi_mock = Jar.of(project.id("backend{s}-ffi-mock"), ffiSharedBackend);
        var backend_mt_java = Jar.of(project.id("backend{s}-java-mt"), core);
        var backend_seq_java = Jar.of(project.id("backend{s}-java-seq"), core);
        var example_shared = Jar.of(project.id("example{s}-shared"), ui, core);
        var example_blackscholes = Jar.of(project.id("example{s}-blackscholes"), example_shared);
        var example_mandel = Jar.of(project.id("example{s}-mandel"), example_shared);
        var example_life = Jar.of(project.id("example{s}-life"), example_shared);
        var example_squares = Jar.of(project.id("example{s}-squares"), core);
        var example_matmul = Jar.of(project.id("example{s}-matmul"), core);
        var example_heal = Jar.of(project.id("example{s}-heal"), example_shared);
        var example_violajones = Jar.of(project.id("example{s}-violajones"), example_shared);
        var example_experiments = Jar.of(project.id("example{s}-experiments"), backend_ffi_opencl); //experiments have some code that expect opencl backend

        var wrapped_shared = Jar.of(project.id("wrap{s}-shared"));
        var jextracted_opencl = JExtract.extract(project.id("extract{ions|ed}-opencl"), jextract, openclCmakeInfo, core);
        var wrapped_jextracted_opencl = Jar.of(project.id("wrap{s}-opencl"), jextracted_opencl, wrapped_shared);

        var jextracted_opengl = JExtract.extract(project.id("extract{ions|ed}-opengl"), jextract, ui, openglCmakeInfo, core);

        // Sigh... We have different src exclusions for wrapped opengl depending on the OS
        var excludedOpenGLWrapSrc = project.rootPath().resolve(
                "wraps/opengl/src/main/java/wrap/opengl/GL" + (mac.isAvailable() ? "Callback" : "Func") + "EventHandler.java");

        var wrapped_jextracted_opengl = Jar.of(project.id("wrap{s}-opengl"), Set.of(excludedOpenGLWrapSrc), jextracted_opengl, wrapped_shared);

        var example_nbody = Jar.of(project.id("example{s}-nbody"), ui, wrapped_jextracted_opengl, wrapped_jextracted_opencl);

        while (!args.isEmpty()) {
            var arg = args.removeFirst();
            switch (arg) {
                case "help" -> System.out.println(logo + "\n" + help);
                case "clean" -> project.clean();
                case "dot" -> {
                    Dag dag = project.all();
                    Dag available = dag.available();
                    Files.writeString(Path.of("bld.dot") , available.toDot());
                    System.out.println("Consider...\n    dot bld.dot -Tsvg > bld.svg");
                }
                case "bld" -> {
                    Dag dag = project.all();
                    Dag available = dag.available();
                    project.build(available);
                }
                case "sanity" -> {
                    final  Pattern copyrightPattern = Pattern.compile("^.*Copyright.*202[0-9].*(Intel|Oracle).*$");
                    final  Pattern copyrightExemptPattern = Pattern.compile("^(robertograham|CMakeFiles|hip)");
                    final  Pattern tabOrEolWsPattern = Pattern.compile("^(.*\\t.*|.* )$");
                    final  Pattern textSuffix  = Pattern.compile("^(.*\\.(java|cpp|h|hpp|md)|pom.xml)$");
                    final  Pattern sourceSuffix  = Pattern.compile("^(.*\\.(java|cpp|h|hpp)|pom.xml)$");

                    Stream.of("core","tools","examples","backends","docs","wraps")
                            .map(hatDir::resolve)
                            .forEach(dir-> {
                                Util.recurse(dir,
                                        (d)-> true, // we do this foir all subdirs
                                        (f)-> textSuffix.matcher(f.getFileName().toString()).matches() && Util.grepLines(tabOrEolWsPattern, f),
                                        (c)-> System.out.println("File contains WS issue (TAB or EOLWs) " + c)
                                );
                                Util.recurse(dir,
                                        (d)-> !copyrightExemptPattern.matcher(d.getFileName().toString()).matches(), // we skip these subdirs
                                        (f)-> sourceSuffix.matcher(f.getFileName().toString()).matches() && !Util.grepLines(copyrightPattern, f),
                                        (c)-> System.out.println("File does not contain copyright " + c)
                                );
                            });
                    args.clear();
                }

                case "run" -> {
                    if (args.size() > 1) {
                        String backendName = args.removeFirst();
                        String runnableName = args.removeFirst();
                        if (project.get(backendName) instanceof Jar backend) {
                            if (project.get(runnableName) instanceof Jar runnable) {
                                List<String> vmOpts = new ArrayList<>();
                                //vmOpts.add("-DnoModuleOp=true");
                                //vmOpts.add("-DbufferTracking=true");
                                var dag = new job.Dag(runnable, backend);
                                if (runnableName.equals("nbody") && mac.isAvailable()) {  // nbody (or anything using OpenGL on mac) needs this
                                    vmOpts.add("-XstartOnFirstThread");
                                }
                                runnable.run(runnableName + ".Main", dag.ordered(), vmOpts,args);
                            } else {
                                System.err.println("Failed to find runnable " + runnableName);
                            }
                        } else {
                            System.err.println("Failed to find " + backendName);
                        }
                    } else {
                        System.err.println("For run we expect 'run backend runnable' ");
                    }
                    args.clear(); //!! :)
                }
                case "test" -> {
                    if (args.size() > 0) {
                        String backendName = args.removeFirst();
                        if (project.get(backendName) instanceof Jar backend) {
                           Path file = Paths.get("test_report.txt");
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
                                   return String.format("Global passed: %d, failed: %d, pass-rate: %.2f%%",
                                       passed, failed, ((float)(passed * 100 / (passed + failed))));
                               }
                           }

                           try {
                              Files.deleteIfExists(file);
                           } catch (IOException e) {
                              e.printStackTrace();
                           }
                           var suite = new String[] {
                               "oracle.code.hat.TestArrays",
                               "oracle.code.hat.TestMatMul",
                               "oracle.code.hat.TestMandel",
                               "oracle.code.hat.TestLocal",
                               "oracle.code.hat.TestReductions"
                           };
                           for(var s:suite){
                              List<String> vmOpts = new ArrayList<>();
                              var dag = new job.Dag(tests, backend);
                              args.add(s);
                              tests.run("oracle.code.hat.engine.HatTestEngine", dag.ordered(), vmOpts,args);
                              args.remove(args.size()-1);
                           }
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
                           System.err.println("Failed to find backend   " + backendName);
                        }
                    } else {
                        System.err.println("For test we require a backend ");
                    }
                    args.clear(); //!! :)
                }
                case "exp" -> {
                    if (args.size() > 1) {
                        String backendName = args.removeFirst();
                        String runnableName = "experiments";
                        String className = args.removeFirst();
                        if (project.get(backendName) instanceof Jar backend) {
                            if (project.get(runnableName) instanceof Jar runnable) {
                                List<String> vmOpts = new ArrayList<>();
                                //vmOpts.add("-DnoModuleOp=true");
                                //vmOpts.add("-DbufferTracking=true");
                                runnable.run(runnableName + "."+className, new job.Dag(runnable, backend).ordered(), vmOpts,args);
                            } else {
                                System.err.println("Failed to find runnable " + runnableName);
                            }
                        } else {
                            System.err.println("Failed to find " + backendName);
                        }
                    } else {
                        System.err.println("For run we expect 'run backend runnable' ");
                    }
                    args.clear(); //!! :)
                }
                default -> {
                    System.out.println("'" + arg + "' was unexpected ");
                    System.out.println(help);
                    args.clear();
                }
            }
        }
    }
}
