package bldr;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.regex.Matcher;

import static bldr.Bldr.assertExists;
import static java.io.IO.println;

public class CMakeProbe implements Capabilities.Probe {
    public interface CMakeVar<T> {
        String name();

        T value();
    }

    public record CMakeTypedVar(String name, String type, String value, String comment)
            implements CMakeVar<String> {
        static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+):([^=]*)=(.*)$");

        CMakeTypedVar(Matcher matcher, String comment) {
            this(
                    "CMAKE_" + matcher.group(1).trim(),
                    matcher.group(2).trim(),
                    matcher.group(3).trim(),
                    comment.substring(2).trim());
        }

        static boolean onMatch(String line, String comment, Consumer<CMakeTypedVar> consumer) {
            return regex.matches(line, matcher -> consumer.accept(new CMakeTypedVar(matcher, comment)));
        }
    }

    public record CMakeSimpleVar(String name, String value) implements CMakeVar {
        static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{(.*)\\}>\\}$");

        CMakeSimpleVar(Matcher matcher) {
            this(
                    "CMAKE_" + matcher.group(1).trim(),
                    (matcher.group(2).isEmpty()) ? "" : matcher.group(2).trim());
        }

        static boolean onMatch(String line, String comment, Consumer<CMakeSimpleVar> consumer) {
            return regex.matches(line, matcher -> consumer.accept(new CMakeSimpleVar(matcher)));
        }
    }

    public record CMakeDirVar(String name, Bldr.DirPathHolder value) implements CMakeVar {
        static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{(.*)\\}>\\}$");

        static boolean onMatch(String line, String comment, Consumer<CMakeSimpleVar> consumer) {
            return regex.matches(line, matcher -> consumer.accept(new CMakeSimpleVar(matcher)));
        }
    }

    public record CMakeContentVar(String name, String value) implements CMakeVar {
        static final Regex startRegex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{(.*)$");
        static final Regex endRegex = Regex.of("^(.*)\\}>\\}$");
    }

    public record CMakeRecipeVar(String name, String value) implements CMakeVar<String> {
        static final Regex varPattern = Regex.of("<([^>]*)>");
        static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{<(.*)>\\}>\\}$");

        CMakeRecipeVar(Matcher matcher) {
            this(
                    "CMAKE_" + matcher.group(1).trim(),
                    "<" + ((matcher.group(2).isEmpty()) ? "" : matcher.group(2).trim()) + ">");
        }

        public String expandRecursively(Map<String, CMakeVar<?>> varMap, String value) { // recurse
            String result = value;
            if (varPattern.pattern().matcher(value) instanceof Matcher matcher && matcher.find()) {
                var v = matcher.group(1);
                if (varMap.containsKey(v)) {
                    String replacement = varMap.get(v).value().toString();
                    result =
                            expandRecursively(
                                    varMap,
                                    value.substring(0, matcher.start())
                                            + replacement
                                            + value.substring(matcher.end()));
                }
            }
            return result;
        }

        public String expand(Map<String, CMakeVar<?>> vars) {
            return expandRecursively(vars, value());
        }

        static boolean onMatch(String line, String comment, Consumer<CMakeRecipeVar> consumer) {
            return regex.matches(line, matcher -> consumer.accept(new CMakeRecipeVar(matcher)));
        }
    }

    Bldr.BuildDir dir;

    Map<String, CMakeVar<?>> varMap = new HashMap<>();

    public CMakeProbe(Bldr.BuildDir dir, Capabilities capabilities) {
        this.dir = Bldr.BuildDir.of(dir.path("cmakeprobe"));
        this.dir.clean();

        try {
            this.dir.cmakeLists($-> {$
                    .append(
                         """
                         cmake_minimum_required(VERSION 3.21)
                         project(cmakeprobe)
                         set(CMAKE_CXX_STANDARD 14)
                         foreach(VarName ${VarNames})
                            message("${VarName}={<{${${VarName}}}>}")
                         endforeach()
                         """);
                        capabilities
                                .capabilities()
                                .filter(capability -> capability instanceof Capabilities.CMakeCapability)
                                .map(capability -> (Capabilities.CMakeCapability) capability)
                                .forEach(p -> $.append("find_package(").append(p.name).append(")\n")
                        );

                        //println("content = {"+$+"}");
                    });

            var cmakeProcessBuilder =
                    new ProcessBuilder()
                            .directory(this.dir.path().toFile())
                            .redirectErrorStream(true)
                            .command("cmake", "-LAH")
                            .start();
            List<String> lines =
                    new BufferedReader(new InputStreamReader(cmakeProcessBuilder.getInputStream()))
                            .lines()
                            .toList();

            String comment = null;
            String contentName = null;
            StringBuilder content = null;

            for (String line : lines) {

             //   frameworkMap.values().forEach(framework ->
               //     framework.regex.matches(line,
                //            m->println(line)
                 //   )
                //);
                if (line.startsWith("//")) {
                    comment = line;
                    content = null;

                } else if (comment != null) {
                    if (CMakeTypedVar.onMatch(
                            line,
                            comment,
                            v -> {
                                if (varMap.containsKey(v.name())) {
                                    var theVar = varMap.get(v.name());
                                    if (theVar.value().equals(v.value())) {
                                      /*  println(
                                                "replacing duplicate variable with typed variant with the name same value"
                                                        + v
                                                        + theVar);*/
                                    } else {
                                        throw new IllegalStateException(
                                                "Duplicate variable name different value: " + v + theVar);
                                    }
                                    varMap.put(v.name(), v);
                                } else {
                                    varMap.put(v.name(), v);
                                }
                            })) {
                    } else {
                        println("failed to parse " + line);
                    }
                    comment = null;
                    content = null;
                    contentName = null;
                } else if (!line.isEmpty()) {
                    if (content != null) {
                        if (CMakeContentVar.endRegex.pattern().matcher(line) instanceof Matcher matcher
                                && matcher.matches()) {
                            content.append("\n").append(matcher.group(1));
                            var v = new CMakeContentVar(contentName, content.toString());
                            contentName = null;
                            content = null;
                            varMap.put(v.name(), v);
                        } else {
                            content.append("\n").append(line);
                        }
                    } else if (!line.endsWith("}>}")
                            && CMakeContentVar.startRegex.pattern().matcher(line) instanceof Matcher matcher
                            && matcher.matches()) {
                        contentName = "CMAKE_" + matcher.group(1);
                        content = new StringBuilder(matcher.group(2));
                    } else if (CMakeRecipeVar.regex.pattern().matcher(line) instanceof Matcher matcher
                            && matcher.matches()) {
                        CMakeVar<String> v = new CMakeRecipeVar(matcher);
                        if (varMap.containsKey(v.name())) {
                            var theVar = varMap.get(v.name());
                            if (theVar.value().equals(v.value())) {
                              //  println("Skipping duplicate variable name different value: " + v + theVar);
                            } else {
                                throw new IllegalStateException(
                                        "Duplicate variable name different value: " + v + theVar);
                            }
                            varMap.put(v.name(), v);
                        } else {
                            varMap.put(v.name(), v);
                        }
                    } else if (CMakeSimpleVar.regex.pattern().matcher(line) instanceof Matcher matcher
                            && matcher.matches()) {
                        var v =  new CMakeSimpleVar(matcher);
                        if (varMap.containsKey(v.name())) {
                            var theVar = varMap.get(v.name());
                            if (theVar.value().equals(v.value())) {
                               // println("Skipping duplicate variable name different value: " + v + theVar);
                            } else {
                                //throw new IllegalStateException(
                                  //      "Duplicate variable name different vars: " + v + theVar);
                            }
                            // note we don't replace a Typed with a Simple
                        } else {
                            varMap.put(v.name(), v);
                        }
                    } else {
                       // println("Skipping " + line);
                    }
                }
            }

        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }

        capabilities
                .capabilities()
                .filter(capability -> capability instanceof Capabilities.CMakeCapability)
                .map(capability->(Capabilities.CMakeCapability)capability)
                .forEach(capability -> capability.setCmakeProbe(this));

    }

    Bldr.ObjectFile cxxCompileObject(
            Bldr.ObjectFile target, Bldr.CppSourceFile source, List<String> frameworks) {
        CMakeRecipeVar compileObject = (CMakeRecipeVar) varMap.get("CMAKE_CXX_COMPILE_OBJECT");
        Map<String, CMakeVar<?>> localVars = new HashMap<>(varMap);
        localVars.put("DEFINES", new CMakeSimpleVar("DEFINES", ""));
        localVars.put("INCLUDES", new CMakeSimpleVar("INCLUDES", ""));
        localVars.put("FLAGS", new CMakeSimpleVar("FLAGS", ""));
        localVars.put("OBJECT", new CMakeSimpleVar("OBJECT", target.path().toString()));
        localVars.put("SOURCE", new CMakeSimpleVar("SOURCE", source.path().toString()));
        String executable = compileObject.expand(localVars);
        println(executable);
        return target;
    }

    Bldr.ExecutableFile cxxLinkExecutable(
            Bldr.ExecutableFile target, List<Bldr.ObjectFile> objFiles, List<String> frameworks) {
        CMakeRecipeVar linkExecutable = (CMakeRecipeVar) varMap.get("CMAKE_CXX_LINK_EXECUTABLE");
        Map<String, CMakeVar<?>> localVars = new HashMap<>(varMap);
        String executable = linkExecutable.expand(localVars);
        println(executable);
        return target;
    }

    Bldr.SharedLibraryFile cxxCreateSharedLibrary(
            Bldr.SharedLibraryFile target, List<Bldr.ObjectFile> objFiles, List<String> frameworks) {
        CMakeRecipeVar createSharedLibrary =
                (CMakeRecipeVar) varMap.get("CMAKE_CXX_CREATE_SHARED_LIBRARY");
        Map<String, CMakeVar<?>> localVars = new HashMap<>(varMap);
        String executable = createSharedLibrary.expand(localVars);
        println(executable);
        return target;
    }


    public String value(String key) {
        var  v = varMap.get(key);
        return v.value().toString();
    }

    public  boolean hasKey(String includeDirKey) {
        return varMap.containsKey(includeDirKey);
    }

    public static void main(String[] args) {
        var hatDir = assertExists(Bldr.Dir.of("/Users/grfrost/github/babylon-grfrost-fork/hat"));
        var buildDir = hatDir.buildDir("build");
        var backends = hatDir.dir("backends");
        var backend = backends.dir("opencl");
        var cppDir = backend.dir("cpp");
        var opencl = Capabilities.OpenCL.of();
        var opengl = Capabilities.OpenGL.of();
        var cuda =  Capabilities.CUDA.of();
        var hip =  Capabilities.HIP.of();
        Capabilities capabilities = Capabilities.of(opencl, opengl, cuda, hip);
        var cmake = new CMakeProbe(buildDir,capabilities);
        var clinfoObj =
                cmake.cxxCompileObject(
                        buildDir.objectFile("clinfo.cpp.o"),
                        cppDir.cppSourceFile("clinfo.cpp"),
                        List.of("OpenCL"));
        var clinfo =
                cmake.cxxLinkExecutable(
                        buildDir.executableFile("clinfo"), List.of(clinfoObj), List.of("OpenCL"));
    }
}
