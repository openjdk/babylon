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
package job;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public abstract class CMakeInfo extends CMake implements Dependency.Optional, JExtractOptProvider {

    Path asPath(String key) {
        return properties.containsKey(key) ? Path.of((String) properties.get(key)) : null;
    }

    boolean asBoolean(String key) {
        return properties.containsKey(key) && Boolean.parseBoolean((String) properties.get(key));
    }

    String asString(String key) {
        return (properties.containsKey(key) && properties.get(key) instanceof String s) ? s : null;
    }
    List<String> asSemiSeparatedStringList(String key) {
        return Arrays.stream(asString(key).split(";")).toList();
    }

    final String find;
    final String response;
    final static String template = """
            cmake_minimum_required(VERSION 3.22.1)
            project(extractions)
            find_package(__find__)
            get_cmake_property(_variableNames VARIABLES)
            foreach (_variableName ${_variableNames})
               message(STATUS "${_variableName}=${${_variableName}}")
            endforeach()
            """;

    final String text;

    final Set<String> vars;
    Properties properties = new Properties();
    final Path propertiesPath;
    final String sysName;
    final String fwk;
    final boolean darwin;
    final boolean linux;
    final Map<String, String> otherVarMap = new LinkedHashMap<>();
    final boolean available;

    CMakeInfo(Project.Id id, String find, String response, Set<String> varsIn, Set<Dependency> buildDependencies) {
        super(id, id.project().confPath().resolve("cmake-info").resolve(find), buildDependencies);
        this.find = find;
        this.response = response;

        this.vars = new LinkedHashSet<>(Set.of(
                "CMAKE_HOST_SYSTEM_NAME",
                "CMAKE_HOST_SYSTEM_PROCESSOR",
                "CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES",
                response
        ));
        this.vars.addAll(varsIn);

        this.text = template.replaceAll("__find__", find).replaceAll("__response__", response);
        this.propertiesPath = cmakeSourceDir().resolve("properties");
        if (Files.exists(propertiesPath)) {
            properties = new Properties();
            try {
                properties.load(Files.newInputStream(propertiesPath));
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        } else {
            id.project().mkdir(cmakeBuildDir());
            try {
                Files.writeString(CMakeLists_txt, this.text, StandardCharsets.UTF_8, StandardOpenOption.CREATE);
                Pattern p = Pattern.compile("-- *([A-Za-z_0-9]+)=(.*)");
                cmakeInit((line) -> {
                    if (p.matcher(line) instanceof Matcher matcher && matcher.matches()) {
                        //   System.out.println("GOT "+matcher.group(1)+"->"+matcher.group(2));
                        if (this.vars.contains(matcher.group(1))) {
                            properties.put(matcher.group(1), matcher.group(2));
                        } else {
                            otherVarMap.put(matcher.group(1), matcher.group(2));
                        }
                    } else {
                        // System.out.println("skipped " + line);
                    }
                });
                properties.store(Files.newOutputStream(propertiesPath), "A comment");
            } catch (IOException ioException) {
                throw new IllegalStateException(ioException);
            }
        }
        available = asBoolean(response);
        sysName = asString("CMAKE_HOST_SYSTEM_NAME");
        darwin = sysName.equals("Darwin");
        linux = sysName.equals("Linux");
        fwk = darwin?asString("CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES"):null;
    }

    @Override
    public void writeCompilerFlags(Path outputDir) {
        if (darwin) {
            try {
                Path compileFLags = outputDir.resolve("compile_flags.txt");
                Files.writeString(compileFLags, "-F" + fwk + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE);
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }
    }
    @Override
    public boolean isAvailable() {
        return available;
    }
}
