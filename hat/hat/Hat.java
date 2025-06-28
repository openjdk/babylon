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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Set;

public class Hat {
    public static void main(String[] argArr) throws IOException, InterruptedException {

        Path userDir = Path.of(System.getProperty("user.dir"));
        var project = new Job.Project(userDir.getFileName().toString().equals("intellij") ? userDir.getParent() : userDir);

        var mac = new Job.Mac(project.id("mac-1.0"), Set.of());
        var opencl = new Job.OpenCL(project.id("opencl-1.0"),  Set.of());
        var opengl = new Job.OpenGL(project.id("opengl-1.0"),  Set.of());
        var cuda = new Job.Cuda(project.id("cuda-1.0"),  Set.of());

        var core = Job.Jar.of(project.id("core-1.0"));
        var backend_ffi_native = Job.CMake.of(project.id("backend-ffi-1.0"), core);
        var ffiSharedBackend = Job.Jar.of(project.id("backend-ffi-shared-1.0"), core, backend_ffi_native);
        var backend_ffi_cuda = Job.Jar.of(project.id("backend-ffi-cuda-1.0"), core, cuda, ffiSharedBackend);
        var backend_ffi_opencl = Job.Jar.of(project.id("backend-ffi-opencl-1.0"), opencl, core, ffiSharedBackend);
        var backend_ffi_mock = Job.Jar.of(project.id("backend-ffi-mock-1.0"), core, ffiSharedBackend);
        var backend_mt_java = Job.Jar.of(project.id("backend-java-mt-1.0"), core);
        var backend_seq_java = Job.Jar.of(project.id("backend-java-mt-1.0"), core);
        var example_mandel = Job.RunnableJar.of(project.id("example-mandel-1.0"), core);
        var example_life = Job.RunnableJar.of(project.id("example-life-1.0"), core);
        var example_squares = Job.RunnableJar.of(project.id("example-squares-1.0"), core);
        var example_heal = Job.RunnableJar.of(project.id("example-heal-1.0"), core);
        var example_violajones = Job.RunnableJar.of(project.id("example-violajones-1.0"), core);
        var extractions_opengl = Job.JExtract.of(project.id("extraction-opengl-1.0"), Job.JExtract.Mac.of(opengl,"GLUT", "OpenGL"), mac, opengl, core);
        var extractions_opencl = Job.JExtract.of(project.id("extraction-opencl-1.0"), Job.JExtract.Mac.of(opencl,"OpenCL"), mac, opencl, core);
        var wraps_wrap = Job.Jar.of(project.id("wrap-wrap-1.0"));
        var wraps_clwrap = Job.Jar.of(project.id("wrap-clwrap-1.0"), extractions_opencl, wraps_wrap);

        var wraps_glwrap = Job.Jar.of(project.id("wrap-glwrap-1.0"),
                Set.of(project.rootPath().resolve("wraps/glwrap/src/main/java/wrap/glwrap/GLCallbackEventHandler.java")), //exclude
                extractions_opengl,
                wraps_wrap);
        var example_nbody = Job.RunnableJar.of(project.id("example-nbody-1.0"), Set.of(wraps_glwrap, wraps_clwrap, wraps_wrap, core, mac));
        if (argArr.length == 0) {
            project.start("run", "ffi-opencl", "nbody");
        } else {
            project.start(argArr);
        }
    }
}