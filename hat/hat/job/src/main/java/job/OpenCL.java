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

import java.util.Set;

public class OpenCL extends CMakeInfo {
    public OpenCL(Project.Id id, Set<Dependency> buildDependencies) {
        super(id, "OpenCL", "OPENCL_FOUND", Set.of(
                "OPENCL_FOUND",
                "OpenCL_FOUND",
                "OpenCL_INCLUDE_DIRS",
                "OpenCL_LIBRARY",
                "OpenCL_VERSION_STRING"
        ), buildDependencies);
    }
    public OpenCL(Project.Id id, Dependency ...dependencies) {
        this(id,Set.of(dependencies));
    }

    @Override
    public void jExtractOpts(ForkExec.Opts opts) {
        if (isAvailable()) {
            if (darwin) {
                opts.add(
                        "--library", ":/System/Library/Frameworks/OpenCL.framework/OpenCL",
                        "--header-class-name", "opencl_h",
                        fwk + "/OpenCL.framework/Headers/opencl.h"
                );
            } else if (linux) {
                opts.add(
                        "--library", asString("OpenCL_LIBRARY"),
                        "--include-dir","\"/usr/include/linux;/usr/include\"",
                        "--header-class-name", "opencl_h",
                        asString("OpenCL_INCLUDE_DIRS") + "/CL/opencl.h"
                );
            }
        }
    }
}
