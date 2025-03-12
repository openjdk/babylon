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
package nbody;

public enum Mode {
    HAT, OpenCL, Cuda, OpenCL4, Cuda4, JavaSeq, JavaMT, JavaSeq4, JavaMT4;

    public static Mode of(String s) {
        return switch (s) {
            case "HAT" -> Mode.HAT;
            case "OpenCL" -> Mode.OpenCL;
            case "Cuda" -> Mode.Cuda;
            case "JavaSeq" -> Mode.JavaSeq;
            case "JavaMT" -> Mode.JavaMT;
            case "JavaSeq4" -> Mode.JavaSeq4;
            case "JavaMT4" -> Mode.JavaMT4;
            case "OpenCL4" -> Mode.OpenCL4;
            case "Cuda4" -> Mode.Cuda4;
            default -> throw new IllegalStateException("No mode " + s);
        };
    }
}
