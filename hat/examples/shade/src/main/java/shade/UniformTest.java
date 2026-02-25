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
package shade;

import hat.types.vec3;
import hat.types.vec4;
import optkl.util.carriers.ArenaAndLookupCarrier;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;

import static hat.types.ivec2.ivec2;

public class UniformTest {
    static void main(String[] args) {
        var alc = ArenaAndLookupCarrier.of(MethodHandles.lookup(), Arena.global());
        Uniforms uniforms = Uniforms.create(alc);
        var fc = uniforms.fragColor();
        var resolution = uniforms.iResolution();
        IO.println("fc " + fc);
        IO.println("resolution " + resolution);
        resolution.of(vec3.add(resolution, 2));
        IO.println("resolution " + resolution);
        vec4 color = vec4.vec4(1f, 2f, 3f, 4f);
        IO.println("color = " + color);
        fc.of(vec4.div(vec4.add(fc, color), 2f));
        IO.println("fc " + fc);
        uniforms.iTime(0L);
        IO.println("iTime = " + uniforms.iTime());
        uniforms.iTime(2L);
        IO.println("iTime = " + uniforms.iTime());
        uniforms.iFrame(0L);
        IO.println("iFrame = " + uniforms.iFrame());
        uniforms.iFrame(1L);
        IO.println("iFrame = " + uniforms.iFrame());
        var iMouse = uniforms.iMouse();
        IO.println("iMouse " + iMouse);
        iMouse.of(ivec2(0, 1));
    }
}
