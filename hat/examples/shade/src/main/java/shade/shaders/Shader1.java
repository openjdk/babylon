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
package shade.shaders;
import hat.types.F32;
import hat.types.mat3;
import hat.types.mat2;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import static hat.types.F32.*;
import static hat.types.mat3.*;

import static hat.types.mat2.*;
import static hat.types.vec2.*;
import static hat.types.vec3.*;
import static hat.types.vec4.*;
import shade.Shader;
import shade.Uniforms;
//https://www.shadertoy.com/view/Md23DV
public class Shader1 implements Shader {

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {

            int w = uniforms.iResolution().x();
            int wDiv3 = uniforms.iResolution().x() / 3;
            int h = uniforms.iResolution().y();
            int hDiv3 = uniforms.iResolution().y() / 3;
            boolean midx = (fragCoord.x() > wDiv3 && fragCoord.x() < (w - wDiv3));
            boolean midy = (fragCoord.y() > hDiv3 && fragCoord.y() < (h - hDiv3));
            if (uniforms.iMouse().x() > wDiv3) {
                if (midx && midy) {
                    return vec4(fragCoord.x(), .0f, fragCoord.y(), 0.f);
                } else {
                    return vec4(0f, 0f, .5f, 0f);
                }
            } else {
                return vec4(1f, 1f, .5f, 0f);
            }
        }
}
