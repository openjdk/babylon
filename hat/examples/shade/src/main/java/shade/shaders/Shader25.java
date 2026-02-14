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
import hat.types.vec2;
import hat.types.vec4;
import shade.Shader;
import shade.Uniforms;

//https://www.shadertoy.com/view/Md23DV
public class Shader25 implements Shader {

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        //            ivec2->vec2
        var fres = vec2.vec2(uniforms.iResolution());
        //            vec2(fragCoord.xy / iResolution.xy);
        var p = fragCoord.div(fres);
        //            r = 2.0*vec2(fragCoord.xy - 0.5*iResolution.xy)/iResolution.y
        var r = fragCoord.sub(fres.mul(.5f)).div(fres.y()).mul(16f);

        float t = ((float) uniforms.iFrame()) / 15f;

        float v1 = F32.sin(r.x() + t);
        float v2 = F32.sin(r.y() + t);
        float v3 = F32.sin((r.x() + r.y()) + t);
        float v4 = F32.sin(r.length() + (1.7f * t));
        float v = v1 + v2 + v3 + v4;

        var ret = vec4.vec4(1f, 1f, 1f, 1f);

        if (p.x() < 1f / 10f) { // Part I
            ret = vec4.vec4(v1);
        } else if (p.x() < 2f / 10f) { // Part II
            // horizontal waves
            ret = vec4.vec4(v2);
        } else if (p.x() < 3f / 10f) { // Part III
            // diagonal waves
            ret = vec4.vec4(v3);
        } else if (p.x() < 4f / 10f) { // Part IV
            // circular waves
            ret = vec4.vec4(v4);
        } else if (p.x() < 5f / 10f) { // Part V
            // the sum of all waves
            ret = vec4.vec4(v);
        } else if (p.x() < 6f / 10f) { // Part VI
            // Add periodicity to the gradients
            ret = vec4.vec4(F32.sin(2f * v));
        } else { // Part VII
            // mix colors
            ret = vec4.vec4(F32.sin(v), F32.sin(v + 0.5f * F32.PI), F32.sin(v + F32.PI), 1f);
        }
        return ret.add(.5f).mul(.5f).clamp(0f, 1f);
    }
}
