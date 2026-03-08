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

import hat.Accelerator;
import hat.backend.Backend;
import hat.types.vec2;
import hat.types.vec4;
import shade.Config;
import shade.Shader;
import shade.ShaderApp;
import hat.buffer.Uniforms;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static hat.types.vec4.vec4;

//https://www.shadertoy.com/view/Md23DV
public class MouseSensitiveShader implements Shader {
    static public vec4 createPixel(vec2 fres, float ftime, vec2 fmouse,vec2 fragCoord){
        float w = fres.x();
        float wDiv3 = fres.x() / 3;
        float h = fres.y();
        float hDiv3 = fres.y() / 3;
        boolean midx = (fragCoord.x() > wDiv3 && fragCoord.x() < (w - wDiv3));
        boolean midy = (fragCoord.y() > hDiv3 && fragCoord.y() < (h - hDiv3));
        if (fmouse.x() > wDiv3) {
            if (midx && midy) {
                return vec4(fragCoord.x(), .0f, fragCoord.y(), 0.f);
            } else {
                return vec4(0f, 0f, .5f, 0f);
            }
        } else {
            return vec4(1f, 1f, .5f, 0f);
        }
    }

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        return createPixel(vec2.vec2(uniforms.iResolution().x(),uniforms.iResolution().y()),uniforms.iTime(),vec2.vec2(uniforms.iMouse().x(),uniforms.iMouse().y()),fragCoord);

    }




    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "512"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "512"))),
            new MouseSensitiveShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }
}
