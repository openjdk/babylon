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
import hat.types.F32;
import static hat.types.F32.*;
import hat.types.vec4;
import static hat.types.vec4.*;
import hat.types.vec2;
import static hat.types.vec2.*;
import shade.Config;
import shade.Shader;
import shade.ShaderApp;
import hat.buffer.Uniforms;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import static hat.types.vec4.normalize;


public class HelloWorldShader implements Shader {

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        float fTime = uniforms.iTime();
        var v = vec4(1f);
        // v = vec4.add(v,v);
        return vec4(1f, abs(cos(fTime)),sin(fTime),0f);
    }

    static Config controls = Config.of(
           Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST)  : null,
          // new Accelerator(MethodHandles.lookup(), Backend.FIRST),
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "2024"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "2024"))),
            Integer.parseInt(System.getProperty("targetFps", "60")),
            new HelloWorldShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }

}
