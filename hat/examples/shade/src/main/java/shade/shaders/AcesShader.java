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
import hat.types.mat3;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Config;
import shade.Shader;
import shade.ShaderApp;
import shade.Uniforms;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.PI;
import static hat.types.F32.abs;
import static hat.types.F32.exp;
import static hat.types.mat3.mat3;
import static hat.types.vec2.div;
import static hat.types.vec2.mul;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.add;
import static hat.types.vec3.clamp;
import static hat.types.vec3.div;
import static hat.types.vec3.mul;
import static hat.types.vec3.pow;
import static hat.types.vec3.sin;
import static hat.types.vec3.sub;
import static hat.types.vec3.vec3;
import static hat.types.vec4.vec4;

/*
// Based on http://www.oscars.org/science-technology/sci-tech-projects/aces
vec3 aces_tonemap(vec3 color){
    mat3 m1 = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    mat3 m2 = mat3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    vec3 v = m1 * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 position = (fragCoord/iResolution.xy)* 2.0 - 1.0;
    position.x += iTime * 0.2;

    vec3 color = pow(sin(position.x * 4.0 + vec3(0.0, 1.0, 2.0) * 3.1415 * 2.0 / 3.0) * 0.5 + 0.5, vec3(2.0)) * (exp(abs(position.y) * 4.0) - 1.0);;

    if(position.y < 0.0){
        color = aces_tonemap(color);
    }

    fragColor = vec4(color,1.0);
}
 */

// https://www.shadertoy.com/view/XsGfWV
public class AcesShader implements Shader {
    static vec3 aces_tonemap(vec3 color) {
        mat3 m1 = mat3(
                0.59719f, 0.07600f, 0.02840f,
                0.35458f, 0.90834f, 0.13383f,
                0.04823f, 0.01566f, 0.83777f
        );
        mat3 m2 = mat3(
                1.60475f, -0.10208f, -0.00327f,
                -0.53108f, 1.10813f, -0.07276f,
                -0.07367f, -0.00605f, 1.07602f
        );
        vec3 v = mul(color, m1);
        vec3 a = sub(mul(v, add(v, 0.0245786f)), 0.000090537f);
        vec3 b = add(mul(v, add(mul(0.983729f, v), 0.4329510f)), 0.238081f);
        return clamp(pow(mul(div(a, b), m2), 1.0f / 2.2f), 0f, 1f);
    }

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fres = vec3.xy(uniforms.iResolution());
        vec2 position = sub(mul(div(fragCoord, fres), 2f), 1);
        //fragCoord.div(vec2(uniforms.iResolution())).mul(2f).sub(1f); //fragCoord/iResolution.xy)* 2.0 - 1.0;
        position = vec2(position.x() + uniforms.iTime() * 0.2f, position.y()); // position.x += iTime * 0.2;

        vec3 v012 = vec3(0f, 1f, 2f);
        vec3 v0123x2Pi = mul(mul(v012, PI), 2f);
        vec3 v0123x2PiDiv3 = div(v0123x2Pi, 3f);
        vec3 sinCoef = add(v0123x2PiDiv3, position.x() * 4);
        vec3 color = mul(pow(add(mul(sin(sinCoef), 0.5f), 0.5f), 2f), exp(abs(position.y()) * 4f) - 1f);
        if (position.y() < 0f) {
            color = aces_tonemap(color);
        }
        fragColor = vec4(clamp(color, 0f, 1f), 1.0f);
        return fragColor;
    }

    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "1024"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "1024"))),
            Integer.parseInt(System.getProperty("targetFps", "30")),
            new AcesShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }
}
