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

import hat.types.F32;
import hat.types.vec3;
import shade.shaders.AcesShader;
import shade.shaders.AnimShader;
import shade.shaders.IntroShader;
import shade.shaders.RandShader;
import shade.shaders.SeaScapeShader;
import shade.shaders.MouseSensitiveShader;
import shade.shaders.GroovyShader;
import shade.shaders.SpiralShader;
import shade.shaders.SquareWaveShader;
import shade.shaders.TutorialShader;
import shade.shaders.WavesShader;
import  hat.types.vec2;
import static hat.types.vec2.vec2;
import static hat.types.vec4.vec4;

enum ShaderEnum {
    Blue((uniform, fragColor, fragCoord) -> {
        return vec4(0f, 0f, 1f, 0f);
    }),
    Gradient((uniforms, fragColor, fragCoord) -> {
        var fResolution = vec3.xy(uniforms.iResolution());
        float fFrame = uniforms.iFrame();
        var uv = vec2.div(fragCoord,fResolution);
        return vec4(uv.x(), uv.y(), F32.max(fFrame / 100f, 1f), 0f);
    }),

    MouseSensitive(new MouseSensitiveShader()),
    Groovy(new GroovyShader()),
    Rand(new RandShader()),
    Spiral(new SpiralShader()),
    Aces(new AcesShader()),
    Anim(new AnimShader()),
    Waves(new WavesShader()),
    Intro(new IntroShader()),
    Tutorial(new TutorialShader()),
    SquareWave(new SquareWaveShader()),
    SeaScape(new SeaScapeShader());
    Shader shader;

    ShaderEnum(Shader shader) {
        this.shader = shader;
    }
}
