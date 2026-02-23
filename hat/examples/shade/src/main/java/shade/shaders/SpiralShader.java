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


//https://shadertoy.com/view/3llcDl
public class SpiralShader implements Shader {

    /*
    // variant of https://shadertoy.com/view/3llcDl
    // inspired by https://www.facebook.com/eric.wenger.547/videos/2727028317526304/

    void mainImage(out vec4 fragColor,  vec2 fragCoord ){

        vec2 fResolution = iResolution.xy;

        vec2  U = ((2.*fragCoord - fResolution)) / fResolution.y; // normalized coordinates
        vec2  z = U - vec2(-1,0);
        U.x = U.x-.5;                      // Moebius transform

        U = U * mat2(z,-z.y,z.x) / dot(U,U);

        U = U+.5;
                      // offset   spiral, zoom   phase            // spiraling
        U =   log(length(U))*vec2(.5, -.5) + iTime/8. + atan(U.y, U.x)/6.2832 * vec2(6, 1);
        // n
        U = U * 3./vec2(2,1);
        z = //vec2(1);
        fwidth(U);
        U = fract(U)*5.;

        vec2 I = floor(U);
        U = fract(U);              // subdiv big square in 5x5
        I.x = mod( I.x - 2.*I.y , 5.);                            // rearrange
        U = vec2(U.x+ float(I.x==1.||I.x==3.),U.y+float(I.x<2.));     // recombine big tiles

        float id = -1.;

        if (I.x != 4.){
            U =U/2.;                                     // but small times
            id = mod(floor(I.x/2.)+I.y,5.);
        }
        U = abs(fract(U)*2.-1.); float v = max(U.x,U.y);          // dist to border
        fragColor =   smoothstep(.7,-.7, (v-.95)/( abs(z.x-z.y)>1.?.1:z.y*8.))  // draw AA tiles
            * (id<0.?vec4(1): .6 + .6 * cos( id  + vec4(0,23,21,0)  ) );// color
    }
     */


    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        // variant of https://shadertoy.com/view/3llcDl
// inspired by https://www.facebook.com/eric.wenger.547/videos/2727028317526304/

        var fResolution = vec3.xy(uniforms.iResolution());
        vec2 U = div(sub(mul(fragCoord,2f),fResolution),fResolution.y());//.sub(fResolution).div(fResolution.y());
        // normalized coordinates
        var z = sub(U, vec2(-1f, 0f));

        U = sub(U, vec2(.5f, 0f));
        U = div(
                mul(U, mat2(z.x(), z.y(), -z.y(), z.x())),
                dot(U, U)
        );
        // offset   spiral, zoom   phase            // spiraling
        U = add(U, vec2(.5f, 0f));
        //U =   log(length(U))*vec2(.5, -.5) + iTime/8. + atan(U.y, U.x)/6.2832 * vec2(6, 1);
        U = add(
                add(
                        mul(log(length(U)), vec2(.5f, 0.5f)),
                        vec2(uniforms.iTime()/8f)
                ),
                mul(
                        div(F32.atan(U.x(),U.y()), 6.2832f),
                        vec2(6f,1f)
                )
        );


        U = div(mul(U, vec2(3f)),vec2(2f, 1f));
        z = vec2(.001f);//fwidth(U); // this resamples the image.  Not sure how we do this!
        U = mul(fract(U),5f);
        vec2 I = floor(U);
        U = fract(U);             // subdiv big square in 5x5
        I = vec2(mod(I.x() - 2.f * I.y(), 5f), I.y());                            // rearrange
        U = add(U, vec2((I.x() == 1f || I.x() == 3f) ? 1f : 0f, I.x() < 2.0 ? 1f : 0f));     // recombine big tiles
        float id = -1f;
        if (I.x() != 4f) {
            U = div(U,2f);                                     // but small times
            id = mod(floor(I.x() / 2f) + I.y(), 5f);
        }
        U = sub(abs(mul(fract(U),2f)),1f);
        float v = max(U.x(), U.y());          // dist to border

        return
                normalize(
                        smoothstep(
                        vec4(.7f),
                        vec4(-.7f),
                        mul(div(vec4(v - .95f),abs(z.x() - z.y()) > 1f
                                        ? .1f
                                        : z.y() * 8f
                                )
                                ,id < 0f
                                                ? vec4(1f)
                                                : add(mul(vec4(.6f),.6f),
                                                cos(add(vec4(id),vec4(0f, 23f, 21f, 0f)))
                                        )
                                )
                )
                );// color

    };




}
