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
import hat.types.mat2;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Shader;
import shade.Uniforms;

/*
float square(vec2 r, vec2 bottomLeft, float side) {
    vec2 p = r - bottomLeft;
    return ( p.x > 0.0 && p.x < side && p.y>0.0 && p.y < side ) ? 1.0 : 0.0;
}

float character(vec2 r, vec2 bottomLeft, float charCode, float squareSide) {
    vec2 p = r - bottomLeft;
    float ret = 0.0;
    float num, quotient, remainder, divider;
    float x, y;
    num = charCode;
    for(int i=0; i<20; i++) {
        float boxNo = float(19-i);
        divider = pow(2., boxNo);
        quotient = floor(num / divider);
        remainder = num - quotient*divider;
        num = remainder;

        y = floor(boxNo/4.0);
        x = boxNo - y*4.0;
        if(quotient == 1.) {
            ret += square( p, squareSide*vec2(x, y), squareSide );
        }
    }
    return ret;
}

mat2 rot(float th) { return mat2(cos(th), -sin(th), sin(th), cos(th)); }

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float G = 990623.; // compressed characters :-)
    float L = 69919.;
    float S = 991119.;

    float t = iTime;

    vec2 r = (fragCoord.xy - 0.5*iResolution.xy) / iResolution.y;
    //vec2 rL = rot(t)*r+0.0001*t;
    //vec2 rL = r+vec2(cos(t*0.02),sin(t*0.02))*t*0.05;
    float c = 0.05;//+0.03*sin(2.5*t);
    vec2 pL = (mod(r+vec2(cos(0.3*t),sin(0.3*t)), 2.0*c)-c)/c;
    float circ = 1.0-smoothstep(0.75, 0.8, length(pL));
    vec2 rG = rot(2.*3.1415*smoothstep(0.,1.,mod(1.5*t,4.0)))*r;
    vec2 rStripes = rot(0.2)*r;

    float xMax = 0.5*iResolution.x/iResolution.y;
    float letterWidth = 2.0*xMax*0.9/4.0;
    float side = letterWidth/4.;
    float space = 2.0*xMax*0.1/5.0;

    r += 0.001; // to get rid off the y=0 horizontal blue line.
    float maskGS = character(r, vec2(-xMax+space, -2.5*side)+vec2(letterWidth+space, 0.0)*0.0, G, side);
    float maskG = character(rG, vec2(-xMax+space, -2.5*side)+vec2(letterWidth+space, 0.0)*0.0, G, side);
    float maskL1 = character(r, vec2(-xMax+space, -2.5*side)+vec2(letterWidth+space, 0.0)*1.0, L, side);
    float maskSS = character(r, vec2(-xMax+space, -2.5*side)+vec2(letterWidth+space, 0.0)*2.0, S, side);
    float maskS = character(r, vec2(-xMax+space, -2.5*side)+vec2(letterWidth+space, 0.0)*2.0 + vec2(0.01*sin(2.1*t),0.012*cos(t)), S, side);
    float maskL2 = character(r, vec2(-xMax+space, -2.5*side)+vec2(letterWidth+space, 0.0)*3.0, L, side);
    float maskStripes = step(0.25, mod(rStripes.x - 0.5*t, 0.5));

    float i255 = 0.00392156862;
    vec3 blue = vec3(43., 172., 181.)*i255;
    vec3 pink = vec3(232., 77., 91.)*i255;
    vec3 dark = vec3(59., 59., 59.)*i255;
    vec3 light = vec3(245., 236., 217.)*i255;
    vec3 green = vec3(180., 204., 18.)*i255;

    vec3 pixel = blue;
    pixel = mix(pixel, light, maskGS);
    pixel = mix(pixel, light, maskSS);
    pixel -= 0.1*maskStripes;
    pixel = mix(pixel, green, maskG);
    pixel = mix(pixel, pink, maskL1*circ);
    pixel = mix(pixel, green, maskS);
    pixel = mix(pixel, pink, maskL2*(1.-circ));

    float dirt = pow(texture(iChannel0, 4.0*r).x, 4.0);
    pixel -= (0.2*dirt - 0.1)*(maskG+maskS); // dirt
    pixel -= smoothstep(0.45, 2.5, length(r));
    fragColor = vec4(pixel, 1.0);
}
 */

//https://www.shadertoy.com/view/Md23DV
public class IntroShader implements Shader {

    float square(vec2 r, vec2 bottomLeft, float side) {
        vec2 p = vec2.sub(r,  bottomLeft);
        return ( p.x() > 0f && p.x() < side && p.y()>0f && p.y() < side ) ? 1f : 0f;
    }

    float character(vec2 r, vec2 bottomLeft, float charCode, float squareSide) {
        vec2 p = vec2.sub(r, bottomLeft);
        float ret = 0f;
        float num=charCode;
        float quotient, remainder;
        float x, y;
        for(int i=0; i<20; i++) {
            float boxNo = 19f-i;
            float divider = F32.pow(2f, boxNo);
            quotient = F32.floor(num / divider);
            remainder = num - quotient*divider;
            num = remainder;

            y = F32.floor(boxNo/4f);
            x = boxNo - y*4f;
            if(quotient == 1f) {
                ret += square( p, vec2.mul(vec2.vec2(squareSide),vec2.vec2(x, y)), squareSide );
            }
        }
        return ret;
    }

    mat2 rot(float th) { return mat2.mat2(F32.cos(th), -F32.sin(th), F32.sin(th), F32.cos(th)); }

    @Override public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord ) {
        float G = 990623f; // compressed characters :-)
        float L = 69919f;
        float S = 991119f;

        float t = uniforms.iTime();
        vec2 fres = vec2.vec2(uniforms.iResolution());
        vec2 r = vec2.div(vec2.sub(fragCoord, vec2.mul(vec2.vec2(-0.5f),fres)),vec2.vec2(fres.y()));

        float c = 0.05f;

         vec2 cos_3_sin_3 = vec2.vec2(F32.cos(0.3f*t),F32.sin(0.3f*t));
         vec2 rplus = vec2.add(r,cos_3_sin_3);
         vec2 cAsVec2= vec2.vec2(c);
         vec2 pL = vec2.div(vec2.sub(vec2.mod(rplus),cAsVec2),cAsVec2);
        float circ = 1f-F32.smoothstep(0.75f, 0.8f, vec2.length(pL));

        vec2 rG = mat2.mul(rot(2f*3.1415f*F32.smoothstep(0f,1f,F32.mod(1.5f*t,4.0f))),r);
        vec2 rStripes = mat2.mul(rot(0.2f),r);


        float xMax = 0.5f*fres.x()/fres.y();
        float letterWidth = 2f*xMax*0.9f/4f;
        float side = letterWidth/4f;
        float space = 2f*xMax*0.1f/5f;
        vec2 letterWidthPlusSpace = vec2.vec2(letterWidth+space, 0f);
        r = vec2.add(r,vec2.vec2(0.001f)); // to get rid off the y=0 horizontal blue line.
        float maskGS = character(r, vec2.vec2(-xMax+space, -2.5f*side), G, side);
        float maskG = character(rG, vec2.vec2(-xMax+space, -2.5f*side), G, side);
        float maskL1 = character(r, vec2.add(vec2.vec2(-xMax+space, -2.5f*side)
                ,vec2.mul(letterWidthPlusSpace,vec2.vec2(1f))), L, side);
        float maskSS = character(r, vec2.add(vec2.vec2(-xMax+space, -2.5f*side)
                ,vec2.mul(letterWidthPlusSpace,vec2.vec2(2f))), S, side);

          float maskS = character(r, vec2.add(
                 vec2.vec2(-xMax+space, -2.5f*side),
                vec2.add(
                        vec2.mul(
                                vec2.vec2(letterWidth+space, 0f),vec2.vec2(2.0f)
                        )
                        , vec2.vec2(
                                0.01f*F32.sin(2.1f*t) ,0.012f*F32.cos(t))
                )), S, side);

        float maskL2 = character(r, vec2.add(vec2.vec2(-xMax+space, -2.5f*side),vec2.mul(letterWidthPlusSpace,vec2.vec2(3f))), L, side);
        float maskStripes = F32.step(0.25f, F32.mod(rStripes.x() - 0.5f*t, 0.5f));

        vec3 i255 = vec3.vec3(0.00392156862f);

        vec3 blue = vec3.mul(vec3.vec3(43f, 172f, 181f),i255);
        vec3 pink = vec3.mul(vec3.vec3(232f, 77f, 91f),i255);
     //   vec3 dark = vec3.mul(vec3.vec3(59f, 59f, 59f),i255);
        vec3 light = vec3.mul(vec3.vec3(245f, 236f, 217f),i255);
        vec3 green = vec3.mul(vec3.vec3(180f, 204f, 18f),i255);

        vec3 pixel = blue;
        pixel = vec3.mix(pixel, light, maskGS);
        pixel = vec3.mix(pixel, light, maskSS);
        pixel = vec3.sub(pixel,vec3.vec3(0.1f*maskStripes));
        pixel = vec3.mix(pixel, green, maskG);
        pixel = vec3.mix(pixel, pink, maskL1*circ);
        pixel = vec3.mix(pixel, green, maskS);
        pixel = vec3.mix(pixel, pink, maskL2*(1f-circ));

        float dirt = .1f;// F32.pow(texture(iChannel0, 4f*r).x, 4f);
        pixel = vec3.sub(pixel,vec3.vec3((0.2f*dirt - 0.1f)*(maskG+maskS))); // dirt
        pixel = vec3.sub(pixel, vec3.vec3(F32.smoothstep(0.45f, 2.5f, vec2.length(r))));
        fragColor = vec4.vec4(pixel, 1f);
        return fragColor;
    }
}
