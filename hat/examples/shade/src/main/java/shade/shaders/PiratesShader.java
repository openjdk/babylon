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
import hat.buffer.Uniforms;
import hat.types.F32;
import hat.types.mat3;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Config;
import shade.Shader;
import shade.ShaderApp;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.abs;
import static hat.types.F32.max;
import static hat.types.F32.min;
import static hat.types.F32.pow;
import static hat.types.F32.sqrt;
import static hat.types.mat2.mat2;
import static hat.types.mat3.mat3;
import static hat.types.vec2.add;
import static hat.types.vec2.cos;
import static hat.types.vec2.div;
import static hat.types.vec2.dot;
import static hat.types.vec2.length;
import static hat.types.vec2.mul;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.add;
import static hat.types.vec3.clamp;
import static hat.types.vec3.cross;
import static hat.types.vec3.distance;
import static hat.types.vec3.div;
import static hat.types.vec3.dot;
import static hat.types.vec3.max;
import static hat.types.vec3.mix;
import static hat.types.vec3.mul;
import static hat.types.vec3.neg;
import static hat.types.vec3.normalize;
import static hat.types.vec3.pow;
import static hat.types.vec3.reflect;
import static hat.types.vec3.sub;
import static hat.types.vec3.vec3;
import static hat.types.vec4.normalize;
import static hat.types.vec4.vec4;
class  PiratesShader implements Shader{
/*
float fbm( vec2 p )
{
    return 0.5000*texture( iChannel1, p*1.00 ).x +
           0.2500*texture( iChannel1, p*2.02 ).x +
           0.1250*texture( iChannel1, p*4.03 ).x +
           0.0625*texture( iChannel1, p*8.04 ).x;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float time = mod( iTime, 60.0 );
   vec2 p = (2.0*fragCoord-iResolution.xy) / iResolution.y;
    vec2 i = p;

    // camera
    p += vec2(1.0,3.0)*0.001*2.0*cos( iTime*5.0 + vec2(0.0,1.5) );
    p += vec2(1.0,3.0)*0.001*1.0*cos( iTime*9.0 + vec2(1.0,4.5) );
    float an = 0.3*sin( 0.1*time );
    float co = cos(an);
    float si = sin(an);
    p = mat2( co, -si, si, co )*p*0.85;

    // water
    vec2 q = vec2(p.x,1.0)/p.y;
    q.y -= 0.9*time;
    vec2 off = texture( iChannel0, 0.1*q*vec2(1.0,2.0) - vec2(0.0,0.007*iTime) ).xy;
    q += 0.4*(-1.0 + 2.0*off);
    vec3 col = 0.2*sqrt(texture( iChannel0, 0.05*q *vec2(1.0,4.0) + vec2(0.0,0.01*iTime) ).zyx);
    float re = 1.0-smoothstep( 0.0, 0.7, abs(p.x-0.6) - abs(p.y)*0.5+0.2 );
    col += 1.0*vec3(1.0,0.9,0.73)*re*0.2*(0.1+0.9*off.y)*5.0*(1.0-col.x);
    float re2 = 1.0-smoothstep( 0.0, 2.0, abs(p.x-0.6) - abs(p.y)*0.85 );
    col += 0.7*re2*smoothstep(0.35,1.0,texture( iChannel1, 0.075*q *vec2(1.0,4.0) ).x);

    // sky
    vec3 sky = vec3(0.0,0.05,0.1)*1.4;
    // stars
    sky += 0.5*smoothstep( 0.95,1.00,texture( iChannel1, 0.25*p ).x);
    sky += 0.5*smoothstep( 0.85,1.0,texture( iChannel1, 0.25*p ).x);
    sky += 0.2*pow(1.0-max(0.0,p.y),2.0);
    // clouds
    float f = fbm( 0.002*vec2(p.x,1.0)/p.y );
    vec3 cloud = vec3(0.3,0.4,0.5)*0.7*(1.0-0.85*smoothstep(0.4,1.0,f));
    sky = mix( sky, cloud, 0.95*smoothstep( 0.4, 0.6, f ) );
    sky = mix( sky, vec3(0.33,0.34,0.35), pow(1.0-max(0.0,p.y),2.0) );
    col = mix( col, sky, smoothstep(0.0,0.1,p.y) );

    // horizon
    col += 0.1*pow(clamp(1.0-abs(p.y),0.0,1.0),9.0);

    // moon
    float d = length(p-vec2(0.6,0.5));
    vec3 moon = vec3(0.98,0.97,0.95)*(1.0-0.1*smoothstep(0.2,0.5,f));
    col += 0.8*moon*exp(-4.0*d)*vec3(1.1,1.0,0.8);
    col += 0.2*moon*exp(-2.0*d);
    moon *= 0.85+0.15*smoothstep(0.25,0.7,fbm(0.05*p+0.3));
    col = mix( col, moon, 1.0-smoothstep(0.2,0.22,d) );

    // postprocess
    col = pow( 1.4*col, vec3(1.5,1.2,1.0) );
    col *= clamp(1.0-0.3*length(i), 0.0, 1.0 );

    // fade
    col *=       smoothstep( 3.0, 6.0,time);
    col *= 1.0 - smoothstep(44.0,50.0,time);

    fragColor = vec4( col, 1.0 );
}
 */

    float fbm( vec2 p )
    {
        float t1x = .1f;
        float t2x = .2f;
        float t3x = .3f;
        float t4x = .4f;
        return 0.5000f*t1x +//texture( iChannel1, p*1.00 ).x +
                0.2500f* t2x+//texture( iChannel1, p*2.02 ).x +
                0.1250f* t3x+//texture( iChannel1, p*4.03 ).x +
                0.0625f* t4x;//texture( iChannel1, p*8.04 ).x;
    }
//    https://www.shadertoy.com/view/ldXXDj
    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
/* We need textures for this
        float time = F32.mod( uniforms.iTime(), 60.0f );
        vec2 p = div(sub(mul(2.0f,fragCoord),vec2(uniforms.iResolution().x(),uniforms.iResolution().y())), uniforms.iResolution().y());
        vec2 i = p;

        // camera
        p = add(p, mul(vec2(1.0f,3.0f),mul(0.001f*2.0f,cos( add(uniforms.iTime()*5.0f, vec2(0.0f,1.5f) )))));
        p += vec2(1.0f,3.0f)*0.001f*1.0f*cos( uniforms.iTime()*9.0f + vec2(1.0f,4.5f) );
        float an = 0.3f*F32.sin( 0.1f*time );
        float co = F32.cos(an);
        float si = F32.sin(an);
        p = mul(mul(p,mat2( co, -si, si, co )),0.85f);

        // water
        vec2 q = div(vec2(p.x(),1.0f),p.y());
        q.y() -= 0.9f*time;
        vec2 off = vec2(.1f,.1f);//texture( iChannel0, 0.1*q*vec2(1.0,2.0) - vec2(0.0,0.007*iTime) ).xy;
        q = add(q,mul(0.4f,(add(-1.0f , vec2.mul(2.0f,off)))));
        vec3 col = 0.2f*sqrt(texture( iChannel0, 0.05f*q *vec2(1.0f,4.0f) + vec2(0.0f,0.01f*uniforms.iTime()) ).zyx);
        float re = 1.0f-F32.smoothstep( 0.0f, 0.7f, abs(p.x()-0.6f) - abs(p.y())*0.5f+0.2f );
        col += 1.0f*vec3(1.0f,0.9f,0.73f)*re*0.2f*(0.1f+0.9f*off.y())*5.0f*(1.0f-col.x());
        float re2 = 1.0f-F32.smoothstep( 0.0f, 2.0f, abs(p.x()-0.6f) - abs(p.y())*0.85f );
        col += 0.7f*re2*F32.smoothstep(0.35f,1.0f,texture( iChannel1, 0.075*q *vec2(1f,4.0f) ).x);

        // sky
        vec3 sky = vec3(0.0f,0.05f,0.1f)*1.4f;
        // stars
        sky += 0.5f*vec3.smoothstep( 0.95f,1.00f,texture( iChannel1, 0.25*p ).x);
        sky += 0.5f*vec3.smoothstep( 0.85f,1.0f,texture( iChannel1, 0.25*p ).x);
        sky += 0.2f*pow(1.0f-max(0.0,p.y()),2.0f);
        // clouds
        float f = .1f;//fbm( 0.002*vec2(p.x,1.0)/p.y );
        vec3 cloud = vec3(0.3f,0.4f,0.5f)*0.7*(1.0f-0.85f*smoothstep(0.4f,1.0f,f));
        sky = mix( sky, cloud, 0.95f*F32.smoothstep( 0.4f, 0.6f, f ) );
        sky = mix( sky, vec3(0.33f,0.34f,0.35f), pow(1.0f-max(0.0f,p.y()),2.0f) );
        col = mix( col, sky, F32.smoothstep(0.0f,0.1f,p.y()) );

        // horizon
        col += 0.1*pow(clamp(1.0-abs(p.y),0.0,1.0),9.0);

        // moon
        float d = length(p-vec2(0.6,0.5));
        vec3 moon = vec3(0.98,0.97,0.95)*(1.0-0.1*smoothstep(0.2,0.5,f));
        col += 0.8*moon*exp(-4.0*d)*vec3(1.1,1.0,0.8);
        col += 0.2*moon*exp(-2.0*d);
        moon = mul(moon,0.85f+0.15f*smoothstep(0.25,0.7,fbm(0.05f*p+0.3f)));
        col = mix( col, moon, 1.0f-F32.smoothstep(0.2f,0.22f,d) );

        // postprocess
        col = pow( mul(1.4f,col), vec3(1.5f,1.2f,1.0f) );
        col = mul(col, F32.clamp(1.0f-0.3f*length(i), 0.0f, 1.0f ));

        // fade
        col =       mul(col,F32.smoothstep( 3.0f, 6.0f,time));
        col = mul(col, 1.0f - F32.smoothstep(44.0f,50.0f,time));

        fragColor = vec4( col, 1.0f );

 */
        return vec4(1,1,1,1);
    }

    ;
    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "512"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "512"))),
            Integer.parseInt(System.getProperty("targetFps", "2")),
            new PiratesShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }
}
