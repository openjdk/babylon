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
import hat.ComputeContext;
import hat.Accelerator.Compute;
import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.types.F32;
import hat.types.mat2;
import hat.types.mat3;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import hat.buffer.Uniforms;
import shade.ShaderViewer;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.abs;
import static hat.types.F32.cos;
import static hat.types.F32.fract;
import static hat.types.F32.max;
import static hat.types.F32.min;
import static hat.types.F32.mix;
import static hat.types.F32.pow;
import static hat.types.F32.sin;
import static hat.types.F32.smoothstep;
import static hat.types.mat2.mat2;
import static hat.types.mat3.mat3;
import static hat.types.vec2.abs;
import static hat.types.vec2.add;
import static hat.types.vec2.cos;
import static hat.types.vec2.div;
import static hat.types.vec2.dot;
import static hat.types.vec2.floor;
import static hat.types.vec2.fract;
import static hat.types.vec2.length;
import static hat.types.vec2.mix;
import static hat.types.vec2.mul;
import static hat.types.vec2.sin;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.add;
import static hat.types.vec3.div;
import static hat.types.vec3.dot;
import static hat.types.vec3.mix;
import static hat.types.vec3.mul;
import static hat.types.vec3.normalize;
import static hat.types.vec3.reflect;
import static hat.types.vec3.sub;
import static hat.types.vec3.vec3;

/*
/*
 * "Seascape" by Alexander Alekseev aka TDM - 2014
 * License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
 * Contact: tdmaav@gmail.com
 *

const int NUM_STEPS = 32;
const float PI             = 3.141592;
const float EPSILON      = 1e-3;
#define EPSILON_NRM (0.1 / iResolution.x)
//#define AA

// sea
const int ITER_GEOMETRY = 3;
const int ITER_FRAGMENT = 5;
const float SEA_HEIGHT = 0.6;
const float SEA_CHOPPY = 4.0;
const float SEA_SPEED = 0.8;
const float SEA_FREQ = 0.16;
const vec3 SEA_BASE = vec3(0.0,0.09,0.18);
const vec3 SEA_WATER_COLOR = vec3(0.8,0.9,0.6)*0.6;
#define SEA_TIME (1.0 + iTime * SEA_SPEED)
const mat2 octave_m = mat2(1.6,1.2,-1.2,1.6);

// math
mat3 fromEuler(vec3 ang) {
    vec2 a1 = vec2(sin(ang.x),cos(ang.x));
    vec2 a2 = vec2(sin(ang.y),cos(ang.y));
    vec2 a3 = vec2(sin(ang.z),cos(ang.z));
    mat3 m;
    m[0] = vec3(a1.y*a3.y+a1.x*a2.x*a3.x,a1.y*a2.x*a3.x+a3.y*a1.x,-a2.y*a3.x);
    m[1] = vec3(-a2.y*a1.x,a1.y*a2.y,a2.x);
    m[2] = vec3(a3.y*a1.x*a2.x+a1.y*a3.x,a1.x*a3.x-a1.y*a3.y*a2.x,a2.y*a3.y);
    return m;
}
float hash( vec2 p ) {
    float h = dot(p,vec2(127.1,311.7));
    return fract(sin(h)*43758.5453123);
}
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );
    vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix(
                    mix( hash( i + vec2(0.0,0.0) ),hash( i + vec2(1.0,0.0) ), u.x),
                    mix( hash( i + vec2(0.0,1.0) ),hash( i + vec2(1.0,1.0) ), u.x),
                     u.y);

}

// lighting
float diffuse(vec3 n,vec3 l,float p) {
    return pow(dot(n,l) * 0.4 + 0.6,p);
}
float specular(vec3 n,vec3 l,vec3 e,float s) {
    float nrm = (s + 8.0) / (PI * 8.0);
    return pow(max(dot(reflect(e,n),l),0.0),s) * nrm;
}

// sky
vec3 getSkyColor(vec3 e) {
    e.y = (max(e.y,0.0)*0.8+0.2)*0.8;
    return vec3(pow(1.0-e.y,2.0), 1.0-e.y, 0.6+(1.0-e.y)*0.4) * 1.1;
}

// sea
float sea_octave(vec2 uv, float choppy) {
    uv += noise(uv);
    vec2 wv = 1.0-abs(sin(uv));
    vec2 swv = abs(cos(uv));
    wv = mix(wv,swv,wv);
    return pow(1.0-pow(wv.x * wv.y,0.65),choppy);
}

float map(vec3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    vec2 uv = p.xz;
     uv.x *= 0.75;


    float d, h = 0.0;
    for(int i = 0; i < ITER_GEOMETRY; i++) {
        d = sea_octave((uv+SEA_TIME)*freq,choppy);
        d += sea_octave((uv-SEA_TIME)*freq,choppy);
        h += d * amp;
        uv *= octave_m;
        freq *= 1.9;
        amp *= 0.22;

        choppy = mix(choppy,1.0,0.2);
    }
    return p.y - h;
}

float map_detailed(vec3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    vec2 uv = p.xz;
     uv.x *= 0.75;

    float d, h = 0.0;
    for(int i = 0; i < ITER_FRAGMENT; i++) {
        d = sea_octave((uv+SEA_TIME)*freq,choppy);
        d += sea_octave((uv-SEA_TIME)*freq,choppy);
        h += d * amp;
        uv *= octave_m;
         freq *= 1.9;
          amp *= 0.22;


        choppy = mix(choppy,1.0,0.2);
    }
    return p.y - h;
}

vec3 getSeaColor(vec3 p, vec3 n, vec3 l, vec3 eye, vec3 dist) {
    float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fresnel = min(fresnel * fresnel * fresnel, 0.5);

    vec3 reflected = getSkyColor(reflect(eye, n));
    vec3 refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12;

    vec3 color = mix(refracted, reflected, fresnel);

    float atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten;

    color += specular(n, l, eye, 600.0 * inversesqrt(dot(dist,dist)));

    return color;
}

// tracing
vec3 getNormal(vec3 p, float eps) {
    vec3 n;
    n.y = map_detailed(p);
    n.x = map_detailed(vec3(p.x+eps,p.y,p.z)) - n.y;
    n.z = map_detailed(vec3(p.x,p.y,p.z+eps)) - n.y;
    n.y = eps;
    return normalize(n);
}

float heightMapTracing(vec3 ori, vec3 dir, out vec3 p) {
    float tm = 0.0;
    float tx = 1000.0;
    float hx = map(ori + dir * tx);
    if(hx > 0.0) {
        p = ori + dir * tx;
        return tx;
    }
    float hm = map(ori);
    for(int i = 0; i < NUM_STEPS; i++) {
        float tmid = mix(tm, tx, hm / (hm - hx));
        p = ori + dir * tmid;
        float hmid = map(p);
        if(hmid < 0.0) {
            tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
        if(abs(hmid) < EPSILON) break;
    }
    return mix(tm, tx, hm / (hm - hx));
}

vec3 getPixel(in vec2 coord, float time) {
    vec2 uv = coord / iResolution.xy;
    uv = uv * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;

    // ray
    vec3 ang = vec3(sin(time*3.0)*0.1,sin(time)*0.2+0.3,time);
    vec3 ori = vec3(0.0,3.5,time*5.0);
    vec3 dir = normalize(vec3(uv.xy,-2.0));
     dir.z += length(uv) * 0.14;

    dir = normalize(dir) * fromEuler(ang);

    // tracing
    vec3 p;
    heightMapTracing(ori,dir,p);
    vec3 dist = p - ori;
    vec3 n = getNormal(p, dot(dist,dist) * EPSILON_NRM);
    vec3 light = normalize(vec3(0.0,1.0,0.8));

    // color
    return mix(
            getSkyColor(dir),
            getSeaColor(p,n,light,dir,dist),
            pow(smoothstep(0.0,-0.02,dir.y),0.2));
}

// main
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    float time = iTime * 0.3 + iMouse.x*0.01;

#ifdef AA
    vec3 color = vec3(0.0);
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            vec2 uv = fragCoord+vec2(i,j)/3.0;
            color += getPixel(uv, time);
        }
    }
    color /= 9.0;
#else
    vec3 color = getPixel(fragCoord, time);
#endif

            // post
            fragColor = vec4(pow(color,vec3(0.65)), 1.0);
}
*/

//https://www.shadertoy.com/view/MdXyzX
public class SeaScapeShader  {

    static final int NUM_STEPS = 32;
    static final float EPSILON = 1e-3f;

    // sea
    final  static public int ITER_GEOMETRY = 3;
    final   static public int ITER_FRAGMENT = 5;
    final  static public  float SEA_HEIGHT = 0.6f;
    final  static public  float SEA_CHOPPY = 4.0f;
    final  static public  float SEA_SPEED = 0.8f;
    final   static public float SEA_FREQ = 0.16f;

    @Reflect public static float SEA_TIME(float ftime){
        return 1.0f + ftime * SEA_SPEED;
    }

    @Reflect
    public static  mat3 fromEuler(vec3 ang) {
        vec2 a1 = vec2(sin(ang.x()), cos(ang.x()));
        vec2 a2 = vec2(sin(ang.y()), cos(ang.y()));
        vec2 a3 = vec2(sin(ang.z()), cos(ang.z()));

        vec3 m_0 = vec3(a1.y() * a3.y() + a1.x() * a2.x() * a3.x(), a1.y() * a2.x() * a3.x() + a3.y() * a1.x(), -a2.y() * a3.x());
        vec3 m_1 = vec3(-a2.y() * a1.x(), a1.y() * a2.y(), a2.x());
        vec3 m_2 = vec3(a3.y() * a1.x() * a2.x() + a1.y() * a3.x(), a1.x() * a3.x() - a1.y() * a3.y() * a2.x(), a2.y() * a3.y());
        mat3 m = mat3(
                m_0.x(), m_0.y(), m_0.z(),  // row maj I think
                m_1.x(), m_1.y(), m_1.z(),
                m_2.x(), m_2.y(), m_2.z()

             //   m_0.x(), m_1.x(), m_2.x(),// column maj I think
              //  m_0.y(), m_1.y(), m_2.y(),
              //  m_0.z(), m_1.z(), m_2.z()
        );
        return m;
    }
    @Reflect public static  float hash(vec2 p) {
        float h = dot(p, vec2(127.1f, 311.7f));
        return fract(sin(h) * 43758.5453123f);
    }

    @Reflect public static  float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = mul(f, mul(f, sub(3.0f, mul(2.0f, f))));

        return -1.0f + 2.0f* F32.mix(
                                F32.mix( hash(add(i, vec2(0.0f, 0.0f))), hash(add(i, vec2(1.0f, 0.0f))), u.x()),
                                F32.mix( hash(add(i, vec2(0.0f, 1.0f))), hash(add(i, vec2(1.0f, 1.0f))), u.x()),
                                u.y()
        );
    }


    @Reflect public static   float diffuse(vec3 n, vec3 l, float p) {
        return pow(dot(n, l) * 0.4f + 0.6f, p);
    }

    @Reflect public static  float specular(vec3 n, vec3 l, vec3 e, float s) {
        float nrm = (s + 8.0f) / (F32.PI * 8.0f);
        return pow(max(dot(reflect(e, n), l), 0.0f), s) * nrm;
    }

    @Reflect public static  vec3 getSkyColor(vec3 e) {
        var ey = max(e.y(), 0.0f) * 0.8f + 0.2f*.8f;
        e = vec3(e.x(), ey, e.z());
        return vec3.mul(
                vec3(F32.pow(1.0f - e.y(), 2.0f), 1.0f - e.y(), 0.6f + (1.0f - e.y()) * 0.4f),
                1.1f
        );
    }


    @Reflect public static  float sea_octave(vec2 uv, float choppy) {
        uv = add(uv, noise(uv));
        vec2 wv = sub(1.0f, abs(sin(uv)));
        vec2 swv = abs(cos(uv));
        wv = mix(wv, swv, wv);
        return pow(1.0f - pow(wv.x() * wv.y(), 0.65f), choppy);
    }


    @Reflect public static  float map(float ftime,vec3 p) {
        final mat2 octave_m = mat2(1.6f, 1.2f, -1.2f, 1.6f);
        float freq = SEA_FREQ;
        float amp = SEA_HEIGHT;
        float choppy = SEA_CHOPPY;
        vec2 uv = vec2(p.x(), p.z());
        uv = mul(uv, vec2(uv.x()*0.75f, 1f));// uv.x *= 0.75;

        float d;
        float h = 0.0f;
        for (int i = 0; i < ITER_GEOMETRY; i++) {
            d = sea_octave(mul(add(uv, SEA_TIME(ftime)), freq), choppy);
            d += sea_octave(mul(sub(uv, SEA_TIME(ftime)), freq), choppy);
            h += d * amp;
            uv = vec2.mul(uv, octave_m);
            freq *= 1.9f;
            amp *= 0.22f;
            choppy = mix(choppy, 1.0f, 0.2f);
        }
        return p.y() - h;
    }

    @Reflect public static  float map_detailed(float ftime, vec3 p) {
        final mat2 octave_m = mat2(1.6f, 1.2f, -1.2f, 1.6f);
        float freq = SEA_FREQ;
        float amp = SEA_HEIGHT;
        float choppy = SEA_CHOPPY;
        vec2 uv = vec2(p.x(), p.z());
        uv = mul(uv, vec2(uv.x()*0.75f, 1f));

        float d;
        float h = 0.0f;
        for (int i = 0; i < ITER_FRAGMENT; i++) {
            d = sea_octave(mul(add(uv, SEA_TIME(ftime)), freq), choppy);
            d += sea_octave(mul(sub(uv, SEA_TIME(ftime)), freq), choppy);
            h += d * amp;
            uv = mul(uv, octave_m);
            freq *= 1.9f;
            amp *= 0.22f;
            choppy = mix(choppy, 1.0f, 0.2f);
        }
        return p.y() - h;
    }

    @Reflect public static vec3 getSeaColor(vec3 p, vec3 n, vec3 l, vec3 eye, vec3 dist) {
        vec3 SEA_BASE = vec3(0.0f, 0.09f, 0.18f);
        vec3 SEA_WATER_COLOR = mul(vec3(0.8f, 0.9f, 0.6f), 0.6f);


        float fresnel = F32.clamp(1.0f - vec3.dot(n, mul(-1f,eye)), 0.0f, 1.0f);
        fresnel = min(fresnel * fresnel * fresnel, 0.5f);

        vec3 reflected = getSkyColor(reflect(eye, n));
        vec3 refracted = add(SEA_BASE, mul(mul(diffuse(n, l, 80.0f), SEA_WATER_COLOR), 0.12f));

        vec3 color = mix(refracted, reflected, fresnel);

        float atten = max(1.0f - dot(dist, dist) * 0.001f, 0.0f);
        color = add(color, mul(mul(SEA_WATER_COLOR, (p.y() - SEA_HEIGHT)), 0.18f * atten));

        color = add(color, specular(n, l, eye, 600.0f * F32.inversesqrt(dot(dist, dist))));

        return color;
    }


    @Reflect public static    vec3 getNormal(float ftime, vec3 p, float eps) {
        float ny = map_detailed(ftime, p);

        vec3 n = vec3(
                map_detailed(ftime, vec3(p.x() + eps, p.y(), p.z())) - ny,
                eps,
                map_detailed(ftime, vec3(p.x(), p.y(), p.z() + eps)) - ny
        );
        return normalize(n);
    }





    @Reflect public static  vec3 heightMapTracingVec3( float ftime, vec3 ori, vec3 dir, vec3 p) {
        float tm = 0.0f;
        float tx = 1000.0f;
        float hx = map(ftime,add(ori, mul(dir, tx)));
        if (hx > 0.0) {
            p = add(ori, mul(dir, tx));
            return p;
        }
        float hm = map(ftime, ori);
        int brk=0;
        for (int i = 0; brk != 1 && i < NUM_STEPS; i++) {
            float tmid = mix(tm, tx, hm / (hm - hx));
            p = add(ori, mul(dir, tmid));
            float hmid = map(ftime, p);
            if (hmid < 0.0) {
                tx = tmid;
                hx = hmid;
            } else {
                tm = tmid;
                hm = hmid;
            }
            if (abs(hmid) < EPSILON) brk=1;
        }
        return  p;
    }


    @Reflect public static vec3 getPixel( vec2 fres, float ftime, vec2 coord, float time) {

        vec2 uv = div(coord, fres);
        uv = sub(mul(uv, 2.0f), 1.0f);
        uv = mul(uv, fres.x() / fres.y());

        // ray
        vec3 ang = vec3(sin(time * 3.0f) * 0.1f, sin(time) * 0.2f + 0.3f, time);
        vec3 ori = vec3(0.0f, 3.5f, time * 5.0f);
        vec3 dir = normalize(vec3(uv.x(), uv.y(), -2.0f));
        dir = add(dir, vec3(0f, 0f, length(uv) * 0.14f));
        dir = mul(normalize(dir), fromEuler(ang));

        // tracing
        vec3 p = vec3(0f);
        p  = heightMapTracingVec3( ftime, ori, dir, p);
        vec3 dist = sub(p, ori);
        float EPSILON_NRM = 0.1f / fres.x();
        vec3 n = getNormal(ftime, p, dot(dist, dist) * EPSILON_NRM);
        vec3 light = normalize(vec3(0.0f, 1.0f, 0.8f));

        // color
        return mix(
                getSkyColor(dir),
                getSeaColor(p, n, light, dir, dist),
                pow(smoothstep(0.0f, -0.02f, dir.y()), 0.2f));
    }


    @Reflect public static vec4 createPixel(vec2 fres, float ftime, vec2 fmouse, vec2 fragCoord){
        final float time = ftime *.3f + fmouse.x()*.01f;//SEA_TIME = 1f + ftime * SEA_SPEED;

        vec3 color = vec3(0f);
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                vec2 uv = add(fragCoord, div(vec2(i, j), 3.0f));
                var pix = getPixel( fres, ftime,uv, time);
                color = add(color, pix);
            }
        }
        color = div(color, 9.0f);
        return vec4.vec4(vec3.pow(color, vec3(0.65f)), 1.0f);
    }

    //https://www.shadertoy.com/view/Ms2SD1
    @Reflect public static vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        return createPixel(vec2(uniforms.iResolution().x(),uniforms.iResolution().y()),uniforms.iTime(),vec2(uniforms.iMouse().x(),uniforms.iMouse().y()),fragCoord);
    }

    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RW F32Array f32Array) {
        int width = (int) uniforms.iResolution().x();
        int height = (int) uniforms.iResolution().y();
        var fragColor = mainImage(uniforms, vec4.vec4(0f), vec2.vec2((float)(kc.gix % width), (float)(height-(kc.gix / width))));
        f32Array.array(kc.gix * 3, fragColor.x());
        f32Array.array(kc.gix * 3+1, fragColor.y());
        f32Array.array(kc.gix * 3+2, fragColor.z());
    }

    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {
        computeContext.dispatchKernel(NDRange.of1D(width * height), (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }

    public static void update(  Accelerator acc, Uniforms uniforms, F32Array f32Array, int width, int height) {
        acc.compute((@Reflect Compute) cc -> compute(cc, uniforms, f32Array, width, height));
    }

    static void main(String[] args) {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var shader = ShaderViewer.of(acc, SeaScapeShader.class,512, 512);
        shader.startLoop((uniforms, f32Array) -> update( acc, uniforms, f32Array, shader.view.getWidth(), shader.view.getHeight()));
    }
}
