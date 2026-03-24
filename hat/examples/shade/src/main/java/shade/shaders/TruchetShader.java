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
import hat.Accelerator.Compute;
import hat.ComputeContext;

import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.types.F32;
import hat.types.mat2;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import hat.buffer.Uniforms;
import shade.ShaderViewer;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.PIx2;
import static hat.types.F32.abs;
import static hat.types.F32.cos;
import static hat.types.F32.fract;
import static hat.types.F32.log;
import static hat.types.F32.max;
import static hat.types.F32.min;
import static hat.types.F32.sin;
import static hat.types.F32.smoothstep;
import static hat.types.mat2.mat2;
import static hat.types.vec2.abs;
import static hat.types.vec2.add;
import static hat.types.vec2.div;
import static hat.types.vec2.dot;
import static hat.types.vec2.floor;
import static hat.types.vec2.fract;
import static hat.types.vec2.length;
import static hat.types.vec2.mul;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.add;
import static hat.types.vec3.cos;
import static hat.types.vec3.mix;
import static hat.types.vec3.mul;
import static hat.types.vec3.pow;
import static hat.types.vec3.vec3;
import static hat.types.vec4.normalize;
import static hat.types.vec4.vec4;

public class TruchetShader  {

  @Reflect public static mat2 rot(float a) {
        //  mat2 rot(float a) {return mat2(cos(a),sin(a),-sin(a),cos(a));}
        return mat2(cos(a), sin(a), -sin(a), cos(a));
    }

    @Reflect public static vec3 hue(float t, float f) {
        //  return f+f*cos(PI2*t*(vec3(1,.75,.75)+vec3(.96,.57,.12)));
        return add(f, mul(f, cos(mul(PIx2 * t,
                add(vec3(1f, .75f, .75f), vec3(.96f, .57f, .12f))
        ))));
    }

    @Reflect public static float hash21(vec2 a) {
        // return fract(sin(dot(a,vec2(27.69,32.58)))*43758.53);
        return fract(sin(dot(a, vec2(27.69f, 32.58f))) * 43758.53f);
    }

    @Reflect public static  float box(vec2 p, vec2 b) {
        //vec2 d = abs(p)-b;
        vec2 d = sub(abs(p), b);
        // return length(max(d,0.)) + min(max(d.x,d.y),0.);}
        return length(vec2.max(d, 0f)) + min(max(d.x(), d.y()), 0f);
    }

    @Reflect public static vec2 pattern(vec2 p, float sc) {
        mat2 r90 = rot(1.5707f);
        // vec2 uv = p;

        //vec2 id = floor(p*sc);
        vec2 id = floor(mul(p, sc));
        //p = fract(p*sc)-.5;
        p = sub(fract(mul(p, sc)), .5f);

        float rnd = hash21(id);

        // turn tiles
        if (rnd > .5f) {
            p = mul(p, r90);
        }
        rnd = fract(rnd * 32.54f);
        if (rnd > .4f) {
            p = mul(p, r90);
        }
        if (rnd > .8f) {
            p = mul(p, r90);
        }

        // randomize hash for type
        rnd = fract(rnd * 47.13f);

        float tk = .075f;
        // kind of messy and long winded
        float d = box(sub(p, vec2(.6f, .7f)), vec2(.25f, .75f)) - .15f;
        float l = box(sub(p, vec2(.7f, .5f)), vec2(.75f, .15f)) - .15f;
        float b = box(add(p, vec2(0f, .7f)), vec2(.05f, .25f)) - .15f;
        float r = box(add(p, vec2(.6f, 0f)), vec2(.15f, .05f)) - .15f;
        d = abs(d) - tk;

        if (rnd > .92f) {
            d = box(sub(p, vec2(-.6f, .5f)), vec2(.25f, .15f)) - .15f;
            l = box(sub(p, vec2(.6f, .6f)), vec2(.25f)) - .15f;
            b = box(add(p, vec2(.6f, .6f)), vec2(.25f)) - .15f;
            r = box(sub(p, vec2(.6f, -.6f)), vec2(.25f)) - .15f;
            d = abs(d) - tk;

        } else if (rnd > .6f) {
            d = F32.abs(p.x() - .2f) - tk;//length(p.x()-.2f)-tk;
            l = box(sub(p, vec2(-.6f, .5f)), vec2(.25f, .15f)) - .15f;
            b = box(add(p, vec2(.6f, .6f)), vec2(.25f)) - .15f;
            r = box(sub(p, vec2(.3f, 0f)), vec2(.25f, .05f)) - .15f;
        }

        l = abs(l) - tk;
        b = abs(b) - tk;
        r = abs(r) - tk;

        float e = min(d, min(l, min(b, r)));

        if (rnd > .6f) {
            r = max(r, -box(sub(p, vec2(.2f, .2f)), vec2(tk * 1.3f)));
            d = max(d, -box(add(p, vec2(-.2f, .2f)), vec2(tk * 1.3f)));
        } else {
            l = max(l, -box(sub(p, vec2(.2f, .2f)), vec2(tk * 1.3f)));
        }

        d = min(d, min(l, min(b, r)));

        return vec2(d, e);
    }

    @Reflect public static vec4 createPixel(vec2 fres, float ftime, vec2 fmouse, vec2 fragCoord){


        vec3 color = vec3(0f);
        // vec2 uv = (2.*F-R.xy)/max(R.x,R.y);
        var uv = div(sub(mul(2f,fragCoord),fres),max(fres.x(),fres.y()));

        // uv *= rot(T*.095);
        //   uv = mul(uv, rot(fTime * .095f));

        // uv = vec2(log(length(uv)), atan(uv.y, uv.x)*6./PI2);
        // Original.
        //uv = vec2(log(length(uv)), atan(uv.y, uv.x))*8./6.2831853;

        uv = vec2(log(length(uv)), F32.atan(uv.y(), uv.x()) * 6f / (PIx2));

        float scale = 8f;
        for (float i = 0f; i < 4f; i++) {
            float ff = (i * .05f) + .2f;
            // uv.x+=T*ff;
            uv = add(uv, vec2(ftime * ff, 0f));

            float fwidth = 0.0001f;
            float px = fwidth;//fwidth(uv.x*scale);
            // vec2 d = pattern(uv,scale);
            vec2 d = pattern(uv, scale);
            // vec3 clr = hue(sin(uv.x+(i*8.))*.2+.4,(.5+i)*.15);
            vec3 clr = hue(sin(uv.x() + (i * 8f)) * .2f + .4f, (.5f + i) * .15f);
            // C = mix(C,vec3(.001),smoothstep(px,-px,d.y-.04));
            color = mix(color, vec3(.001f), smoothstep(px, -px, d.y() - .04f));
            // C = mix(C,clr,smoothstep(px,-px,d.x));
            color = mix(color, clr, smoothstep(px, -px, d.x()));
            scale *= .5f;
        }

        // Output to screen
        color = pow(color, vec3(.4545f));
        return normalize(vec4(color, 1.0f));
    }

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

    private static void update(  Accelerator acc, Uniforms uniforms, F32Array f32Array, int width, int height) {
        acc.compute((@Reflect Compute) cc -> compute(cc, uniforms, f32Array, width, height));
    }

    static void main(String[] args) {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var shader = ShaderViewer.of(acc, TruchetShader.class,1024, 1024);
        shader.startLoop((uniforms, f32Array) -> update( acc, uniforms, f32Array, shader.view.getWidth(), shader.view.getHeight()));
    }
}

