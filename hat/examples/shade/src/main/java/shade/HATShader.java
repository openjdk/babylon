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

import hat.ComputeContext;
import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import hat.buffer.Uniforms;

import hat.types.F32;
import hat.types.mat2;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import static hat.types.F32.*;
import static hat.types.vec2.*;
import static hat.types.vec3.*;
import static hat.types.vec4.*;

public class HATShader {

    @Reflect public static mat2 rot(float a) {
        return mat2.mat2(cos(a), sin(a), -sin(a), cos(a));
    }

    @Reflect public  static vec3 hue(float t, float f) {
        return add(f, mul(f, cos(mul(PIx2 * t,
                add(vec3(1f, .75f, .75f), vec3(.96f, .57f, .12f))
        ))));
    }

    @Reflect public static float hash21(vec2 a) {
        return fract(sin(dot(a, vec2(27.69f, 32.58f))) * 43758.53f);
    }

    @Reflect public static float box(vec2 p, vec2 b) {
        vec2 d = sub(abs(p), b);
        return length(vec2.max(d, 0f)) + min(max(d.x(), d.y()), 0f);
    }

    @Reflect public  static vec2 pattern(vec2 p, float sc) {
        mat2 r90 = rot(1.5707f);
        vec2 id = floor(mul(p, sc));
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

    @Reflect
    public static vec4 mainImageTruchet(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec3 R = uniforms.iResolution();
        vec2 fres = vec2(R.x(),R.y());
        float fTime = uniforms.iTime();
        vec3 color = vec3(0f);
        var uv = div(sub(mul(2f,fragCoord),fres),max(fres.x(),fres.y()));
        uv = vec2(log(length(uv)), F32.atan(uv.y(), uv.x()) * 6f / (PIx2));

        float scale = 8f;
        for (float i = 0f; i < 4f; i++) {
            float ff = (i * .05f) + .2f;
            uv = add(uv, vec2(fTime * ff, 0f));
            float fwidth = 0.0001f;
            float px = fwidth;//fwidth(uv.x*scale);
            vec2 d = pattern(uv, scale);
            vec3 clr = hue(sin(uv.x() + (i * 8f)) * .2f + .4f, (.5f + i) * .15f);
            color = mix(color, vec3(.001f), smoothstep(px, -px, d.y() - .04f));
            color = mix(color, clr, smoothstep(px, -px, d.x()));
            scale *= .5f;
        }
        // Output to screen
        color = pow(color, vec3(.4545f));
        return normalize(vec4(color, 1.0f));
    }

    @Reflect public static vec3 palette(float t) {
        vec3 a = vec3(0.5f, 0.5f, 0.5f);
        vec3 b = vec3(0.5f, 0.5f, 0.5f);
        vec3 c = vec3(1.0f, 1.0f, 1.0f);
        vec3 d = vec3(0.263f, 0.416f, 0.557f);
        return add(a, mul(b, cos(mul(add(mul(c, vec3(t)), d), vec3(6.28318f)))));
    }

    @Reflect
    static public vec4 mainImageTutorial(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fResolution = vec2(uniforms.iResolution().x(),uniforms.iResolution().y());
        float fTime = uniforms.iTime();
        vec2 uv = div(sub(mul(fragCoord, 2f), fResolution), fResolution.y());
        vec2 uv0 = uv;
        vec3 color = vec3(0f);
        for (float i = 0f; i < 4f; i++) {
            var uv1_5 = mul(uv, 1.5f);
            var f = fract(uv1_5);
            uv = sub(f, vec2(0.5f));
           vec3 col = palette(length(uv0) + i * .4f + fTime * .4f);
            float d = length(uv) * exp(-length(uv0));
            d = sin(d * 8f + fTime) / 8f;
            d = abs(d);
            d = pow(0.01f / d, 1.2f);
            color = add(color, mul(col, d));
        }

        fragColor = vec4(color, 1.0f);
        return normalize(fragColor);
    }

    @Reflect
    public static void penumbra(KernelContext kc, Uniforms uniforms, F32Array image) {
        if (kc.gix < kc.gsx) {
            vec3 fres =  uniforms.iResolution();
            int width = (int) fres.x();
        //    int height = (int) fres.y();
            int x= kc.gix % width;
            int y= kc.gix / width;
            int offsetx = kc.gix*3;
            int offsety =offsetx+1;
            int offsetz=offsety+1;
            var fragCoord = vec2(x,y);
            var fragColor = vec4(image.array(offsetx), image.array(offsety), image.array(offsetz),0f);
           fragColor = mainImageTruchet(uniforms, fragColor, fragCoord);
           // fragColor = mainImageTutorial(uniforms, fragColor, fragCoord);
            image.array(offsetx, fragColor.x());
            image.array(offsety, fragColor.y());
            image.array(offsetz, fragColor.z());
        }
    }


    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {
        computeContext.dispatchKernel(
                NDRange.of1D(width * height),               //0..S32Array2D.size()
                (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }
}
