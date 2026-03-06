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
@Reflect public static
    float calc(Uniforms uniforms, vec2 p, float time ) {
        // non p dependent
        float ltime = 0.5f-0.5f* F32.cos(time*0.06f);
        float zoom = F32.pow( 0.9f, 50.0f*ltime );
        vec2  cen = add(vec2.vec2( 0.2655f,0.301f ), zoom*0.8f*F32.cos(4.0f+2.0f*ltime));

        vec2 c = sub(vec2.vec2( -0.745f, 0.186f ) , 0.045f*zoom*(1.0f-ltime*0.5f));
        vec2 fres  = vec2.vec2(uniforms.iResolution().x(), uniforms.iResolution().y());
/*
   p = (2.0*p-iResolution.xy)/iResolution.y;
   vec2 z = cen + (p-cen)*zoom;
 */
        p = div(sub(mul(2.0f,p), fres),fres.y());
        //
        // p = vec2.div(vec2.sub(vec2.mul(2.0f,p),vec2.vec2(uniforms.iResolution().x(),uniforms.iResolution().y()))/uniforms.iResolution().y();
        vec2 z = add(cen, mul(sub(p,cen),zoom));


        // full derivatives version
        vec2 dz = vec2.vec2( 1.0f, 0.0f );
        int brk=0;
        for( int i=0; brk==0 && i<256; i++ ) {
            dz = mul(2.0f,vec2.vec2(z.x()*dz.x()-z.y()*dz.y(), z.x()*dz.y() + z.y()*dz.x() ));
            z = add(vec2.vec2( z.x()*z.x() - z.y()*z.y(), 2.0f*z.x()*z.y() ), c);
            if( dot(z,z)>200.0f ){
                brk=1;
            }
        }
        float d = F32.sqrt( dot(z,z)/dot(dz,dz) )*F32.log(dot(z,z));



        return F32.sqrt( clamp( (150.0f/zoom)*d, 0.0f, 1.0f ) );
    }

    //    https://www.shadertoy.com/view/Mss3R8
    @Reflect
    public static vec4 mainImageJulia(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {

        final int AA = 2;
        float scol = 0.0f;
        for( int j=0; j<AA; j++ )
            for( int i=0; i<AA; i++ ) {
                vec2 of = add(-0.5f, div(vec2.vec2( (float)i, (float)j ),(float)AA));
                var c = vec2.add(fragCoord,of);
                scol += calc(uniforms, c, (float)uniforms.iTime() );
            }
        scol/=(float)AA*AA;


        vec3 vcol = vec3.pow( vec3.vec3(scol), vec3.vec3(0.9f,1.1f,1.4f) );
        vec2 fres  = vec2.vec2(uniforms.iResolution().x(), uniforms.iResolution().y());
        vec2 uv = div(fragCoord,fres);//iResolution.xy;
        //  vcol *= 0.7 + 0.3*pow(16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y),0.25);
        var p = F32.pow(16.0f*uv.x()*uv.y()*(1.0f-uv.x())*(1.0f-uv.y()),0.2f);
        vcol = vec3.mul(vcol,0.7f + 0.3f*p);


        fragColor = vec4( vcol, 1.0f );
        return fragColor;
    }


    /*
  vec2 ortho(vec2 v)
         {
             return vec2(v.y, -v.x);
         }
  */
    @Reflect static
    public vec2 ortho(vec2 v) {
        return vec2(v.y(), -v.x());
    }

    /*
    void stroke(float dist, vec3 color, inout vec3 fragColor, float thickness, float aa)
            {
                float alpha = smoothstep(0.5 * (thickness + aa), 0.5 * (thickness - aa), abs(dist));
                fragColor = mix(fragColor, color, alpha);
            }
     */
    @Reflect static
    public vec3 stroke(float dist, vec3 color, vec3 fragColor, float thickness, float aa) {
        float alpha = smoothstep(0.5f * (thickness + aa), 0.5f * (thickness - aa), abs(dist));
        return mix(fragColor, color, alpha);
    }
    /*

            void fill(float dist, vec3 color, inout vec3 fragColor, float aa)
            {
                float alpha = smoothstep(0.5*aa, -0.5*aa, dist);
                fragColor = mix(fragColor, color, alpha);
            }

     */

    @Reflect static
    public  vec3 fill(float dist, vec3 color, vec3 fragColor, float aa) {
        float alpha = smoothstep(0.5f * aa, -0.5f * aa, dist);
        return mix(fragColor, color, alpha);
    }

    /*
    void renderGrid(vec2 pos, out vec3 fragColor)
            {
                vec3 background = vec3(1.0);
                vec3 axes = vec3(0.4);
                vec3 lines = vec3(0.7);
                vec3 sublines = vec3(0.95);
                float subdiv = 10.0;

                float thickness = 0.003;
                float aa = length(fwidth(pos));

                fragColor = background;

                vec2 toSubGrid = pos - round(pos*subdiv)/subdiv;
                stroke(min(abs(toSubGrid.x), abs(toSubGrid.y)), sublines, fragColor, thickness, aa);

                vec2 toGrid = pos - round(pos);
                stroke(min(abs(toGrid.x), abs(toGrid.y)), lines, fragColor, thickness, aa);

                stroke(min(abs(pos.x), abs(pos.y)), axes, fragColor, thickness, aa);
            }
     */
    @Reflect static
    public  void renderGrid(vec2 pos, vec3 fragColor) {
        vec3 background = vec3(1.0f);
        vec3 axes = vec3(0.4f);
        vec3 lines = vec3(0.7f);
        vec3 sublines = vec3(0.95f);
        float subdiv = 10.0f;

        float thickness = 0.003f;
        float fwidthPos = 0.01f;
        float aa = fwidthPos;//?length(fwidthPos);

        fragColor = background;

        vec2 toSubGrid = sub(pos, div(vec2.round(mul(pos, subdiv)), subdiv));
        stroke(min(abs(toSubGrid.x()), abs(toSubGrid.y())), sublines, fragColor, thickness, aa);

        vec2 toGrid = sub(pos, round(pos));
        stroke(min(abs(toGrid.x()), abs(toGrid.y())), lines, fragColor, thickness, aa);

        stroke(min(abs(pos.x()), abs(pos.y())), axes, fragColor, thickness, aa);
    }

    /*
    float sdistLine(vec2 a, vec2 b, vec2 pos)
                {
                    return dot(pos - a, normalize(ortho(b - a)));
                }
    */
    @Reflect static
    public  float sdistLine(vec2 a, vec2 b, vec2 pos) {
        return dot(sub(pos, a), vec2.normalize(ortho(sub(b, a))));
    }
    /*
            float sdistTri(vec2 a, vec2 b, vec2 c, vec2 pos)
            {
                return max( sdistLine(a, b, pos),
                        max(sdistLine(b, c, pos),
                            sdistLine(c, a, pos)));
            }
 */

    @Reflect static
    public float sdistTri(vec2 a, vec2 b, vec2 c, vec2 pos) {
        return max(sdistLine(a, b, pos),
                max(sdistLine(b, c, pos),
                        sdistLine(c, a, pos)));
    }

    /*
    float sdistQuadConvex(vec2 a, vec2 b, vec2 c, vec2 d, vec2 pos)
            {
                return max(  sdistLine(a, b, pos),
                        max( sdistLine(b, c, pos),
                         max(sdistLine(c, d, pos),
                             sdistLine(d, a, pos))));
            }
     */
    @Reflect static
    public float sdistQuadConvex(vec2 a, vec2 b, vec2 c, vec2 d, vec2 pos) {
        return max(sdistLine(a, b, pos),
                max(sdistLine(b, c, pos),
                        max(sdistLine(c, d, pos),
                                sdistLine(d, a, pos))));
    }

    /*
    void renderUnitSquare(vec2 pos, inout vec3 fragColor)
            {
            #if 0
                // Put a texture in there
                if (pos.x >= 0.0 && pos.y >= 0.0 && pos.x <= 1.0 && pos.y <= 1.0)
                {
                    fragColor.rgb = texture(iChannel0, pos).rgb;
                }
            #endif

                float dist = sdistQuadConvex(vec2(0, 0),
                                             vec2(1, 0),
                                             vec2(1, 1),
                                             vec2(0, 1), pos);
                stroke(dist, vec3(0, 0, 1), fragColor, 0.007, length(fwidth(pos)));
            }
     */
    @Reflect static
    public vec3 renderUnitSquare(vec2 pos, vec3 fragColor) {

        float dist = sdistQuadConvex(vec2(0, 0),
                vec2(1, 0),
                vec2(1, 1),
                vec2(0, 1), pos);
        float fwidthPos = .0f;
        return stroke(dist, vec3(0, 0, 1), fragColor, 0.007f, fwidthPos/*length(fwidth(pos)*/);
    }

    /*
    void renderAxes(vec2 origin, vec2 pos, inout vec3 fragColor)
            {
                float len = 0.1;
                float thickness = 0.0075;
                float aa = length(fwidth(pos));

                float xshaft = sdistQuadConvex(origin + vec2(0.5*thickness),
                                               origin - vec2(0.5*thickness),
                                               origin + vec2(len, -0.5*thickness),
                                               origin + vec2(len, 0.5*thickness), pos);
                float xhead = sdistTri(origin + vec2(len, -2.0*thickness),
                                       origin + vec2(len + 6.0*thickness, 0),
                                       origin + vec2(len, 2.0*thickness), pos);

                fill(min(xshaft, xhead), vec3(1, 0, 0), fragColor, aa);

                float yshaft = sdistQuadConvex(origin - vec2(0.5*thickness),
                                               origin + vec2(0.5*thickness),
                                               origin + vec2(0.5*thickness, len),
                                               origin + vec2(-0.5*thickness, len), pos);
                float yhead = sdistTri(origin + vec2(2.0*thickness, len),
                                       origin + vec2(0, len + 6.0*thickness),
                                       origin + vec2(-2.0*thickness, len), pos);

                fill(min(yshaft, yhead), vec3(0, 0.75, 0), fragColor, aa);

            }
     */
    @Reflect static
    public vec3 renderAxes(vec2 origin, vec2 pos, vec3 fragColor) {
        float len = 0.1f;
        float thickness = 0.0075f;
        float fwidthPos = 0.01f;
        float aa = fwidthPos;//length(fwidth(pos));

        float xshaft = sdistQuadConvex(add(origin, vec2(0.5f * thickness)),
                sub(origin, vec2(0.5f * thickness)),
                add(origin, vec2(len, -0.5f * thickness)),
                add(origin, vec2(len, 0.5f * thickness)), pos);

        float xhead = sdistTri(add(origin, vec2(len, -2.0f * thickness)),
                add(origin, vec2(len + 6.0f * thickness, 0f)),
                add(origin, vec2(len, 2.0f * thickness)), pos);

        fragColor = fill(min(xshaft, xhead), vec3(1f, 0f, 0f), fragColor, aa);

        float yshaft = sdistQuadConvex(add(origin, vec2(0.5f * thickness)),
                add(origin, vec2(0.5f * thickness)),
                add(origin, vec2(0.5f * thickness, len)),
                add(origin, vec2(-0.5f * thickness, len)), pos);

        float yhead = sdistTri(add(origin, vec2(2.0f * thickness, len)),
                add(origin, vec2(0, len + 6.0f * thickness)),
                add(origin, vec2(-2.0f * thickness, len)), pos);

        fragColor = fill(min(yshaft, yhead), vec3(0f, 0.75f, 0f), fragColor, aa);

        return fragColor;
    }

    /*
    vec2 cmul(vec2 a, vec2 b)
            {
                return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
            }
*/
    @Reflect static
    public vec2 cmul(vec2 a, vec2 b) {
        return vec2(a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x());
    }
    /*
            vec2 cdiv(vec2 a, vec2 b)
            {
                return cmul(a, vec2(b.x, -b.y)) / dot(b, b);
            }
     */

    @Reflect static
    public vec2 cdiv(vec2 a, vec2 b) {
        return div(cmul(a, vec2(b.x(), -b.y())), dot(b, b));
    }

    @Reflect static
    public vec4 mainImageMobius(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        fragColor = vec4(1f, 1f, 1f, 1f);
        float aspect = uniforms.iResolution().x() / uniforms.iResolution().y();
        vec2 pos = sub(mul(div(fragCoord, uniforms.iResolution().y()), 1.5f), vec2((1.5f * aspect - 1.0f) / 2.0f, 0.25f));

        // apply a Möbius transformation to the plane
        vec2 a = vec2(1f, sin(0.4f * uniforms.iTime()));
        vec2 b = vec2(0f);
        vec2 c = vec2(0.5f * cos(0.6f * uniforms.iTime()), 0.5f * sin(0.5f * uniforms.iTime()));
        vec2 d = vec2(1f, cos(0.3f * uniforms.iTime()));
        pos = sub(pos, vec2(0.5f));
        pos = cdiv(add(cmul(a, pos), b), add(cmul(c, pos), d));
        pos = add(pos, vec2(0.5f));

        // render the grid and stuff
        fragColor = vec4(fragColor.x(), fragColor.y(), fragColor.z(), 1.0f);

        renderGrid(pos, vec3(fragColor.x(),fragColor.y(),fragColor.z()));
        fragColor = vec4(renderUnitSquare(pos, vec3(fragColor.x(),fragColor.y(),fragColor.z())), 1f);
        fragColor = vec4(renderAxes(vec2(0f), pos, vec3(fragColor.x(),fragColor.y(),fragColor.z())), 1f);
        return normalize(fragColor);
    }





    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RW F32Array image) {
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
          //  fragColor = mainImageTutorial(uniforms, fragColor, fragCoord);
          //  fragColor = mainImageJulia(uniforms, fragColor, fragCoord);
        //    fragColor = mainImageMobius(uniforms, fragColor, fragCoord);
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
