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
import hat.types.vec2;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import hat.buffer.Uniforms;
import shade.ShaderViewer;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.abs;
import static hat.types.F32.floor;
import static hat.types.F32.log;
import static hat.types.F32.max;
import static hat.types.F32.mod;
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
import static hat.types.vec4.add;
import static hat.types.vec4.cos;
import static hat.types.vec4.div;
import static hat.types.vec4.mul;
import static hat.types.vec4.normalize;
import static hat.types.vec4.smoothstep;
import static hat.types.vec4.vec4;


//https://shadertoy.com/view/3llcDl
public class SpiralShader{

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


    @Reflect
    public static vec4 createPixel(vec2 fres, float ftime, vec2 fmouse, vec2 fragCoord){

            // variant of https://shadertoy.com/view/3llcDl
// inspired by https://www.facebook.com/eric.wenger.547/videos/2727028317526304/


            vec2 U = div(sub(mul(fragCoord, 2f), fres), fres.y());//.sub(fResolution).div(fResolution.y());
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
                            vec2(ftime / 8f)
                    ),
                    mul(
                            F32.atan(U.x(), U.y())/ 6.2832f,
                            vec2(6f, 1f)
                    )
            );


            U = div(mul(U, vec2(3f)), vec2(2f, 1f));
            z = vec2(.001f);//fwidth(U); // this resamples the image.  Not sure how we do this!
            U = mul(fract(U), 5f);
            vec2 I = floor(U);
            U = fract(U);             // subdiv big square in 5x5
            I = vec2(mod(I.x() - 2.f * I.y(), 5f), I.y());                            // rearrange
            U = add(U, vec2((I.x() == 1f || I.x() == 3f) ? 1f : 0f, I.x() < 2.0 ? 1f : 0f));     // recombine big tiles
            float id = -1f;
            if (I.x() != 4f) {
                U = div(U, 2f);                                     // but small times
                id = mod(floor(I.x() / 2f) + I.y(), 5f);
            }
            U = sub(abs(mul(fract(U), 2f)), 1f);
            float v = max(U.x(), U.y());          // dist to border

            return
                    normalize(
                            smoothstep(
                                    vec4(.7f),
                                    vec4(-.7f),
                                    mul(div(vec4(v - .95f), abs(z.x() - z.y()) > 1f
                                                    ? .1f
                                                    : z.y() * 8f
                                            )
                                            , id < 0f
                                                    ? vec4(1f)
                                                    : add(mul(vec4(.6f), .6f),
                                                    cos(add(vec4(id), vec4(0f, 23f, 21f, 0f)))
                                            )
                                    )
                            )
                    );// color
        }
    @Reflect public static vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord){
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
        var shader = ShaderViewer.of(acc, SpiralShader.class,1024, 1024);
        shader.startLoop((uniforms, f32Array) -> update( acc, uniforms, f32Array, shader.view.getWidth(), shader.view.getHeight()));
    }
}
