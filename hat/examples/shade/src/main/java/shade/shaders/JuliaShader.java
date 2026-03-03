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
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Config;
import shade.Shader;
import shade.ShaderApp;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.clamp;
import static hat.types.vec2.add;
import static hat.types.vec2.div;
import static hat.types.vec2.dot;
import static hat.types.vec2.mul;
import static hat.types.vec2.sub;
import static hat.types.vec4.vec4;

class JuliaShader implements Shader{
/*
// The MIT License
// Copyright © 2013 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Distance to a traditional Julia set for f(z)=z²+c

// More info here:
// https://iquilezles.org/articles/distancefractals

// Related:
//
// Julia - Distance 1 : https://www.shadertoy.com/view/Mss3R8
// Julia - Distance 2 : https://www.shadertoy.com/view/3llyzl
// Julia - Distance 3 : https://www.shadertoy.com/view/4dXGDX


#define AA 3

float calc( vec2 p, float time )
{
    // non p dependent
   float ltime = 0.5-0.5*cos(time*0.06);
    float zoom = pow( 0.9, 50.0*ltime );
   vec2  cen = vec2( 0.2655,0.301 ) + zoom*0.8*cos(4.0+2.0*ltime);

   vec2 c = vec2( -0.745, 0.186 ) - 0.045*zoom*(1.0-ltime*0.5);

    //
    p = (2.0*p-iResolution.xy)/iResolution.y;
   vec2 z = cen + (p-cen)*zoom;

#if 1
    // full derivatives version
   vec2 dz = vec2( 1.0, 0.0 );
   for( int i=0; i<256; i++ )
   {
      dz = 2.0*vec2(z.x*dz.x-z.y*dz.y, z.x*dz.y + z.y*dz.x );
        z = vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y ) + c;
      if( dot(z,z)>200.0 ) break;
   }
   float d = sqrt( dot(z,z)/dot(dz,dz) )*log(dot(z,z));

#else
    // only derivative length version
    float ld2 = 1.0;
    float lz2 = dot(z,z);
    for( int i=0; i<256; i++ )
   {
        ld2 *= 4.0*lz2;
        z = vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y ) + c;
        lz2 = dot(z,z);
      if( lz2>200.0 ) break;
   }
    float d = sqrt(lz2/ld2)*log(lz2);

#endif

   return sqrt( clamp( (150.0/zoom)*d, 0.0, 1.0 ) );
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
   #if 0
   float scol = calc( fragCoord, iTime );
    #else

    float scol = 0.0;
   for( int j=0; j<AA; j++ )
   for( int i=0; i<AA; i++ )
   {
      vec2 of = -0.5 + vec2( float(i), float(j) )/float(AA);
       scol += calc( fragCoord+of, iTime );
   }
   scol /= float(AA*AA);

    #endif

   vec3 vcol = pow( vec3(scol), vec3(0.9,1.1,1.4) );

   vec2 uv = fragCoord/iResolution.xy;
   vcol *= 0.7 + 0.3*pow(16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y),0.25);


   fragColor = vec4( vcol, 1.0 );
}
 */

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
        for( int i=0; i<256; i++ ) {
            dz = mul(2.0f,vec2.vec2(z.x()*dz.x()-z.y()*dz.y(), z.x()*dz.y() + z.y()*dz.x() ));
            z = add(vec2.vec2( z.x()*z.x() - z.y()*z.y(), 2.0f*z.x()*z.y() ), c);
            if( dot(z,z)>200.0f ) break;
        }
        float d = F32.sqrt( dot(z,z)/dot(dz,dz) )*F32.log(dot(z,z));



        return F32.sqrt( clamp( (150.0f/zoom)*d, 0.0f, 1.0f ) );
    }

//    https://www.shadertoy.com/view/Mss3R8
    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {

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

    ;
    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "512"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "512"))),
            Integer.parseInt(System.getProperty("targetFps", "6")),
            new JuliaShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }
}
