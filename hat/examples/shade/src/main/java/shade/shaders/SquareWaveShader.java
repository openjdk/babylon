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
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Shader;
import shade.Uniforms;

import static hat.types.mat2.mat2;
import static hat.types.vec2.length;

//https://www.shadertoy.com/view/4tXSzM
public class SquareWaveShader implements Shader {
    String glsSource = """
            // evening sketch by @mmalex
            // having a go at re-creating @vector_gl's lovely gif https://twitter.com/Vector_GL/status/612337298064150529
            // now with triangle wave!

            #define moblur 4
            #define harmonic 25
            #define triangle 1 // comment this line out for only square

            vec3 circle(vec2 uv, float rr, float cc, float ss) {

                uv*=mat2(cc,ss,-ss,cc);
                if (rr<0.) uv.y=-uv.y;
                rr=abs(rr);
                float r = length(uv)-rr;
                float pix=fwidth(r);
                float c = smoothstep(0.,pix,abs(r));
                float l = smoothstep(0.,pix,abs(uv.x)+step(uv.y,0.)+step(rr,uv.y));
                   return vec3(c,c*l,c*l);
            }
            vec3 ima(vec2 uv, float th0) {
                vec3 col=vec3(1.0);
                vec2 uv0=uv;
                   th0-=max(0.,uv0.x-1.5)*2.;
                   th0-=max(0.,uv0.y-1.5)*2.;
            #ifndef triangle
            float lerpy = 1.;
            #else
            float lerpy =smoothstep(-0.6,0.2,cos(th0*0.1));
            #endif

                for (int i=1;i<harmonic;i+=2) {
                    float th=th0*float(i);
                    float fl=mod(float(i),4.)-2.;// used to be repeated assignment fl=-fl, but compiler bugs. :(
                    float cc=cos(th)*fl,ss=sin(th);
                    float trir=-fl/float(i*i);
                    float sqrr=1./float(i);
                    float rr=mix(trir,sqrr,lerpy);
                    col = min(col, circle(uv,rr,cc,ss));
                    uv.x+=rr*ss;
                    uv.y-=rr*cc;
                }
                float pix=fwidth(uv0.x);
                if (uv.y>0. && fract(uv0.y*10.)<0.5) col.yz=min(col.yz,smoothstep(0.,pix,abs(uv.x)));
                if (uv.x>0. && fract(uv0.x*10.)<0.5) col.yz=min(col.yz,smoothstep(0.,pix,abs(uv.y)));
                if (uv0.x>=1.5) col.xy=vec2(smoothstep(0.,fwidth(uv.y),abs(uv.y)));
                if (uv0.y>=1.5) col.xy=vec2(smoothstep(0.,fwidth(uv.x),abs(uv.x)));
                return col;
            }
            void mainImage( out vec4 fragColor, in vec2 fragCoord )
            {
                vec2 uv = fragCoord.xy / iResolution.yy;
                uv.y=1.-uv.y;
                uv*=5.;
                uv-=1.5;
                float th0=iTime*2.;
                float dt=2./60./float(moblur);
                vec3 col=vec3(0.);
                for (int mb=0;mb<moblur;++mb) {
                    col+=ima(uv,th0);
                    th0+=dt;
                }
                col=pow(col*(1./float(moblur)),vec3(1./2.2));
                fragColor=vec4(col,1.);
            }
            """;
    static final int  moblur=4;
    static final int harmonic=25;
            static final int triangle=1; // comment this line out for only square

    vec3 circle(vec2 uv, float rr, float cc, float ss) {

        uv = vec2.mul(uv,mat2(cc,ss,-ss,cc));
        if (rr<0f){
            uv = vec2.vec2(uv.x(),-uv.y());
        }
        rr= F32.abs(rr);
        float r = length(uv)-rr;
        float pix=1f;// fwidth(r);
        float c = F32.smoothstep(0f,pix,F32.abs(r));
        float l = F32.smoothstep(0f,pix,F32.abs(uv.x())+F32.step(uv.y(),0f)+F32.step(rr,uv.y()));
        return vec3.vec3(c,c*l,c*l);
    }

    vec3 ima(vec2 uv, float th0) {

        vec2 uv0=uv;
        th0-=F32.max(0f,uv0.x()-1.5f)*2f;
        th0-=F32.max(0f,uv0.y()-1.5f)*2f;

        float lerpy = triangle==1?1f :F32.smoothstep(-0.6f,0.2f,F32.cos(th0*0.1f));
        vec3 col=vec3.vec3(1f);
        for (int i=1;i<harmonic;i+=2) {
            float th=th0*(float)i;
            float fl=F32.mod((float)i,4f)-2f;// used to be repeated assignment fl=-fl, but compiler bugs. :(
            float cc=F32.cos(th)*fl,ss=F32.sin(th);
            float trir=-fl/(float)i*i;
            float sqrr=1f/(float)i;
            float rr=F32.mix(trir,sqrr,lerpy);
            col = vec3.min(col, circle(uv,rr,cc,ss));
            uv = vec2.add(uv, vec2.vec2(rr*ss, -rr*cc));
        }
        float pix=.1f;//fwidth(uv0.x);

        /*
          if (uv.y>0. && fract(uv0.y*10.)<0.5) col.yz=min(col.yz,smoothstep(0.,pix,abs(uv.x)));
          if (uv.x>0. && fract(uv0.x*10.)<0.5) col.yz=min(col.yz,smoothstep(0.,pix,abs(uv.y)));
          if (uv0.x>=1.5) col.xy=vec2(smoothstep(0.,fwidth(uv.y),abs(uv.y)));
          if (uv0.y>=1.5) col.xy=vec2(smoothstep(0.,fwidth(uv.x),abs(uv.x)));
         */
        if (uv.y()>0f && F32.fract(uv0.y()*10f)<0.5f){
            vec2 yz = vec2.min(vec2.vec2(col.y(), col.z()),vec2.vec2(F32.smoothstep(0f,pix,F32.abs(uv.x()))));
            col = vec3.vec3(0f, yz.x(),yz.y());
        }
        if (uv.x()>0f && F32.fract(uv0.x()*10f)<0.5f){
            vec2 yz = vec2.max(vec2.vec2(col.y(), col.z()),vec2.vec2(F32.smoothstep(0f,pix,F32.abs(uv.x()))));
            col = vec3.vec3(0f, yz.x(),yz.y());
        }
        if (uv0.x()>=1.5f){
            col=vec3.vec3(
                    vec2.vec2(F32.smoothstep(0f,/*fwidth(uv.y())*/.1f,F32.abs(uv.y()))),
                    col.z()
            );
        }
        if (uv0.y()>=1.5f){
            col=vec3.vec3(
                    vec2.vec2(F32.smoothstep(0f,/*fwidth(uv.x())*/.1f,F32.abs(uv.x()))), col.z()
            );
        }
        return vec3.normalize(col);
    }

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fres = vec2.vec2(uniforms.iResolution());
        float fTime = uniforms.iTime();
        vec2 uv = vec2.div(fragCoord,vec2.vec2(fres.y(),fres.y()));//iResolution.yy;
        uv = vec2.vec2(uv.x(),1f-uv.y());
        uv=vec2.mul (uv,5f);
        uv=vec2.sub(uv, vec2.vec2(1.5f));
        float th0=fTime*2f;
        float dt=2f/60f/(float)moblur;
        vec3 col=vec3.vec3(.1f);
        for (int mb=0;mb<moblur;++mb) {
            col= vec3.add(col,ima(uv,th0));
            th0+=dt;
        }
     //   col=vec3.pow(vec3.mul(col,(1f/(float)moblur)),vec3.vec3(1f/2.2f));
        fragColor=vec4.vec4(vec3.normalize(col),0f);
        return fragColor;
    }
}
