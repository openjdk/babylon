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
import hat.types.F32;
import hat.types.mat2;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Config;
import shade.Shader;
import shade.ShaderApp;
import shade.Uniforms;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.PI;
import static hat.types.F32.PIx2;
import static hat.types.F32.abs;
import static hat.types.F32.cos;
import static hat.types.F32.fract;
import static hat.types.F32.log;
import static hat.types.F32.max;
import static hat.types.F32.min;
import static hat.types.F32.pow;
import static hat.types.F32.sin;
import static hat.types.F32.smoothstep;
import static hat.types.mat2.mat2;

import  hat.types.mat3;
import static hat.types.mat3.mat3;
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
import static hat.types.vec3.neg;
import static hat.types.vec3.vec3;
import static hat.types.vec4.normalize;
import static hat.types.vec4.vec4;
// https://www.shadertoy.com/view/ldfGWn
public class Truchet2Shader implements Shader {
    static String glslSource = """
            float rand(vec3 r) { return fract(sin(dot(r.xy,vec2(1.38984*sin(r.z),1.13233*cos(r.z))))*653758.5453); }

               #define Iterations 64
               #define Thickness 0.1
               #define SuperQuadPower 8.0
               #define Fisheye 0.5

               float truchetarc(vec3 pos)
               {
                  float r=length(pos.xy);
               //   return max(abs(r-0.5),abs(pos.z-0.5))-Thickness;
               //   return length(vec2(r-0.5,pos.z-0.5))-Thickness;
                  return pow(pow(abs(r-0.5),SuperQuadPower)+pow(abs(pos.z-0.5),SuperQuadPower),1.0/SuperQuadPower)-Thickness;
               }

               float truchetcell(vec3 pos)
               {
                  return min(min(
                  truchetarc(pos),
                  truchetarc(vec3(pos.z,1.0-pos.x,pos.y))),
                  truchetarc(vec3(1.0-pos.y,1.0-pos.z,pos.x)));
               }

               float distfunc(vec3 pos)
               {
                  vec3 cellpos=fract(pos);
                  vec3 gridpos=floor(pos);

                  float rnd=rand(gridpos);

                  if(rnd<1.0/8.0) return truchetcell(vec3(cellpos.x,cellpos.y,cellpos.z));
                  else if(rnd<2.0/8.0) return truchetcell(vec3(cellpos.x,1.0-cellpos.y,cellpos.z));
                  else if(rnd<3.0/8.0) return truchetcell(vec3(1.0-cellpos.x,cellpos.y,cellpos.z));
                  else if(rnd<4.0/8.0) return truchetcell(vec3(1.0-cellpos.x,1.0-cellpos.y,cellpos.z));
                  else if(rnd<5.0/8.0) return truchetcell(vec3(cellpos.y,cellpos.x,1.0-cellpos.z));
                  else if(rnd<6.0/8.0) return truchetcell(vec3(cellpos.y,1.0-cellpos.x,1.0-cellpos.z));
                  else if(rnd<7.0/8.0) return truchetcell(vec3(1.0-cellpos.y,cellpos.x,1.0-cellpos.z));
                  else  return truchetcell(vec3(1.0-cellpos.y,1.0-cellpos.x,1.0-cellpos.z));
               }

               vec3 gradient(vec3 pos)
               {
                  const float eps=0.0001;
                  float mid=distfunc(pos);
                  return vec3(
                  distfunc(pos+vec3(eps,0.0,0.0))-mid,
                  distfunc(pos+vec3(0.0,eps,0.0))-mid,
                  distfunc(pos+vec3(0.0,0.0,eps))-mid);
               }

               void mainVR( out vec4 fragColor, in vec2 fragCoord, in vec3 fragRayOri, in vec3 fragRayDir )
               {
                   vec3 ray_dir=fragRayDir;
                  vec3 ray_pos=fragRayOri;

                  float i=float(Iterations);
                  for(int j=0;j<Iterations;j++)
                  {
                     float dist=distfunc(ray_pos);
                     ray_pos+=dist*ray_dir;

                     if(abs(dist)<0.001) { i=float(j); break; }
                  }

                  vec3 normal=normalize(gradient(ray_pos));

                  float ao=1.0-i/float(Iterations);
                  float what=pow(max(0.0,dot(normal,-ray_dir)),2.0);
                  float light=ao*what*1.4;

                  float z=ray_pos.z/2.0;
               //   vec3 col=(sin(vec3(z,z+pi/3.0,z+pi*2.0/3.0))+2.0)/3.0;
                  vec3 col=(cos(ray_pos/2.0)+2.0)/3.0;

                  vec3 reflected=reflect(ray_dir,normal);
                  vec3 env=texture(iChannel0,reflected*reflected*reflected).xyz;

                  fragColor=vec4(col*light+0.1*env,1.0);
               }

               void mainImage( out vec4 fragColor, in vec2 fragCoord )
               {
                  const float pi=3.141592;

                  vec2 coords=(2.0*fragCoord.xy-iResolution.xy)/length(iResolution.xy);

                  float a=iTime/3.0;
                  mat3 m=mat3(
                  0.0,1.0,0.0,
                  -sin(a),0.0,cos(a),
                  cos(a),0.0,sin(a));
                  m*=m;
                  m*=m;

               //   vec3 ray_dir=m*normalize(vec3(1.4*coords,-1.0+Fisheye*(coords.x*coords.x+coords.y*coords.y)));
                  vec3 ray_dir=m*normalize(vec3(2.0*coords,-1.0+dot(coords,coords)));


                  float t=iTime/3.0;
                  vec3 ray_pos=vec3(
                   2.0*(sin(t+sin(2.0*t)/2.0)/2.0+0.5),
                   2.0*(sin(t-sin(2.0*t)/2.0-pi/2.0)/2.0+0.5),
                   2.0*((-2.0*(t-sin(4.0*t)/4.0)/pi)+0.5+0.5));

                   mainVR(fragColor,fragCoord,ray_pos,ray_dir);

                     float vignette=pow(1.0-length(coords),0.3);
                  fragColor.xyz*=vec3(vignette);
               }

            """;
      static final int  Iterations=64;
      static final float Thickness=0.1f;
    static final float  SuperQuadPower=8f;
    static final float   Fisheye =5f;
    float rand(vec3 r) { return fract(sin(dot(vec2(r.x(),r.y()),vec2(1.38984f*sin(r.z()),1.13233f*cos(r.z()))))*653758.5453f); }
    float truchetarc(vec3 pos) {
        float r=length(vec2(pos.x(), pos.y()));
        return pow(pow(abs(r-.5f),SuperQuadPower)+pow(abs(pos.z()-0.5f),SuperQuadPower),1.0f/SuperQuadPower)-Thickness;
    }

    float truchetcell(vec3 pos) {
        return min(min(
                        truchetarc(pos),
                        truchetarc(vec3(pos.z(),1.0f-pos.x(),pos.y()))),
                truchetarc(vec3(1.0f-pos.y(),1.0f-pos.z(),pos.x())));
    }

    float distfunc(vec3 pos) {
        vec3 cellpos=vec3.fract(pos);
        vec3 gridpos=vec3.floor(pos);

        float rnd=rand(gridpos);

        if(rnd<1.0f/8.0f) return truchetcell(vec3(cellpos.x(),cellpos.y(),cellpos.z()));
        else if(rnd<2.0f/8.0f) return truchetcell(vec3(cellpos.x(),1.0f-cellpos.y(),cellpos.z()));
        else if(rnd<3.0f/8.0f) return truchetcell(vec3(1.0f-cellpos.x(),cellpos.y(),cellpos.z()));
        else if(rnd<4.0f/8.0f) return truchetcell(vec3(1.0f-cellpos.x(),1.0f-cellpos.y(),cellpos.z()));
        else if(rnd<5.0f/8.0f) return truchetcell(vec3(cellpos.y(),cellpos.x(),1.0f-cellpos.z()));
        else if(rnd<6.0f/8.0f) return truchetcell(vec3(cellpos.y(),1.0f-cellpos.x(),1.0f-cellpos.z()));
        else if(rnd<7.0f/8.0f) return truchetcell(vec3(1.0f-cellpos.y(),cellpos.x(),1.0f-cellpos.z()));
        else  return truchetcell(vec3(1.0f-cellpos.y(),1.0f-cellpos.x(),1.0f-cellpos.z()));
    }

    vec3 gradient(vec3 pos) {
                   float eps=0.0001f;
        float mid=distfunc(pos);
        return vec3(
                distfunc(add(pos,vec3(eps,0.0f,0.0f)))-mid,
                distfunc(add(pos,vec3(0.0f,eps,0.0f)))-mid,
                distfunc(add(pos,vec3(0.0f,0.0f,eps)))-mid);
    }

    vec4 mainVR( vec4 fragColor,  vec2 fragCoord,  vec3 fragRayOri,  vec3 fragRayDir )
    {
        vec3 ray_dir=fragRayDir;
        vec3 ray_pos=fragRayOri;

        float i=(float)Iterations;
        for(int j=0;j<Iterations;j++) {
            float dist=distfunc(ray_pos);
            ray_pos = add(ray_pos,mul(dist,ray_dir));

            if(abs(dist)<0.001f) { i=(float)j; break; }
        }

        vec3 normal=vec3.normalize(gradient(ray_pos));

        float ao=1.0f-i/(float)Iterations;
     //  float what=pow(max(0.0,dot(normal,-ray_dir)),2.0);
        var term = vec3.dot(normal,neg(ray_dir));
        float what=F32.pow(F32.max(0.0f,term),2.0f);
        float light=ao*what*1.4f;

        float z=ray_pos.z()/2.0f;
        //   vec3 col=(sin(vec3(z,z+pi/3.0,z+pi*2.0/3.0))+2.0)/3.0;
         //   vec3 col=(cos(ray_pos/2.0)+2.0)/3.0;
        var rayPosDiv2 = vec3.div(ray_pos,2f);
        var cosRayPosDiv2Plus2 = add(vec3.cos(rayPosDiv2),2f);
        var col = vec3.div(cosRayPosDiv2Plus2,3f);
        //(cos(vec3.add(vec3.div(ray_pos,2.0f)),2.0f))/3.0f;

        vec3 reflected=vec3.reflect(ray_dir,normal);
       // vec3 env=texture(iChannel0,reflected*reflected*reflected).xyz;
        vec3 env=mul(mul(mul(50f,reflected),reflected),reflected);//texture(iChannel0,reflected*reflected*reflected).xyz;

        return normalize(vec4(add(mul(col,light),mul(0.1f,env)),1.0f));
    }
    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fres = vec2(uniforms.iResolution().x(),uniforms.iResolution().y());
        //    vec2 coords=(2.0*fragCoord.xy-iResolution.xy)/length(iResolution.xy);
        //  //    vec2 coords=(2FragCoord-fres)/length(fres);
        //vec2 coords=vec2.div(vec2.sub(vec2.mul(2.0f,fragCoord),fres)),length(fres));
        var twoFragCoord = mul(2f,fragCoord);
        vec2 coords = div(sub(twoFragCoord,fres),length(fres));
        float a=uniforms.iTime()/3.0f;
        mat3 m=mat3(
                0.0f,1.0f,0.0f,
                -sin(a),0.0f,cos(a),
                cos(a),0.0f,sin(a));
        m=mat3.mul(m,m);
        m=mat3.mul(m,m);

        //   vec3 ray_dir=m*normalize(vec3(1.4*coords,-1.0+Fisheye*(coords.x*coords.x+coords.y*coords.y)));
        //   vec3 ray_dir=m*normalize(vec3(2.0*coords,-1.0+dot(coords,coords)));
        var coordMul2 = mul(2f,coords);
        var m1PlusDotCoords  = -1f+dot(coords,coords);
        var v3 = vec3(coordMul2,m1PlusDotCoords);
        var n = vec3.normalize(v3);

        vec3 ray_dir=vec3.mul(n,m);


        float t=uniforms.iTime()/3.0f;
        vec3 ray_pos=vec3(
                2.0f*(sin(t+sin(2.0f*t)/2.0f)/2.0f+0.5f),
                2.0f*(sin(t-sin(2.0f*t)/2.0f-PI/2.0f)/2.0f+0.5f),
                2.0f*((-2.0f*(t-sin(4.0f*t)/4.0f)/PI)+0.5f+0.5f));

        fragColor = mainVR(fragColor,fragCoord,ray_pos,ray_dir);

        float vignette=pow(1.0f-length(coords),0.3f);

        fragColor=vec4(mul(vec3(fragColor.x(),fragCoord.y(),fragColor.z()),vignette),1f);

        return normalize(fragColor);
    }

    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "260"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "260"))),
            Integer.parseInt(System.getProperty("targetFps", "4")),
            new Truchet2Shader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }

}

