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
package hat.types;

public interface F32 {
    float PI = (float)Math.PI;
    float PIx2 = PI*2;
    float E = (float)Math.E;
    static float sqrt(float f){return (float)Math.sqrt(f);}
    static float inversesqrt(float f){return 1f/sqrt(f);}
    static float cos(float f){return (float)Math.cos(f);}
    static float sin(float f){return (float)Math.sin(f);}
    static float tan(float f){return (float)Math.tan(f);}
    static float atan(float f){return (float)Math.atan(f);}
    static float atan(float l,float r){return (float)Math.atan2(l,r);}
    static float floor(float f){return (float)Math.floor(f);}
    static float mix(float x, float y, float a){return x*(1f-a)+y*a; }
    static float pow(float x, float pow){return (float)Math.pow(x,pow); }
    static float smoothstep(float edge0, float edge1, float x ){
        float t = (edge1>edge0)
                ? Math.clamp((x - edge1) / (edge0 - edge1), 0f, 1f)
                : Math.clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
        return t * t * (3.0f - 2.0f * t);

    }
    static float step(float edge, float x ){
        return x<edge?0f:1f;
    }
    static float abs(float f){return Math.abs(f);}
    static float mod(float x, float y){
        return x - y * F32.floor(x/y);
    }
    static float fract(float f) {
        return f - floor(f);
    }
    static float exp(float f) {return (float)Math.exp(f);}
    static float log(float f) {return (float)Math.log(f);}
    static float min(float lhs, float rhs){ return Math.min(lhs,rhs);}
    static float max(float lhs, float rhs){ return Math.max(lhs,rhs);}
    // watch out ! Shader version is min,max, value!!
    static float clamp(float value,float min, float max){return Math.clamp(value,min,max);}

    static float  mul(float lhs, float rhs) {
        return lhs*rhs;
    }
    static float  add(float lhs, float rhs) {
        return lhs+rhs;
    }
    static float  sub(float lhs, float rhs) {
        return lhs-rhs;
    }
    static float  div(float lhs, float rhs) {
        return lhs/rhs;
    }
    static float neg(float f) {return -f;}
    static float round(float f) {return Math.round(f);}
}
