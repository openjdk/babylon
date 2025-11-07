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
package view.f32;

import hat.util.StreamMutable;

import java.util.ArrayList;
import java.util.List;

public class F32Mesh3D {
    String name;
    private F32Mesh3D(String name){
        this.name = name;
    }
    public static F32Mesh3D of(String name){
        return new F32Mesh3D(name);
    }
    public record Face ( F32Triangle3D triangle, F32Vec3 centerVec3Idx, F32Vec3 normalIdx, F32Vec3 v0VecIdx){
        static Face of (F32Triangle3D tri){
           return  new Face(tri,  F32Triangle3D.getCentre(tri),F32Triangle3D.normal(tri),tri.v0());
        }
    }

    public List<Face> faces = new ArrayList<>();

    public List<F32Vec3> vecEntries = new ArrayList<>();// F32Vec3.F32Vec3Pool.Idx[MAX];

    public Face tri(int v0Idx, int v1Idx, int v2Idx, int rgb) {
        Face face =Face.of(F32Triangle3D.f32Triangle3DPool.of(F32Vec3.f32Vec3Pool.idx(v0Idx), F32Vec3.f32Vec3Pool.idx(v1Idx), F32Vec3.f32Vec3Pool.idx(v2Idx), rgb));
        faces.add(face);
        return face;
    }


    public void fin(){
        var  triSumIdx = StreamMutable.of(0);
        var first  = StreamMutable.of(true);
        faces.forEach(face ->{
            if (first.get().equals(true)){
                triSumIdx.set(face.centerVec3Idx.idx());
                first.set(false);
            }else {
                triSumIdx.set(F32Vec3.addVec3(F32Vec3.f32Vec3Pool.idx(triSumIdx.get()), face.centerVec3Idx).idx());
            }
        });
        var meshCenterVec3 = F32Vec3.divScaler(F32Vec3.f32Vec3Pool.idx(triSumIdx.get()), faces.size());
        faces.forEach(face ->{
            var v0CenterDiff = F32Vec3.subVec3(meshCenterVec3,face.v0VecIdx);
            float normDotProd = F32Vec3.dotProd(v0CenterDiff, face.normalIdx);
            if (normDotProd >0f) { // the normal from the center from the triangle was pointing out, so re wind it
                F32Triangle3D.rewind(face.triangle);
            }
        });
        cube(meshCenterVec3.x(),meshCenterVec3.y(), meshCenterVec3.z(), .1f );
    }

    public F32Mesh3D quad(int v0, int v1, int v2, int v3, int rgb) {
  /*
       v0-----v1
        |\    |
        | \   |
        |  \  |
        |   \ |
        |    \|
       v3-----v2
   */

        tri(v0, v1, v2, rgb);
        tri(v0, v2, v3, rgb);
        return this;
    }

    public F32Mesh3D pent(int v0, int v1, int v2, int v3, int v4, int rgb) {
  /*
       v0-----v1
       |\    | \
       | \   |  \
       |  \  |   v2
       |   \ |  /
       |    \| /
       v4-----v3
   */

        tri(v0, v1, v3, rgb);
        tri(v1, v2, v3, rgb);
        tri(v0, v3, v4, rgb);
        return this;
    }
    public F32Mesh3D hex(int v0, int v1, int v2, int v3, int v4, int v5, int rgb) {
  /*
       v0-----v1
      / |\    | \
     /  | \   |  \
    v5  |  \  |   v2
     \  |   \ |  /
      \ |    \| /
       v4-----v3
   */

        tri(v0, v1, v3, rgb);
        tri(v1, v2, v3, rgb);
        tri(v0, v3, v4, rgb);
        tri(v0, v4, v5, rgb);
        return this;
    }


    /*
               a-----------d
              /|          /|
             / |         / |
           h------------g  |
           |   |        |  |
           |   b--------|--c
           |  /         | /
           | /          |/
           e------------f

     */


    public F32Mesh3D cube(
            float x,
            float y,
            float z,
            float s) {
        var a = vec3(x - (s * .5f), y - (s * .5f), z - (s * .5f));  //000  000 111 111
        var b = vec3(x - (s * .5f), y + (s * .5f), z - (s * .5f));  //010  010 101 101
        var c = vec3(x + (s * .5f), y + (s * .5f), z - (s * .5f));  //110  011 001 100
        var d = vec3(x + (s * .5f), y - (s * .5f), z - (s * .5f));  //100  001 011 110
        var e = vec3(x - (s * .5f), y + (s * .5f), z + (s * .5f));  //011  110 100 001
        var f = vec3(x + (s * .5f), y + (s * .5f), z + (s * .5f));  //111  111 000 000
        var g = vec3(x + (s * .5f), y - (s * .5f), z + (s * .5f));  //101  101 010 010
        var h = vec3(x - (s * .5f), y - (s * .5f), z + (s * .5f));  //001  100 110 011
        quad(a.idx(), b.idx(), c.idx(), d.idx(), 0xff0000); //front
        quad(b.idx(), e.idx(), f.idx(), c.idx(), 0x0000ff); //top
        quad(d.idx(), c.idx(), f.idx(), g.idx(), 0xffff00); //right
        quad(h.idx(), e.idx(), b.idx(), a.idx(), 0xffffff); //left
        quad(g.idx(), f.idx(), e.idx(), h.idx(), 0x00ff00);//back
        quad(g.idx(), h.idx(), a.idx(), d.idx(), 0xffa500);//bottom
        return this;
    }
    /*
http://paulbourke.net/dataformats/obj/

     */


    public F32Mesh3D cubeoctahedron(
            float x,
            float y,
            float z,
            float s) {

        var v1 = vec3(x + (s * .30631559f), y + (s * .20791225f), z + (s * .12760004f));
        var v2 = vec3(x + (s * .12671047f), y + (s * .20791227f), z + (s * .30720518f));
        var v3 = vec3(x + (s * .12671045f), y + (s * .38751736f), z + (s * .12760002f));
        var v4 = vec3(x + (s * .30631556f), y + (s * .20791227f), z + (s * .48681026f));
        var v5 = vec3(x + (s * .48592068f), y + (s * .20791225f), z + (s * .30720514f));
        var v6 = vec3(x + (s * .30631556f), y + (s * .56712254f), z + (s * .48681026f));
        var v7 = vec3(x + (s * .12671047f), y + (s * .56712254f), z + (s * .30720512f));
        var v8 = vec3(x + (s * .12671042f), y + (s * .3875174f), z + (s * .48681026f));
        var v9 = vec3(x + (s * .48592068f), y + (s * .38751736f), z + (s * .1276f));
        var v10 = vec3(x + (s * .30631556f), y + (s * .56712254f), z + (s * .1276f));
        var v11 = vec3(x + (s * .48592068f), y + (s * .56712254f), z + (s * .30720512f));
        var v12 = vec3(x + (s * .48592068f), y + (s * .38751743f), z + (s * .4868103f));

        tri(v1.idx(), v2.idx(), v3.idx(), 0xff0000);
        tri(v4.idx(), v2.idx(), v5.idx(), 0x7f8000);
        tri(v5.idx(), v2.idx(), v1.idx(), 0x3fc000);
        tri(v6.idx(), v7.idx(), v8.idx(), 0x1fe000);
        tri(v9.idx(), v10.idx(), v11.idx(), 0x0ff000);
        tri(v8.idx(), v2.idx(), v4.idx(), 0x07f800);
        tri(v5.idx(), v1.idx(), v9.idx(), 0x03fc00);
        tri(v3.idx(), v7.idx(), v10.idx(), 0x01fe00);
        tri(v8.idx(), v7.idx(), v2.idx(), 0x00ff00);
        tri(v2.idx(), v7.idx(), v3.idx(), 0x007f80);
        tri(v8.idx(), v4.idx(), v6.idx(), 0x003fc0);
        tri(v6.idx(), v4.idx(), v12.idx(), 0x001fe0);
        tri(v11.idx(), v12.idx(), v9.idx(), 0x000ff0);
        tri(v9.idx(), v12.idx(), v5.idx(), 0x0007f8);
        tri(v7.idx(), v6.idx(), v10.idx(), 0x0003fc);
        tri(v6.idx(), v11.idx(), v10.idx(), 0x0001fe);
        tri(v1.idx(), v3.idx(), v9.idx(), 0x0000ff);
        tri(v9.idx(), v3.idx(), v10.idx(), 0x00007f);
        tri(v12.idx(), v4.idx(), v5.idx(), 0x00003f);
        tri(v6.idx(), v12.idx(), v11.idx(), 0x00001f);
        return this;
    }


     public F32Mesh3D rubric(float s) {
        for (int x = -1; x < 2; x++) {
            for (int y = -1; y < 2; y++) {
                for (int z = -1; z < 2; z++) {
                    cube(x * .5f, y * .5f, z * .5f, s);
                }
            }
        }
        return this;
    }

    public  F32Vec3.F32Vec3Pool.Idx  vec3(float x, float y, float z) {
        var newVec = F32Vec3.f32Vec3Pool.of(x,y, z);
        vecEntries.add(newVec);
        return newVec;
    }


}
