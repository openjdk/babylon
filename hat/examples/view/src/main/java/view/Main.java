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

/*
Elite mesh info from

   Based on mesh descriptions found here
       https://6502disassembly.com/a2-elite/
       https://6502disassembly.com/a2-elite/meshes.html

 Based on this excellent youtube 3D graphics  series  https://www.youtube.com/watch?v=ih20l3pJoeU
*/

package view;

import view.f32.F32Mesh3D;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] argArr) {
        var args = new ArrayList<>(List.of(argArr));
       // args.add("ANACONDA");
        var eliteReader = new EliteMeshReader();
        boolean old =true;// Boolean.getBoolean("old");
        var wire =
              //  Graphics2DRenderer.wireOf(1024,1024);
                RasterizingRenderer.wireOf(1024, 1024);
        var fill = RasterizingRenderer.fillOf(1024, 1024);
        Runnable cubeoctahedron =  () -> {
            for (int x = -2; x < 6; x += 2) {
                for (int y = -2; y < 6; y += 2) {
                    for (int z = -2; z < 6; z += 2) {
                        if (old) {
                            F32Mesh3D.of("cubeoctahedron").cubeoctahedron(x, y, z, 2).fin();
                        }else{
                            F32.Mesh.of("cubeoctahedron").cubeoctahedron(x, y, z, 2).fin();
                        }
                    }
                }
            }
        };
        Runnable elite = ()->eliteReader.load(args.getFirst(), old);
        ViewFrame viewFrame = (args.size() > 0 ? ViewFrame.of("view",old,wire, elite): ViewFrame.of("view",old, fill,cubeoctahedron));
        while (true) {
            viewFrame.update();
        }
    }
}
