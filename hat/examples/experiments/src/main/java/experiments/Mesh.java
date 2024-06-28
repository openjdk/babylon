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
package experiments;

import hat.Schema;
import hat.buffer.Buffer;

import java.util.Random;

public interface Mesh extends Buffer {
    interface Point3D extends StructChild {
        int x();void x(int x);
        int y();void y(int y);
        int z();void z(int z);
    }

    int points();void points(int points);
    Point3D point(long idx);

    interface Vertex3D extends StructChild {
        int from();void from(int id);
        int to();void to(int id);
    }

    int vertices(); void vertices(int vertices);
    Vertex3D vertex(long idx);

    Schema<Mesh> schema = Schema.of(Mesh.class, cascade -> cascade
            .arrayLen("points").array("point", p -> p.fields("x","y","z"))
            .arrayLen("vertices").array("vertex", v -> v.fields("from","to"))
    );

    public static void main(String[] args) {
        Mesh.schema.toText(t -> System.out.print(t));
        var mesh = Mesh.schema.allocate( 100, 10);
        mesh.points(100);
        mesh.vertices(10);
        Random random = new Random(System.currentTimeMillis());
        for (int p=0; p< mesh.points(); p++){
            var point3D = mesh.point(p);
            point3D.x(random.nextInt(100));
            point3D.y(random.nextInt(100));
            point3D.z(random.nextInt(100));
        }
        for (int v=0; v< mesh.vertices(); v++){
            var vertex3D = mesh.vertex(v);
            vertex3D.from(random.nextInt(mesh.points()));
            vertex3D.to(random.nextInt(mesh.points()));
        }
        System.out.println(Buffer.getLayout(mesh));
    }
}
