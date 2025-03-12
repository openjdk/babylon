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

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.ffi.OpenCLBackend;
import static hat.backend.ffi.Config.*;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.Schema;
import hat.buffer.Buffer;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.CodeReflection;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class Mesh {
    public interface MeshData extends Buffer {
        interface Point3D extends Struct {
            int x();

            void x(int x);

            int y();

            void y(int y);

            int z();

            void z(int z);

        }

       int points();

      //  void points(int points);

        Point3D point(long idx);

        interface Vertex3D extends Struct {
            int from();

            void from(int id);

            int to();

            void to(int id);

        }

        int vertices();

       // void vertices(int vertices);

        Vertex3D vertex(long idx);


         GroupLayout LAYOUT = MemoryLayout.structLayout(
                JAVA_INT.withName("points"),
                MemoryLayout.sequenceLayout(100,
                MemoryLayout.structLayout(
                JAVA_INT.withName("x"),
                JAVA_INT.withName("y"),
                JAVA_INT.withName("z")
                ).withName(Point3D.class.getSimpleName())
                ).withName("point"),
                    JAVA_INT.withName("vertices"),
                            MemoryLayout.sequenceLayout(10,
                            MemoryLayout.structLayout(
                            JAVA_INT.withName("from"),
                            JAVA_INT.withName("to")
                            ).withName(Vertex3D.class.getSimpleName())
                ).withName("vertex")
            ).withName(MeshData.class.getSimpleName());

        static GroupLayout getLayout() {
            return LAYOUT;
        }


        Schema<MeshData> schema = Schema.of(MeshData.class, cascade -> cascade
                .arrayLen("points").array("point", p -> p.fields("x", "y", "z"))
                .arrayLen("vertices").array("vertex", v -> v.fields("from", "to"))
        );
        static  MeshData create(Accelerator accelerator) {
            return schema.allocate(accelerator,100,10);
        }
    }

    public static class Compute {
        @CodeReflection
        public static void initPoints(KernelContext kc, MeshData mesh) {
            if (kc.x < kc.maxX) {
                MeshData.Point3D point = mesh.point(kc.x);
                point.x(kc.x);
                point.y(0);
                point.z(0);
            }
        }

        @CodeReflection
        public static void buildMesh(ComputeContext cc, MeshData meshData) {
            cc.dispatchKernel(meshData.points(),
                    kc -> initPoints(kc, meshData)
            );

        }
    }


    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup()
                ,new OpenCLBackend(of(PROFILE(), GPU(), TRACE())));
                //,new DebugBackend(
                //DebugBackend.HowToRunCompute.REFLECT,
                //DebugBackend.HowToRunKernel.BABYLON_INTERPRETER));
      //  MeshData.schema.toText(t -> System.out.print(t));

        var boundSchema = new BoundSchema<>(MeshData.schema, 100, 10);
        var meshDataNew = boundSchema.allocate(accelerator.lookup,accelerator);
        var meshDataOld = MeshData.create(accelerator);

        String layoutNew = Buffer.getLayout(meshDataNew).toString();
        String layoutOld = Buffer.getLayout(meshDataOld).toString();
        if (layoutOld.equals(layoutNew)) {
            MeshData meshData = MeshData.create(accelerator);
            Random random = new Random(System.currentTimeMillis());
            for (int p = 0; p < meshData.points(); p++) {
                var point3D = meshData.point(p);
                point3D.x(random.nextInt(100));
                point3D.y(random.nextInt(100));
                point3D.z(random.nextInt(100));
            }
            for (int v = 0; v < meshData.vertices(); v++) {
                var vertex3D = meshData.vertex(v);
                vertex3D.from(random.nextInt(meshData.points()));
                vertex3D.to(random.nextInt(meshData.points()));
            }

            accelerator.compute(cc -> Compute.buildMesh(cc, meshData));
        }else{
            System.out.println("layouts differ");
        }

    }
}
