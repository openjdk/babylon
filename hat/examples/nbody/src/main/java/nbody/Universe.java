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
package nbody;

import hat.Accelerator;
import hat.buffer.Buffer;
import hat.ifacemapper.Schema;

public interface Universe extends Buffer {
    int length();

    interface Body extends Struct {
        float x();

        float y();

        float z();

        float vx();

        float vy();

        float vz();

        void x(float x);

        void y(float y);

        void z(float z);

        void vx(float vx);

        void vy(float vy);

        void vz(float vz);
    }

    Body body(long idx);

    /*
    typedef struct Body_s{
        float x;
        float y;
        float y;
        float vx;
        float vy;
        float y;
    } Body_t;

    typedef struct Universe_s{
       int length;
       Body_t body[1];
    }Universe_t;

     */
    Schema<Universe> schema = Schema.of(Universe.class, resultTable -> resultTable

            .arrayLen("length").array("body", array -> array
                    .fields("x", "y", "z", "vx", "vy", "vz")
            )
    );

    static Universe create(Accelerator accelerator, int length) {
        return schema.allocate(accelerator, length);
    }

}
