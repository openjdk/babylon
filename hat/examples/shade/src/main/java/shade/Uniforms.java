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
package shade;

import hat.Accelerator;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.Schema;

public interface Uniforms extends Buffer {
    interface ivec2 extends Struct{
        int x();
        int y();
        void x(int x);
        void y(int y);
    }
    interface vec2 extends Struct{
        float x();
        float y();
        void x(float x);
        void y(float y);
    }
    interface vec3 extends Struct{
        float x();
        float y();
        float z();
        void x(float x);
        void y(float y);
        void z(float z);
    }
    interface vec4 extends Struct{
        float x();
        float y();
        float z();
        float w();
        void x(float x);
        void y(float y);
        void z(float z);
        void w(float w);
    }
    vec2 fragCoord();
    vec4 fragColor();
    ivec2 iResolution();

    Schema<Uniforms> schema= Schema.of(Uniforms.class, uniforms->uniforms
            .field("fragCoord", fragCoord->fragCoord.fields("x","y"))
            .field("fragColor", fragColor->fragColor.fields("x","y","z","w"))
            .field("iResolution", iResolution->iResolution.fields("x","y"))
    );

    static Uniforms create(Accelerator accelerator) {
        return  BoundSchema.of(accelerator ,schema).allocate();
    }
}
