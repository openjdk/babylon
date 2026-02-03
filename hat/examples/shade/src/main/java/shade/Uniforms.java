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
    interface ivec2Field extends ivec2.Field, Struct {
        void x(int x);
        void y(int y);
    }

    interface vec2Field extends vec2.Field, Struct {
        void x(float x);
        void y(float y);
    }

    interface vec3Field extends vec3.Field, Struct {
        void x(float x);
        void y(float y);
        void z(float z);
    }

    interface vec4Field extends vec4.Field, Struct {
        void x(float x);
        void y(float y);
        void z(float z);
        void w(float w);
    }

    vec2Field fragCoord();

    vec4Field fragColor();

    ivec2Field iResolution();

    long iTime();
    void iTime(long iTime);
    ivec2Field iMouse();
    long iFrame();
    void iFrame(long iFrame);
    Schema<Uniforms> schema = Schema.of(Uniforms.class, uniforms -> uniforms
            .field("fragCoord", fragCoord -> fragCoord.fields("x", "y"))
            .field("fragColor", fragColor -> fragColor.fields("x", "y", "z", "w"))
            .field("iResolution", iResolution -> iResolution.fields("x", "y"))
            .field("iMouse", iMouse -> iMouse.fields("x", "y"))
            .field("iTime")
            .field("iFrame")
    );

    static Uniforms create(Accelerator accelerator) {
        return BoundSchema.of(accelerator, schema).allocate();
    }
}
