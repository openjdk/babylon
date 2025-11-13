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
package view.f32.pool;

import view.f32.F32;
import view.f32.F32x2;
import view.f32.F32x2Triangle;
import view.f32.F32x3;
import view.f32.F32x3Triangle;
import view.f32.F32x4x4;

public record F32PoolBased(F32x4x4.Factory f32x4x4Factory,
                           F32x3.Factory f32x3Factory,
                           F32x2.Factory f32x2Factory,
                           F32x3Triangle.Factory f32x3TriangleFactory,
                           F32x2Triangle.Factory f32x2TriangleFactory

) implements F32 {
}
