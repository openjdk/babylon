/*
 * Copyright (c) 2024-26, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Reflect;

public record S32x2(int x, int y) {
    @Reflect
    static S32x2 of(int x, int y) {
        return new S32x2(x, y);
    }

    @Reflect
    static S32x2 add(S32x2 lhs, S32x2 rhs) { // We don't have to call this add.
        return new S32x2(lhs.x + rhs.x, lhs.y + rhs.y);
    }

    @Reflect
    static S32x2 sub(S32x2 lhs, S32x2 rhs) {
        return new S32x2(lhs.x - rhs.x, lhs.y - rhs.y);
    }

    @Reflect
    static S32x2 div(S32x2 lhs, S32x2 rhs) {
        return new S32x2(lhs.x / rhs.x, lhs.y / rhs.y);
    }

    @Reflect
    static S32x2 mod(S32x2 lhs, S32x2 rhs) {
        return new S32x2(lhs.x % rhs.x, lhs.y % rhs.y);
    }

    @Reflect
    static S32x2 mul(S32x2 lhs, S32x2 rhs) {
        return new S32x2(lhs.x * rhs.x, lhs.y * rhs.y);
    }

    @Reflect
    public S32x2 mul(S32x2 s32x2) {
        return mul(this, s32x2);
    }

    @Reflect
    public S32x2 mul(int scalar) {
        return mul(this, S32x2.of(scalar, scalar));
    }

    @Reflect
    public S32x2 add(S32x2 s32x2) { // we don't have to call this add either
        return add(this, s32x2);
    }

    @Reflect
    public S32x2 sub(S32x2 s32x2) {
        return sub(this, s32x2);
    }

    @Reflect
    public S32x2 div(S32x2 s32x2) {
        return div(this, s32x2);
    }

    @Reflect
    public S32x2 mod(S32x2 s32x2) {
        return mod(this, s32x2);
    }
}
