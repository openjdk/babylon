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
package hat.tools.textmodel.tokens;

import java.util.function.Predicate;

public class FloatConst extends LeafReplacementToken {
    public final float f;

    public FloatConst(Token t1, Token t2, Token t3) {
        super(t1, t2, t3);
        var s = t1.asString() + t2.asString() + t3.asString();
        this.f = Float.parseFloat(s);
    }

    public static boolean isA(Token t, Predicate<FloatConst> predicate) {
        return t instanceof FloatConst floatConst && predicate.test(floatConst);
    }
    public static boolean isA(Token t) {
        return isA(t, _->true);
    }
}
