/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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
 * @test /nodynamiccopyright/
 * @compile/fail/ref=TestMethodCall.out -XDrawDiagnostics TestMethodCall.java
 */

import java.lang.reflect.code.Quoted;

public class TestMethodCall {
    void test(boolean cond) {
        apply(() -> {});
        apply((int i) -> ""); // ok (int->String)
        apply((int i) -> { return ""; }); // ok (int->String)
        apply((int i) -> { }); // ok (int->V)
        apply((int i) -> { return; }); // ok (int->V)
        apply((int i) -> { if (cond) return; else return; }); // ok (int->V)
        apply((int i) -> { if (cond) return "1"; else return "2"; }); // ok (int->String)
    }

    void testImplicit(boolean cond) {
        apply((i) -> ""); // error - no parameter types
        apply((i) -> { return ""; }); // error - no parameter types
        apply((i) -> { }); // error - no parameter types
        apply((i) -> { return; }); // error - no parameter types
        apply((i) -> { if (cond) return; else return; }); // error - no parameter types
        apply((i) -> { if (cond) return "1"; else return "2"; }); // error - no parameter types
    }

    void testImplicitVar(boolean cond) {
        apply((var i) -> ""); // error - no parameter types
        apply((var i) -> { return ""; }); // error - no parameter types
        apply((var i) -> { }); // error - no parameter types
        apply((var i) -> { return; }); // error - no parameter types
        apply((var i) -> { if (cond) return; else return; }); // error - no parameter types
        apply((var i) -> { if (cond) return "1"; else return "2"; }); // error - no parameter types
    }

    void testBadInferredReturn(boolean cond) {
        apply((int i) -> { if (cond) return; else return ""; }); // error - only one branch returns
        apply((int i) -> { if (cond) { return "2"; } }); // error - one return, but body completes normally
    }

    void apply(Quoted quoted) { }
}
