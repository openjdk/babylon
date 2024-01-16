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
 * @compile/fail/ref=TestAssignment.out -XDrawDiagnostics TestAssignment.java
 */

import java.lang.reflect.code.Quoted;

class TestAssignment {
    void test(boolean cond) {
        Quoted f_NoRet = () -> {};
        Quoted fiS_NoRet = (int i) -> ""; // ok (int->String)
        Quoted fiS_Ret = (int i) -> { return ""; }; // ok (int->String)
        Quoted fiV_NoRet = (int i) -> { }; // ok (int->V)
        Quoted fiV_Ret = (int i) -> { return; }; // ok (int->V)
        Quoted fiV_RetRet = (int i) -> { if (cond) return; else return; }; // ok (int->V)
        Quoted fiS_RetRet = (int i) -> { if (cond) return "1"; else return "2"; }; // ok (int->String)
    }

    void testImplicit(boolean cond) {
        Quoted fiS_NoRet = (i) -> ""; // error - no parameter types
        Quoted fiS_Ret = (i) -> { return ""; }; // error - no parameter types
        Quoted fiV_NoRet = (i) -> { }; // error - no parameter types
        Quoted fiV_Ret = (i) -> { return; }; // error - no parameter types
        Quoted fiV_RetRet = (i) -> { if (cond) return; else return; }; // error - no parameter types
        Quoted fiS_RetRet = (i) -> { if (cond) return "1"; else return "2"; }; // error - no parameter types
    }

    void testImplicitVar(boolean cond) {
        Quoted fiS_NoRet = (var i) -> ""; // error - no parameter types
        Quoted fiS_Ret = (var i) -> { return ""; }; // error - no parameter types
        Quoted fiV_NoRet = (var i) -> { }; // error - no parameter types
        Quoted fiV_Ret = (var i) -> { return; }; // error - no parameter types
        Quoted fiV_RetRet = (var i) -> { if (cond) return; else return; }; // error - no parameter types
        Quoted fiS_RetRet = (var i) -> { if (cond) return "1"; else return "2"; }; // error - no parameter types
    }

    void testBadInferredReturn(boolean cond) {
        Quoted fi_RetVRetS = (int i) -> { if (cond) return; else return ""; }; // error - only one branch returns
        Quoted fi_RetS = (int i) -> { if (cond) { return "2"; } }; // error - one return, but body completes normally
    }
}