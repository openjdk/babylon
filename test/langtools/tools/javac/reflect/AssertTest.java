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

import java.lang.runtime.CodeReflection;

/*
 * @test
 * @summary Smoke test for code reflection with array access.
 * @build AssertTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester AssertTest
 */
public class AssertTest {

    @CodeReflection
    @IR("""
              func @"assertTest" (%0 : int)void -> {
                  %1 : Var<int> = var %0 @"i";
                  assert
                      ()boolean -> {
                          %2 : int = var.load %1;
                          %3 : int = constant @"1";
                          %4 : boolean = eq %2 %3;
                          yield %4;
                      }
                      ()java.lang.String -> {
                          %5 : java.lang.String = constant @"i does not equal 1";
                          yield %5;
                      };
                  return;
              };
            """)
    public static void assertTest(int i) {
        assert (i == 1) : "i does not equal 1";
    }

    @CodeReflection
    @IR("""
            func @"assertTest2" (%0 : int)void -> {
                  %1 : Var<int> = var %0 @"i";
                  assert ()boolean -> {
                      %2 : int = var.load %1;
                      %3 : int = constant @"1";
                      %4 : boolean = eq %2 %3;
                      yield %4;
                  };
                  return;
              };
            """)
    public static void assertTest2(int i) {
        assert (i == 1);
    }
}
