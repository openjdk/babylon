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
 * @test
 * @enablePreview
 * @build StringTemplateTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester StringTemplateTest
 */

import java.lang.runtime.CodeReflection;

import static java.lang.StringTemplate.RAW;
import static java.lang.StringTemplate.STR;

public class StringTemplateTest {

    @CodeReflection
    @IR("""
            func @"f" (%0 : int)void -> {
                  %1 : Var<int> = var %0 @"y";
                  %2 : java.lang.String = constant @"y = ";
                  %3 : java.lang.String = constant @"";
                  %4 : int = var.load %1;
                  %5 : java.lang.StringTemplate = java.stringTemplate %2 %4 %3;
                  %6 : java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException> = field.load @"java.lang.StringTemplate::STR()java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException>";
                  %7 : java.lang.String = invoke %6 %5 @"java.lang.StringTemplate$Processor::process(java.lang.StringTemplate)java.lang.String";
                  %8 : Var<java.lang.String> = var %7 @"s";
                  return;
            };
            """)
    static void f(int y) {
        String s = STR. "y = \{y}" ;
    }

    @CodeReflection
    @IR("""
            func @"f2" (%0 : int)void -> {
                %1 : Var<int> = var %0 @"y";
                %2 : java.lang.String = constant @"y = ";
                %3 : java.lang.String = constant @"";
                %4 : int = var.load %1;
                %5 : java.lang.StringTemplate = java.stringTemplate %2 %4 %3;
                %6 : java.lang.StringTemplate$Processor<java.lang.StringTemplate, java.lang.RuntimeException> = field.load @"java.lang.StringTemplate::RAW()java.lang.StringTemplate$Processor<java.lang.StringTemplate, java.lang.RuntimeException>";
                %7 : java.lang.StringTemplate = invoke %6 %5 @"java.lang.StringTemplate$Processor::process(java.lang.StringTemplate)java.lang.StringTemplate";
                %8 : Var<java.lang.StringTemplate> = var %7 @"st";
                return;
            };
            """)
    static void f2(int y) {
        StringTemplate st = RAW. "y = \{y}" ;
    }
}
