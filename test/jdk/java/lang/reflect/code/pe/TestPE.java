/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.*;
import jdk.incubator.code.op.CoreOp;

import java.lang.invoke.MethodHandles;
import java.util.*;
import java.util.function.IntConsumer;
import java.util.function.Predicate;

/*
 * @test
 * @modules jdk.incubator.code
 * @build TestPE
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TestPE
 * @enablePreview
 */

public class TestPE {

    public static MethodHandles.Lookup lookup() {
        return MethodHandles.lookup();
    }

    public static Predicate<Op> opConstants() {
        return op -> switch (op) {
            case CoreOp.ConstantOp _ -> true;
            case CoreOp.InvokeOp _ -> false;
            case CoreOp.ReturnOp _ -> false;
            default -> op.result() != null;
        };
    }


    @CodeReflection
    @EvaluatedModel("""
            func @"ifStatement" (%0 : TestPE, %1 : java.util.function.IntConsumer)void -> {
                %2 : Var<java.util.function.IntConsumer> = var %1 @"c";
                %3 : java.util.function.IntConsumer = var.load %2;
                %4 : int = constant @"1";
                invoke %3 %4 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """)
    @EvaluatedModel(value = """
            func @"ifStatement" (%0 : TestPE, %1 : java.util.function.IntConsumer)void -> {
                %2 : int = constant @"1";
                invoke %1 %2 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """,
            ssa = true
    )
    void ifStatement(IntConsumer c) {
        if (true) {
            c.accept(1);
        } else {
            c.accept(2);
        }
    }

    @CodeReflection
    @EvaluatedModel("""
            func @"forStatement" (%0 : TestPE, %1 : java.util.function.IntConsumer)void -> {
                %2 : Var<java.util.function.IntConsumer> = var %1 @"c";
                %3 : java.util.function.IntConsumer = var.load %2;
                %4 : int = constant @"0";
                invoke %3 %4 @"java.util.function.IntConsumer::accept(int)void";
                %5 : java.util.function.IntConsumer = var.load %2;
                %6 : int = constant @"2";
                invoke %5 %6 @"java.util.function.IntConsumer::accept(int)void";
                %7 : java.util.function.IntConsumer = var.load %2;
                %8 : int = constant @"-4";
                invoke %7 %8 @"java.util.function.IntConsumer::accept(int)void";
                %9 : java.util.function.IntConsumer = var.load %2;
                %10 : int = constant @"6";
                invoke %9 %10 @"java.util.function.IntConsumer::accept(int)void";
                %11 : java.util.function.IntConsumer = var.load %2;
                %12 : int = constant @"-8";
                invoke %11 %12 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """)
    @EvaluatedModel(value = """
            func @"forStatement" (%0 : TestPE, %1 : java.util.function.IntConsumer)void -> {
                %2 : int = constant @"0";
                invoke %1 %2 @"java.util.function.IntConsumer::accept(int)void";
                %3 : int = constant @"2";
                invoke %1 %3 @"java.util.function.IntConsumer::accept(int)void";
                %4 : int = constant @"-4";
                invoke %1 %4 @"java.util.function.IntConsumer::accept(int)void";
                %5 : int = constant @"6";
                invoke %1 %5 @"java.util.function.IntConsumer::accept(int)void";
                %6 : int = constant @"-8";
                invoke %1 %6 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """,
            ssa = true
    )
    void forStatement(IntConsumer c) {
        for (int i = 0; i < 5; i++) {
            int v;
            if (i % 2 == 0) {
                v = -i * 2;
            } else {
                v = i * 2;
            }
            c.accept(v);
        }
    }

    boolean b = true;
    int[] x = new int[10];

    @CodeReflection
    @EvaluatedModel("""
            func @"f" (%0 : TestPE, %1 : java.util.function.IntConsumer)void -> {
                %2 : Var<java.util.function.IntConsumer> = var %1 @"c";
                %3 : java.util.function.IntConsumer = var.load %2;
                %4 : int = constant @"1";
                invoke %3 %4 @"java.util.function.IntConsumer::accept(int)void";
                %5 : java.util.function.IntConsumer = var.load %2;
                %6 : int = constant @"3";
                invoke %5 %6 @"java.util.function.IntConsumer::accept(int)void";
                %7 : boolean = field.load %0 @"TestPE::b()boolean";
                cbranch %7 ^block_1 ^block_2;

              ^block_1:
                %8 : java.util.function.IntConsumer = var.load %2;
                %9 : int = constant @"5";
                invoke %8 %9 @"java.util.function.IntConsumer::accept(int)void";
                %10 : java.util.function.IntConsumer = var.load %2;
                %11 : int = constant @"0";
                invoke %10 %11 @"java.util.function.IntConsumer::accept(int)void";
                %12 : java.util.function.IntConsumer = var.load %2;
                %13 : int = constant @"6";
                invoke %12 %13 @"java.util.function.IntConsumer::accept(int)void";
                %14 : java.util.function.IntConsumer = var.load %2;
                %15 : int = constant @"1";
                invoke %14 %15 @"java.util.function.IntConsumer::accept(int)void";
                %16 : java.util.function.IntConsumer = var.load %2;
                %17 : int = constant @"0";
                %18 : int[] = field.load %0 @"TestPE::x()int[]";
                %19 : int = constant @"0";
                %20 : int = array.load %18 %19;
                %21 : int = add %17 %20;
                invoke %16 %21 @"java.util.function.IntConsumer::accept(int)void";
                %22 : java.util.function.IntConsumer = var.load %2;
                %23 : int = constant @"1";
                invoke %22 %23 @"java.util.function.IntConsumer::accept(int)void";
                %24 : java.util.function.IntConsumer = var.load %2;
                %25 : int = constant @"7";
                invoke %24 %25 @"java.util.function.IntConsumer::accept(int)void";
                %26 : java.util.function.IntConsumer = var.load %2;
                %27 : int = constant @"2";
                invoke %26 %27 @"java.util.function.IntConsumer::accept(int)void";
                %28 : java.util.function.IntConsumer = var.load %2;
                %29 : int = constant @"1";
                %30 : int[] = field.load %0 @"TestPE::x()int[]";
                %31 : int = constant @"1";
                %32 : int = array.load %30 %31;
                %33 : int = add %29 %32;
                invoke %28 %33 @"java.util.function.IntConsumer::accept(int)void";
                %34 : java.util.function.IntConsumer = var.load %2;
                %35 : int = constant @"2";
                invoke %34 %35 @"java.util.function.IntConsumer::accept(int)void";
                %36 : java.util.function.IntConsumer = var.load %2;
                %37 : int = constant @"7";
                invoke %36 %37 @"java.util.function.IntConsumer::accept(int)void";
                %38 : java.util.function.IntConsumer = var.load %2;
                %39 : int = constant @"2";
                invoke %38 %39 @"java.util.function.IntConsumer::accept(int)void";
                %40 : java.util.function.IntConsumer = var.load %2;
                %41 : int = constant @"2";
                %42 : int[] = field.load %0 @"TestPE::x()int[]";
                %43 : int = constant @"2";
                %44 : int = array.load %42 %43;
                %45 : int = add %41 %44;
                invoke %40 %45 @"java.util.function.IntConsumer::accept(int)void";
                %46 : java.util.function.IntConsumer = var.load %2;
                %47 : int = constant @"8";
                invoke %46 %47 @"java.util.function.IntConsumer::accept(int)void";
                branch ^block_3;

              ^block_2:
                branch ^block_3;

              ^block_3:
                %48 : java.util.function.IntConsumer = var.load %2;
                %49 : int = constant @"9";
                invoke %48 %49 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """)
    @EvaluatedModel(value = """
            func @"f" (%0 : TestPE, %1 : java.util.function.IntConsumer)void -> {
                %2 : int = constant @"1";
                invoke %1 %2 @"java.util.function.IntConsumer::accept(int)void";
                %3 : int = constant @"3";
                invoke %1 %3 @"java.util.function.IntConsumer::accept(int)void";
                %4 : boolean = field.load %0 @"TestPE::b()boolean";
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : int = constant @"5";
                invoke %1 %5 @"java.util.function.IntConsumer::accept(int)void";
                %6 : int = constant @"0";
                invoke %1 %6 @"java.util.function.IntConsumer::accept(int)void";
                %7 : int = constant @"6";
                invoke %1 %7 @"java.util.function.IntConsumer::accept(int)void";
                %8 : int = constant @"1";
                invoke %1 %8 @"java.util.function.IntConsumer::accept(int)void";
                %9 : int[] = field.load %0 @"TestPE::x()int[]";
                %10 : int = array.load %9 %6;
                %11 : int = add %6 %10;
                invoke %1 %11 @"java.util.function.IntConsumer::accept(int)void";
                %12 : int = constant @"1";
                invoke %1 %12 @"java.util.function.IntConsumer::accept(int)void";
                %13 : int = constant @"7";
                invoke %1 %13 @"java.util.function.IntConsumer::accept(int)void";
                %14 : int = constant @"2";
                invoke %1 %14 @"java.util.function.IntConsumer::accept(int)void";
                %15 : int[] = field.load %0 @"TestPE::x()int[]";
                %16 : int = array.load %15 %12;
                %17 : int = add %12 %16;
                invoke %1 %17 @"java.util.function.IntConsumer::accept(int)void";
                %18 : int = constant @"2";
                invoke %1 %18 @"java.util.function.IntConsumer::accept(int)void";
                %19 : int = constant @"7";
                invoke %1 %19 @"java.util.function.IntConsumer::accept(int)void";
                %20 : int = constant @"2";
                invoke %1 %20 @"java.util.function.IntConsumer::accept(int)void";
                %21 : int[] = field.load %0 @"TestPE::x()int[]";
                %22 : int = array.load %21 %18;
                %23 : int = add %18 %22;
                invoke %1 %23 @"java.util.function.IntConsumer::accept(int)void";
                %24 : int = constant @"8";
                invoke %1 %24 @"java.util.function.IntConsumer::accept(int)void";
                branch ^block_3;

              ^block_2:
                branch ^block_3;

              ^block_3:
                %25 : int = constant @"9";
                invoke %1 %25 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """,
            ssa = true
    )
    void f(IntConsumer c) {
        c.accept(1);

        if (false) {
            c.accept(2);
        } else if (true) {
            c.accept(3);
        } else {
            c.accept(4);
        }

        if (b) {
            c.accept(5);
            for (int i = 0; i < 3; i++) {
                c.accept(i);
                int v;
                if (i == 0) {
                    c.accept(6);
                    v = 1;
                } else {
                    c.accept(7);
                    v = 2;
                }
                c.accept(v);
                c.accept(i + x[i]);
            }

            c.accept(8);
        }

        c.accept(9);
    }

    @CodeReflection
    @EvaluatedModel("""
            func @"constantsInBranches" (%0 : TestPE, %1 : java.util.function.IntConsumer, %2 : int)void -> {
                %3 : Var<java.util.function.IntConsumer> = var %1 @"c";
                %4 : Var<int> = var %2 @"arg";
                %5 : int = var.load %4;
                %6 : Var<int> = var %5 @"x";
                %7 : int = var.load %6;
                %8 : int = constant @"0";
                %9 : boolean = eq %7 %8;
                cbranch %9 ^block_1 ^block_2;

              ^block_1:
                %10 : int = constant @"1";
                var.store %6 %10;
                branch ^block_3;

              ^block_2:
                %11 : int = constant @"2";
                var.store %6 %11;
                branch ^block_3;

              ^block_3:
                %12 : java.util.function.IntConsumer = var.load %3;
                %13 : int = var.load %6;
                invoke %12 %13 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """)
    @EvaluatedModel(value = """
            func @"constantsInBranches" (%0 : TestPE, %1 : java.util.function.IntConsumer, %2 : int)void -> {
                %3 : int = constant @"0";
                %4 : boolean = eq %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : int = constant @"1";
                branch ^block_3(%5);

              ^block_2:
                %6 : int = constant @"2";
                branch ^block_3(%6);

              ^block_3(%7 : int):
                invoke %1 %7 @"java.util.function.IntConsumer::accept(int)void";
                return;
            };
            """,
            ssa = true
    )
    void constantsInBranches(IntConsumer c, int arg) {
        var x = arg;
        if (x == 0) {
            x = 1;
        } else {
            x = 2;
        }
        c.accept(x);
    }

}
