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
package hat.backend.c99codebuilders;


import hat.optools.BinaryArithmeticOrLogicOperation;
import hat.optools.BinaryTestOpWrapper;
import hat.optools.ConstantOpWrapper;
import hat.optools.ConvOpWrapper;
import hat.optools.FieldLoadOpWrapper;
import hat.optools.FieldStoreOpWrapper;
import hat.optools.ForOpWrapper;
import hat.optools.FuncCallOpWrapper;
import hat.optools.IfOpWrapper;
import hat.optools.InvokeOpWrapper;
import hat.optools.JavaBreakOpWrapper;
import hat.optools.JavaContinueOpWrapper;
import hat.optools.JavaLabeledOpWrapper;
import hat.optools.LambdaOpWrapper;
import hat.optools.LogicalOpWrapper;
import hat.optools.OpWrapper;
import hat.optools.ReturnOpWrapper;
import hat.optools.TernaryOpWrapper;
import hat.optools.TupleOpWrapper;
import hat.optools.VarDeclarationOpWrapper;
import hat.optools.VarFuncDeclarationOpWrapper;
import hat.optools.VarLoadOpWrapper;
import hat.optools.VarStoreOpWrapper;
import hat.optools.WhileOpWrapper;
import hat.optools.YieldOpWrapper;

import java.lang.reflect.code.Op;

public interface C99HatBuilderInterface<T extends C99HatBuilder<?>> {


    public T varLoad(C99HatBuildContext buildContext, VarLoadOpWrapper varAccessOpWrapper);

    public T varStore(C99HatBuildContext buildContext, VarStoreOpWrapper varAccessOpWrapper);

    // public T var(BuildContext buildContext, VarDeclarationOpWrapper varDeclarationOpWrapper) ;

    public T varDeclaration(C99HatBuildContext buildContext, VarDeclarationOpWrapper varDeclarationOpWrapper);

    public T varFuncDeclaration(C99HatBuildContext buildContext, VarFuncDeclarationOpWrapper varFuncDeclarationOpWrapper);

    public T fieldLoad(C99HatBuildContext buildContext, FieldLoadOpWrapper fieldLoadOpWrapper);

    public T fieldStore(C99HatBuildContext buildContext, FieldStoreOpWrapper fieldStoreOpWrapper);


    T binaryOperation(C99HatBuildContext buildContext, BinaryArithmeticOrLogicOperation binaryOperatorOpWrapper);

    T logical(C99HatBuildContext buildContext, LogicalOpWrapper logicalOpWrapper);

    T binaryTest(C99HatBuildContext buildContext, BinaryTestOpWrapper binaryTestOpWrapper);

    T conv(C99HatBuildContext buildContext, ConvOpWrapper convOpWrapper);


    T constant(C99HatBuildContext buildContext, ConstantOpWrapper constantOpWrapper);

    T javaYield(C99HatBuildContext buildContext, YieldOpWrapper yieldOpWrapper);

    T lambda(C99HatBuildContext buildContext, LambdaOpWrapper lambdaOpWrapper);

    T tuple(C99HatBuildContext buildContext, TupleOpWrapper lambdaOpWrapper);

    T funcCall(C99HatBuildContext buildContext, FuncCallOpWrapper funcCallOpWrapper);

    T javaIf(C99HatBuildContext buildContext, IfOpWrapper ifOpWrapper);

    T javaWhile(C99HatBuildContext buildContext, WhileOpWrapper whileOpWrapper);

    T javaLabeled(C99HatBuildContext buildContext, JavaLabeledOpWrapper javaLabeledOpWrapperOp);

    T javaContinue(C99HatBuildContext buildContext, JavaContinueOpWrapper javaContinueOpWrapper);

    T javaBreak(C99HatBuildContext buildContext, JavaBreakOpWrapper javaBreakOpWrapper);

    T javaFor(C99HatBuildContext buildContext, ForOpWrapper forOpWrapper);


    public T methodCall(C99HatBuildContext buildContext, InvokeOpWrapper invokeOpWrapper);

    public T ternary(C99HatBuildContext buildContext, TernaryOpWrapper ternaryOpWrapper);

    public T parencedence(C99HatBuildContext buildContext, Op parent, OpWrapper<?> child);

    public T parencedence(C99HatBuildContext buildContext, OpWrapper<?> parent, OpWrapper<?> child);

    public T parencedence(C99HatBuildContext buildContext, Op parent, Op child);

    public T parencedence(C99HatBuildContext buildContext, OpWrapper<?> parent, Op child);

    public T ret(C99HatBuildContext buildContext, ReturnOpWrapper returnOpWrapper);

    default T recurse(C99HatBuildContext buildContext, OpWrapper<?> wrappedOp) {
        switch (wrappedOp) {
            case VarLoadOpWrapper $ -> varLoad(buildContext, $);
            case VarStoreOpWrapper $ -> varStore(buildContext, $);
            case FieldLoadOpWrapper $ -> fieldLoad(buildContext, $);
            case FieldStoreOpWrapper $ -> fieldStore(buildContext, $);
            case BinaryArithmeticOrLogicOperation $ -> binaryOperation(buildContext, $);
            case BinaryTestOpWrapper $ -> binaryTest(buildContext, $);
            case ConvOpWrapper $ -> conv(buildContext, $);
            case ConstantOpWrapper $ -> constant(buildContext, $);
            case YieldOpWrapper $ -> javaYield(buildContext, $);
            case FuncCallOpWrapper $ -> funcCall(buildContext, $);
            case LogicalOpWrapper $ -> logical(buildContext, $);
            case InvokeOpWrapper $ -> methodCall(buildContext, $);
            case TernaryOpWrapper $ -> ternary(buildContext, $);
            case VarDeclarationOpWrapper $ -> varDeclaration(buildContext, $);
            case VarFuncDeclarationOpWrapper $ -> varFuncDeclaration(buildContext, $);
            case LambdaOpWrapper $ -> lambda(buildContext, $);
            case TupleOpWrapper $ -> tuple(buildContext, $);
            case WhileOpWrapper $ -> javaWhile(buildContext, $);
            case IfOpWrapper $ -> javaIf(buildContext, $);
            case ForOpWrapper $ -> javaFor(buildContext, $);

            case ReturnOpWrapper $ -> ret(buildContext, $);
            case JavaLabeledOpWrapper $ -> javaLabeled(buildContext, $);
            case JavaBreakOpWrapper $ -> javaBreak(buildContext, $);
            case JavaContinueOpWrapper $ -> javaContinue(buildContext, $);
            default -> throw new IllegalStateException("handle nesting of op " + wrappedOp.op());
        }
        return (T) this;
    }


}
