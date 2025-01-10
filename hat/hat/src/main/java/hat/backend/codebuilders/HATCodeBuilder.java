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
package hat.backend.codebuilders;


import hat.optools.BinaryArithmeticOrLogicOperation;
import hat.optools.BinaryTestOpWrapper;
import hat.optools.ConstantOpWrapper;
import hat.optools.ConvOpWrapper;
import hat.optools.FieldLoadOpWrapper;
import hat.optools.FieldStoreOpWrapper;
import hat.optools.ForOpWrapper;
import hat.optools.FuncCallOpWrapper;
import hat.optools.FuncOpWrapper;
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
import hat.optools.UnaryArithmeticOrLogicOpWrapper;
import hat.optools.VarDeclarationOpWrapper;
import hat.optools.VarFuncDeclarationOpWrapper;
import hat.optools.VarLoadOpWrapper;
import hat.optools.VarStoreOpWrapper;
import hat.optools.WhileOpWrapper;
import hat.optools.YieldOpWrapper;
import hat.text.CodeBuilder;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public abstract class HATCodeBuilder<T extends HATCodeBuilder<T>> extends CodeBuilder<T> {
    public T suffix_t(String name) {
        return identifier(name).identifier("_t");
    }

    public T suffix_u(String name) {
        return identifier(name).identifier("_u");
    }

    public T suffix_s(String name) {
        return identifier(name).identifier("_s");
    }


    public T intDeclaration(String name) {
        return intType().space().identifier(name);
    }

    public T floatDeclaration(String name) {
        return floatType().space().identifier(name);
    }

    public T booleanDeclaration(String name) {
        return booleanType().space().identifier(name);
    }

    public T byteDeclaration(String name) {
        return charType().space().identifier(name);
    }

    public T shortDeclaration(String name) {
        return shortType().space().identifier(name);
    }

    public T structOrUnion(boolean isStruct) {
        return (isStruct ? structKeyword() : union());
    }


    public T typedefKeyword() {
        return keyword("typedef");
    }


    public T structKeyword() {
        return keyword("struct");
    }

    public T union() {
        return keyword("union");
    }


    public T externC() {
        return externKeyword().space().dquote("C");
    }

    T hashDefineKeyword() {
        return hash().keyword("define");
    }

    T hashIfdefKeyword() {
        return hash().keyword("ifdef");
    }

    T hashIfndefKeyword() {
        return hash().keyword("ifndef");
    }

    protected T hashEndif() {
        return hash().keyword("endif").nl();
    }

    T hashIfdef(String value) {
        return hashIfdefKeyword().space().append(value).nl();
    }

    protected T hashIfndef(String value) {
        return hashIfndefKeyword().space().append(value).nl();
    }

    T hashIfdef(String value, Consumer<T> consumer) {
        return hashIfdef(value).accept(consumer).hashEndif();
    }

    protected T hashIfndef(String value, Consumer<T> consumer) {
        return hashIfndef(value).accept(consumer).hashEndif();
    }
  /*  public T defonce(String name, Runnable r) {
        return ifndef(name+"_ONCE_DEF",()->{
            define(name+"_ONCE_DEF").nl();
            r.run();
        });
    }*/

    T pragmaKeyword() {
        return keyword("pragma");
    }

    public T hashDefine(String name, String... values) {
        hashDefineKeyword().space().identifier(name);
        for (String value : values) {
            space().append(value);
        }
        return nl();
    }

    public T pragma(String name, String... values) {
        hash().pragmaKeyword().space().identifier(name);
        for (String value : values) {
            space().append(value);
        }
        return nl();
    }

    T externKeyword() {
        return keyword("extern");
    }

    protected T camel(String value) {
        return identifier(Character.toString(Character.toLowerCase(value.charAt(0)))).identifier(value.substring(1));
    }

    T camelJoin(String prefix, String suffix) {
        return camel(prefix).identifier(Character.toString(Character.toUpperCase(suffix.charAt(0)))).identifier(suffix.substring(1));
    }

    public final T unsignedCharType() {
        return typeName("unsigned").space().charType();
    }

    public T charTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> charType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedCharTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedCharType(), _ -> identifier(name)));
        return self();
    }

    public T shortTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> shortType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedShortTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedShortType(), _ -> identifier(name)));
        return self();
    }

    public T intTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> intType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedIntTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedIntType(), _ -> identifier(name)));
        return self();
    }

    public T floatTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> floatType(), _ -> identifier(name)));
        return self();
    }

    public T longTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> longType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedLongTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedLongType(), _ -> identifier(name)));
        return self();
    }

    public T doubleTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> doubleType(), _ -> identifier(name)));
        return self();
    }

    private T typedef(Consumer<T> lhs, Consumer<T> rhs) {
        return semicolonTerminatedLine(_ -> typedefKeyword().space().accept(lhs).space().accept(rhs));
    }

    public final T unsignedIntType() {
        return typeName("unsigned").space().intType();
    }

    public final T unsignedLongType() {
        return typeName("unsigned").space().longType();
    }

    public final T unsignedShortType() {
        return typeName("unsigned").space().shortType();
    }


    /* this should not be too C99 specific */
    public static interface CodeBuilderInterface<T extends HATCodeBuilderWithContext<?>> {


         T varLoad(CodeBuilderContext buildContext, VarLoadOpWrapper varAccessOpWrapper);

         T varStore(CodeBuilderContext buildContext, VarStoreOpWrapper varAccessOpWrapper);

        // public T var(BuildContext buildContext, VarDeclarationOpWrapper varDeclarationOpWrapper) ;

         T varDeclaration(CodeBuilderContext buildContext, VarDeclarationOpWrapper varDeclarationOpWrapper);

         T varFuncDeclaration(CodeBuilderContext buildContext, VarFuncDeclarationOpWrapper varFuncDeclarationOpWrapper);

         T fieldLoad(CodeBuilderContext buildContext, FieldLoadOpWrapper fieldLoadOpWrapper);

         T fieldStore(CodeBuilderContext buildContext, FieldStoreOpWrapper fieldStoreOpWrapper);

        T unaryOperation(CodeBuilderContext buildContext, UnaryArithmeticOrLogicOpWrapper unaryOperatorOpWrapper);


        T binaryOperation(CodeBuilderContext buildContext, BinaryArithmeticOrLogicOperation binaryOperatorOpWrapper);

        T logical(CodeBuilderContext buildContext, LogicalOpWrapper logicalOpWrapper);

        T binaryTest(CodeBuilderContext buildContext, BinaryTestOpWrapper binaryTestOpWrapper);

        T conv(CodeBuilderContext buildContext, ConvOpWrapper convOpWrapper);


        T constant(CodeBuilderContext buildContext, ConstantOpWrapper constantOpWrapper);

        T javaYield(CodeBuilderContext buildContext, YieldOpWrapper yieldOpWrapper);

        T lambda(CodeBuilderContext buildContext, LambdaOpWrapper lambdaOpWrapper);

        T tuple(CodeBuilderContext buildContext, TupleOpWrapper lambdaOpWrapper);

        T funcCall(CodeBuilderContext buildContext, FuncCallOpWrapper funcCallOpWrapper);

        T javaIf(CodeBuilderContext buildContext, IfOpWrapper ifOpWrapper);

        T javaWhile(CodeBuilderContext buildContext, WhileOpWrapper whileOpWrapper);

        T javaLabeled(CodeBuilderContext buildContext, JavaLabeledOpWrapper javaLabeledOpWrapperOp);

        T javaContinue(CodeBuilderContext buildContext, JavaContinueOpWrapper javaContinueOpWrapper);

        T javaBreak(CodeBuilderContext buildContext, JavaBreakOpWrapper javaBreakOpWrapper);

        T javaFor(CodeBuilderContext buildContext, ForOpWrapper forOpWrapper);


         T methodCall(CodeBuilderContext buildContext, InvokeOpWrapper invokeOpWrapper);

         T ternary(CodeBuilderContext buildContext, TernaryOpWrapper ternaryOpWrapper);

         T parencedence(CodeBuilderContext buildContext, Op parent, OpWrapper<?> child);

         T parencedence(CodeBuilderContext buildContext, OpWrapper<?> parent, OpWrapper<?> child);

         T parencedence(CodeBuilderContext buildContext, Op parent, Op child);

         T parencedence(CodeBuilderContext buildContext, OpWrapper<?> parent, Op child);

         T ret(CodeBuilderContext buildContext, ReturnOpWrapper returnOpWrapper);

        default T recurse(CodeBuilderContext buildContext, OpWrapper<?> wrappedOp) {
            switch (wrappedOp) {
                case VarLoadOpWrapper $ -> varLoad(buildContext, $);
                case VarStoreOpWrapper $ -> varStore(buildContext, $);
                case FieldLoadOpWrapper $ -> fieldLoad(buildContext, $);
                case FieldStoreOpWrapper $ -> fieldStore(buildContext, $);
                case BinaryArithmeticOrLogicOperation $ -> binaryOperation(buildContext, $);
                case UnaryArithmeticOrLogicOpWrapper $ -> unaryOperation(buildContext, $);
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

    public static class CodeBuilderContext {

        public static class Scope<OW extends OpWrapper<?>> {
            final Scope<?> parent;
            final OW opWrapper;

            public Scope(Scope<?> parent, OW opWrapper) {
                this.parent = parent;
                this.opWrapper = opWrapper;
            }

            public CoreOp.VarOp resolve(Value value) {
                if (value instanceof Op.Result result && result.op() instanceof CoreOp.VarOp varOp) {
                    return varOp;
                }
                if (parent != null) {
                    return parent.resolve(value);
                }
                throw new IllegalStateException("failed to resolve VarOp for value " + value);
            }
        }

        public static class FuncScope extends Scope<FuncOpWrapper> {
            FuncScope(Scope<?> parent, FuncOpWrapper funcOpWrapper) {
                super(parent, funcOpWrapper);
            }

            @Override
            public CoreOp.VarOp resolve(Value value) {
                if (value instanceof Block.Parameter blockParameter) {
                    if (opWrapper.parameterVarOpMap.containsKey(blockParameter)) {
                        return opWrapper.parameterVarOpMap.get(blockParameter);
                    } else {
                        throw new IllegalStateException("what ?");
                    }
                } else {
                    return super.resolve(value);
                }
            }
        }

        public static abstract class LoopScope<T extends OpWrapper<?>> extends Scope<T> {

            public LoopScope(Scope<?> parent, T opWrapper) {
                super(parent, opWrapper);
            }
        }


        public  static class ForScope extends LoopScope<ForOpWrapper> {
            Map<Block.Parameter, CoreOp.VarOp> blockParamToVarOpMap = new HashMap<>();

            ForOpWrapper forOpWrapper() {
                return opWrapper;
            }

            ForScope(Scope<?> parent, ForOpWrapper forOpWrapper) {
                super(parent, forOpWrapper);
                var loopParams = forOpWrapper().op().loopBody().entryBlock().parameters().toArray(new Block.Parameter[0]);
                var updateParams = forOpWrapper().op().update().entryBlock().parameters().toArray(new Block.Parameter[0]);
                var condParams = forOpWrapper().op().cond().entryBlock().parameters().toArray(new Block.Parameter[0]);
                var lastInitOp = forOpWrapper().op().init().entryBlock().ops().getLast();
                var lastInitOpOperand0Result = (Op.Result) lastInitOp.operands().getFirst();
                var lastInitOpOperand0ResultOp = lastInitOpOperand0Result.op();
                CoreOp.VarOp varOps[];
                if (lastInitOpOperand0ResultOp instanceof CoreOp.TupleOp tupleOp) {
                     /*
                     for (int j = 1, i=2, k=3; j < size; k+=1,i+=2,j+=3) {
                        float sum = k+i+j;
                     }
                     java.for
                     ()Tuple<Var<int>, Var<int>, Var<int>> -> {
                         %0 : int = constant @"1";
                         %1 : Var<int> = var %0 @"j";
                         %2 : int = constant @"2";
                         %3 : Var<int> = var %2 @"i";
                         %4 : int = constant @"3";
                         %5 : Var<int> = var %4 @"k";
                         %6 : Tuple<Var<int>, Var<int>, Var<int>> = tuple %1 %3 %5;
                         yield %6;
                     }
                     (%7 : Var<int>, %8 : Var<int>, %9 : Var<int>)boolean -> {
                         %10 : int = var.load %7;
                         %11 : int = var.load %12;
                         %13 : boolean = lt %10 %11;
                         yield %13;
                     }
                     (%14 : Var<int>, %15 : Var<int>, %16 : Var<int>)void -> {
                         %17 : int = var.load %16;
                         %18 : int = constant @"1";
                         %19 : int = add %17 %18;
                         var.store %16 %19;
                         %20 : int = var.load %15;
                         %21 : int = constant @"2";
                         %22 : int = add %20 %21;
                         var.store %15 %22;
                         %23 : int = var.load %14;
                         %24 : int = constant @"3";
                         %25 : int = add %23 %24;
                         var.store %14 %25;
                         yield;
                     }
                     (%26 : Var<int>, %27 : Var<int>, %28 : Var<int>)void -> {
                         %29 : int = var.load %28;
                         %30 : int = var.load %27;
                         %31 : int = add %29 %30;
                         %32 : int = var.load %26;
                         %33 : int = add %31 %32;
                         %34 : float = conv %33;
                         %35 : Var<float> = var %34 @"sum";
                         java.continue;
                     };
                     */
                    varOps = tupleOp.operands().stream().map(operand -> (CoreOp.VarOp) (((Op.Result) operand).op())).toList().toArray(new CoreOp.VarOp[0]);
                } else {
                     /*
                     for (int j = 0; j < size; j+=1) {
                        float sum = j;
                     }
                     java.for
                        ()Var<int> -> {
                            %0 : int = constant @"0";
                            %1 : Var<int> = var %0 @"j";
                            yield %1;
                        }
                        (%2 : Var<int>)boolean -> {
                            %3 : int = var.load %2;
                            %4 : int = var.load %5;
                            %6 : boolean = lt %3 %4;
                            yield %6;
                        }
                        (%7 : Var<int>)void -> {
                            %8 : int = var.load %7;
                            %9 : int = constant @"1";
                            %10 : int = add %8 %9;
                            var.store %7 %10;
                            yield;
                        }
                        (%11 : Var<int>)void -> {
                            %12 : int = var.load %11;
                            %13 : float = conv %12;
                            %14 : Var<float> = var %13 @"sum";
                            java.continue;
                        };

                     */
                    varOps = new CoreOp.VarOp[]{(CoreOp.VarOp) lastInitOpOperand0ResultOp};
                }
                for (int i = 0; i < varOps.length; i++) {
                    blockParamToVarOpMap.put(condParams[i], varOps[i]);
                    blockParamToVarOpMap.put(updateParams[i], varOps[i]);
                    blockParamToVarOpMap.put(loopParams[i], varOps[i]);
                }
            }


            @Override
            public CoreOp.VarOp resolve(Value value) {
                if (value instanceof Block.Parameter blockParameter) {
                    CoreOp.VarOp varOp = this.blockParamToVarOpMap.get(blockParameter);
                    if (varOp != null) {
                        return varOp;
                    }
                }
                return super.resolve(value);
            }
        }

        public static class IfScope extends Scope<IfOpWrapper> {
            IfScope(Scope<?> parent, IfOpWrapper opWrapper) {
                super(parent, opWrapper);
            }
        }

        public static class WhileScope extends LoopScope<WhileOpWrapper> {
            WhileScope(Scope<?> parent, WhileOpWrapper opWrapper) {
                super(parent, opWrapper);
            }

        }

        public Scope<?> scope = null;

        private void popScope() {
            scope = scope.parent;
        }

        private void pushScope(OpWrapper<?> opWrapper) {
            scope = switch (opWrapper) {
                case FuncOpWrapper $ -> new FuncScope(scope, $);
                case ForOpWrapper $ -> new ForScope(scope, $);
                case IfOpWrapper $ -> new IfScope(scope, $);
                case WhileOpWrapper $ -> new WhileScope(scope, $);
                default -> new Scope<>(scope, opWrapper);
            };
        }

        public void scope(OpWrapper<?> opWrapper, Runnable r) {
            pushScope(opWrapper);
            r.run();
            popScope();
        }

        public  FuncOpWrapper funcOpWrapper;

        public CodeBuilderContext(FuncOpWrapper funcOpWrapper) {
            this.funcOpWrapper = funcOpWrapper;
        }

    }

}
