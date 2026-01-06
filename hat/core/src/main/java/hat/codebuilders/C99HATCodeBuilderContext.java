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
package hat.codebuilders;

import hat.dialect.HATF16Op;
import hat.dialect.HATVectorOp;
import hat.types.HAType;
import hat.device.DeviceType;
import hat.dialect.HATMemoryVarOp;
import optkl.OpHelper;
import optkl.ifacemapper.MappableIface;
import optkl.util.Regex;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.codebuilders.BabylonOpDispatcher;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import static optkl.OpHelper.NamedOpHelper.Invoke.invokeOpHelper;

public abstract class C99HATCodeBuilderContext<T extends C99HATCodeBuilderContext<T>> extends C99HATCodeBuilder<T>
        implements BabylonOpDispatcher<T, ScopedCodeBuilderContext> {


    @Override
    public final T varLoadOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = buildContext.scope.resolve(varLoadOp.operands().getFirst());
        switch (resolve) {
            case CoreOp.VarOp $ -> varName($);
            case HATMemoryVarOp $ -> varName($);
            case HATVectorOp.HATVectorVarOp $ -> varName($);
            case HATVectorOp.HATVectorLoadOp $ -> varName($);
            case HATVectorOp.HATVectorBinaryOp $ -> varName($);
            case HATF16Op.HATF16VarOp $ -> varName($);
            case null, default -> {
            }
        }
        return self();
    }

    @Override
    public final T varStoreOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        Op op = buildContext.scope.resolve(varStoreOp.operands().getFirst());

        //TODO see if VarLikeOp marker interface fixes this

        // TODO: each of these is delegating to varName().... maybe varName should be handling these types.

        // When the op is intended to operate as VarOp, then we need to include it in the following switch.
        // This is because HAT has its own dialect, and some of the Ops operate on HAT Types (not included in the Java
        // dialect). For instance, private data structures, local data structures, vector types, etc.
        switch (op) {
            case CoreOp.VarOp varOp -> varName(varOp);
            case HATF16Op.HATF16VarOp hatf16VarOp -> varName(hatf16VarOp);
            case HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp -> varName(hatPrivateInitVarOp);
            case HATMemoryVarOp.HATPrivateVarOp hatPrivateVarOp -> varName(hatPrivateVarOp);
            case HATMemoryVarOp.HATLocalVarOp hatLocalVarOp -> varName(hatLocalVarOp);
            case HATVectorOp.HATVectorVarOp hatVectorVarOp -> varName(hatVectorVarOp);
            case null, default -> throw new IllegalStateException("What type of varStoreOp is this?");
        }
        equals().parenthesisIfNeeded(buildContext, varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }

    @Override
    public final  T convOp(ScopedCodeBuilderContext buildContext, JavaOp.ConvOp convOp) {
        // TODO: I think we need to work out how to handle doubles. If I remove this OpenCL on MAC complains (no FP64)
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(buildContext,JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type(buildContext,(JavaType)convOp.resultType()));
        }
        parenthesisIfNeeded(buildContext, convOp, ((Op.Result) convOp.operands().getFirst()).op());
        return self();
    }

    public abstract  T atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name);

    static Regex atomicIncRegex = Regex.of("(atomic.*)Inc");

    @Override
    public final T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        var invoke = invokeOpHelper(buildContext.lookup,invokeOp);
        if ( invoke.refIs(MappableIface.class,HAType.class,DeviceType.class)) { // we need a common type
            if (invoke.isInstance() && invoke.operandCount() == 1 && invoke.returnsInt() && invoke.named(atomicIncRegex)) {
                if (invoke.operandNAsResultOrThrow(0) instanceof Op.Result instanceResult) {
                    atomicInc(buildContext, instanceResult,
                            ((Regex.Match)atomicIncRegex.is(invoke.name())).stringOf(1) // atomicXXInc -> atomicXX
                    );
                }
            } else if (invoke.isInstance() && invoke.operandNAsResultOrThrow(0) instanceof Op.Result instance) {
                parenWhen(
                        invoke.operandCount() > 1
                                && invokeOpHelper(buildContext.lookup,instance.op()) instanceof OpHelper.NamedOpHelper.Invoke invoke0
                                && invoke0.returnsClassType()
                        ,
                   // When we have patterns like:
                   //
                   // myiFaceArray.array().value(storeAValue);
                   //
                   // We need to generate extra parenthesis to make the struct pointer accessor "->" correct.
                   // This is a common pattern when we have a IFace type that contains a subtype based on
                   // struct or union.
                   // An example of this is for the type F16Array.
                   // The following expression checks that the current invokeOp has at least 2 operands:
                    // Why 2?
                    // - The first one is another invokeOp to load the inner struct from an IFace data structure.
                    //   The first operand is also assignable.
                    // - The second one is the store value, but this depends on the semantics and definition
                    //   of the user code.
                    _->{
                    when(invoke.returnsClassType(), _ -> ampersand());
                    recurse(buildContext, instance.op());
                });

                // Check if the varOpLoad that could follow corresponds to a local/private type
                boolean isLocalOrPrivateDS = (instance.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                        && buildContext.scope.resolve(varLoadOp.operands().getFirst()) instanceof HATMemoryVarOp);

                either(isLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow);

                funcName(invoke.op());

                if (invoke.returnsVoid()) {//   setter
                    switch (invoke.operandCount()) {
                        case 2 -> {
                            if (invoke.opFromOperandNAsResultOrNull(1) instanceof Op op) {
                                equals().recurse(buildContext, op);
                            }
                        }
                        case 3-> {
                            if ( invoke.opFromOperandNAsResultOrThrow(1) instanceof Op op1
                                 && invoke.opFromOperandNAsResultOrThrow(2) instanceof Op op2) {
                                 sbrace(_ -> recurse(buildContext, op1)).equals().recurse(buildContext, op2);
                            }
                        }
                        default -> throw new IllegalStateException("How ");
                    }
                } else {
                    if (invoke.opFromOperandNAsResultOrNull(1) instanceof Op op) {
                        sbrace(_ -> recurse(buildContext, op));
                    }else{
                            // this is just call.
                    }
                }
            }
        } else {// General case
            funcName(invoke.op()).paren(_ ->
                    commaSpaceSeparated(invoke.op().operands(),
                            op -> {if (op instanceof Op.Result result) {recurse(buildContext, result.op());}
                    })
            );
        }
        return self();
    }

}
