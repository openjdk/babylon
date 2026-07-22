/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.phases;

import hat.HATMath;
import hat.device.NonMappableIface;
import hat.types.S16ImplOfF16;
import hat.types.Tensor;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.VarTable;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.util.Optional;
import java.util.SequencedSet;
import java.util.Set;
import java.util.stream.Stream;

import static java.lang.invoke.MethodHandles.lookup;
import static optkl.IfaceValue.Vector.getVectorShape;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.VarAccess.varAccess;
import static optkl.OpHelper.resultFromFirstOperandOrNull;

public class HATPhaseUtils {

    public static final Set<String> NON_MAPPABLE_IFACE = Set.of("createshared", "createlocal", "createprivate");

    public static Op findOpInResultFromFirstOperandsOrNull(Op op, Class<?>... classes) {
        Set<Class<?>> set = Set.of(classes);
        while (!set.contains(op.getClass())) {
            if (resultFromFirstOperandOrNull(op) instanceof Op.Result result) {
                op = result.op();
            } else {
                return null;
            }
        }
        return op;
    }

    public static Class<?> reduceFloatType(Optional<OpHelper.Invoke> invoke) {
        if (invoke.isPresent() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().refType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
    }

    public static Class<?> reduceFloatTypeFromReturnType(Optional<OpHelper.Invoke> invoke) {
        if (invoke.isPresent() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().returnType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
    }

    public record InvokeVar(JavaOp.InvokeOp invokeOp, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        // recursive
        public static String vectorNameOrThrow(Value v) {
            return switch (OpHelper.asOpFromResultOrNull(v)) {
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->
                        vectorNameOrThrow(varLoadOp.operands().getFirst()); // recurse
                case CoreOp.VarOp varOp -> varOp.varName();
                case null -> null;
                default -> throw new IllegalStateException("failed to find vector name");
            };
        }

        public String name() {
            return vectorNameOrThrow(varLoadOp.operands().getFirst());
        }

        //recursive
        public CoreOp.VarOp findVarOpOrNull(Value v) {
            return switch (OpHelper.asOpFromResultOrNull(v)) {
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->
                        findVarOpOrNull(varLoadOp.operands().getFirst()); //recurse
                case CoreOp.VarOp varOp -> varOp;
                case null -> null;
                default -> null;
            };
        }

        public CoreOp.VarOp varOpFromOperand(int idx) {
            return findVarOpOrNull(invokeOp.operands().get(idx));
        }

        public CodeType returnType() {
            return invokeOp.resultType();
        }

        public int laneIdx() {
            return "xyzw".indexOf(invokeOp.invokeReference().name().charAt(0));
        }

        public String resolveName() {
            return varOpFromOperand(1) instanceof CoreOp.VarOp varOp ? varOp.varName() : null;
        }
    }

    public static boolean isVectorOperation(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        OpHelper.Invoke invoke = invoke(lookup, invokeOp);
        return invoke.returns(IfaceValue.Vector.class) && invoke.nameMatchesRegex(OpHelper.RESERVED_METHOD_VECTORS);
    }

    public static boolean isSharedOrPrivate(MethodHandles.Lookup lookup, Op op) {
        return isSharedOrPrivate(lookup, op.operands().getFirst());
    }

    public static boolean isSharedOrPrivate(MethodHandles.Lookup lookup, Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isSharedOrPrivate(lookup, varLoadOp); //recurse
            case CoreOp.VarOp varOp -> {
                // extra analysis
                Value first = varOp.operands().getFirst();
                if (first instanceof Block.Parameter) {
                    // if the var comes from a parameter, then it is global memory
                    yield false;
                }
                // otherwise we continue traversal
                yield isSharedOrPrivate(lookup, varOp);
            }
            case JavaOp.InvokeOp invoke -> {
                // If we get an invoke, we need to get method name, and check the following

                // warp to Invoke
                if (lookup == null) {
                    throw new IllegalStateException("Lookup has not been initialized");
                }

                Stream<OpHelper.Invoke> stream = OpHelper.Invoke.stream(lookup, invoke);
                Optional<OpHelper.Invoke> invokeOptional = stream.findFirst();
                // Check for the right class
                if (invokeOptional.isPresent() && invokeOptional.get().refIs(NonMappableIface.class)) {
                    // check for the method name
                    String lowerCase = invoke.invokeReference().name().toLowerCase();
                    yield NON_MAPPABLE_IFACE.contains(lowerCase);
                }
                yield false;
            }
            default -> false;
        };
    }

    // recursive
    public static String findVectorVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorVarNameOrNull(varLoadOp.operands().getFirst());
    }

    // recursive
    public static String findVectorVarNameOrNull(Value v) {
        switch (v) {
            case Op.Result r when r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                return findVectorVarNameOrNull(varLoadOp);
            }
            case null, default -> {
                if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarOp varOp) {
                    return varOp.varName();
                }
                return null;
            }
        }
    }

    public static String findVarNameOrNull(Value v) {
        return (v instanceof Op.Result r) ? switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> findVarNameOrNull(varLoadOp); //recurse
            case CoreOp.VarOp varOp -> varOp.varName();
            default -> null;
        } : null;
    }

    public static String findVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVarNameOrNull(varLoadOp.operands().getFirst());
    }

    public static boolean isMathOperation(OpHelper.Invoke invoke) {
        return !invoke.returnsVoid() && invoke.refIs(HATMath.class);
    }

    private static boolean is16BitFloat(OpHelper.Invoke invoke, Regex methodName) {
        return invoke.refIs(S16ImplOfF16.class) && invoke.nameMatchesRegex(methodName);
    }

    public static boolean isS16BinaryOp(OpHelper.Invoke invoke) {
        return is16BitFloat(invoke, Regex.of("(add|sub|mul|div)")) && !invoke.returnsVoid();
    }

    public static boolean isTensorOperation(OpHelper.Invoke invoke) {
        if (isTensorCreate(invoke) || isTensorFillOperation(invoke) || isTensorShape(invoke) || isTensorStore(invoke)) {
            return true;
        }
        return isReturnTensorValueOperation(invoke);
    }

    public static boolean isTensorCreate(OpHelper.Invoke invoke) {
        return !invoke.returnsVoid() && invoke.refIs(HATTensorsPhase.TensorMarkers.class) && invoke.nameMatchesRegex("create|of");
    }

    public static boolean isTensorFillOperation(OpHelper.Invoke invoke) {
        return invoke.returnsVoid() && invoke.refIs(Tensor.class) && invoke.nameMatchesRegex("fill");
    }

    public static boolean isTensorShape(OpHelper.Invoke invoke) {
        return !invoke.returnsVoid() && invoke.refIs(Tensor.Shape.class) && invoke.nameMatchesRegex("shape");
    }

    public static boolean isTensorStore(OpHelper.Invoke invoke) {
        return invoke.returnsVoid() && invoke.refIs(Tensor.class) && invoke.nameMatchesRegex("store");
    }

    public static boolean isReturnTensorValueOperation(OpHelper.Invoke invoke) {
        return !invoke.returnsVoid() && invoke.refIs(Tensor.class) && invoke.nameMatchesRegex("create|zeros|shape|load|loadF16|mma");
    }

    public static boolean isVectorSelectOperation(OpHelper.Invoke invoke) {
        return invoke.nameMatchesRegex("[xyzw]") && invoke.refIs(IfaceValue.Vector.class) && invoke.opFromFirstOperandOrThrow() instanceof CoreOp.VarAccessOp.VarLoadOp;
    }

    public static boolean isS16Conversion(OpHelper.Invoke invoke) {
        return !invoke.returnsVoid() && is16BitFloat(invoke, Regex.of("(of|floatToF16|float2bfloat16)")) && invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp;
    }

    public static boolean isS16ToFloatConversion(OpHelper.Invoke invoke) {
        return invoke instanceof OpHelper.Invoke.Static && invoke.nameMatchesRegex("(f16ToFloat|bfloat162float)") && invoke.returnsFloat();
    }

    public static boolean isAttributeSharedOrPrivate(VarTable.HATOpAttribute attribute, OpHelper.Invoke invoke) {
        if (attribute == VarTable.HATOpAttribute.INIT_SHARED || attribute == VarTable.HATOpAttribute.PRIVATE || attribute == VarTable.HATOpAttribute.SHARED) {
            return true;
        } else return attribute == VarTable.HATOpAttribute.NARROW && !invoke.returnsVoid();
    }

    public static boolean isInvokeLoadingFromOnChipMemory(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        OpHelper.Invoke invoke = invoke(lookup, invokeOp);
        if (invoke.refIs(NonMappableIface.class) && invoke.returnsClassType() && !invoke.nameMatchesRegex(OpHelper.RESERVED_METHODS_MEMORY_REGIONS)) {
            SequencedSet<Op.Result> uses = invoke.op().result().uses();
            return uses.stream().filter(use -> use.op() instanceof CoreOp.VarOp)
                    .map(use -> (CoreOp.VarOp) use.op()).anyMatch(_ -> true);
        }
        return false;
    }

    public static boolean isVectorView(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        var invoke = invoke(lookup, invokeOp);
        return (invoke.named("storeFloat4View") || invoke.named("storeFloat2View"))
                && varAccess(lookup, invoke.opFromOperandNOrNull(1)) instanceof OpHelper.VarAccess varAccess
                && varAccess.isLoad() && varAccess.isTypeAssignable(IfaceValue.Vector.class);
    }

    public static IfaceValue.Vector.Shape getVectorShapeFromOperandN(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, int idx) {
        if (invokeOp.operands().get(idx) instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            if (varLoadOp.resultType() instanceof VarType varType) {
                return getVectorShape(lookup, varType.valueType());
            } else {
                return getVectorShape(lookup, varLoadOp.resultType());
            }
        }
        return null;
    }

    public static boolean findIsSharedOrPrivateSpace(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findIsSharedOrPrivateSpace(varLoadOp.operands().getFirst());
        } else if (v.declaringElement() instanceof CoreOp.VarOp varOp) {
            return findIsSharedOrPrivateSpace(varOp.operands().getFirst());
        } else {
            return !(v instanceof Block.Parameter);
        }
    }

    public static String mapLane(int lane) {
        return switch (lane) {
            case 0 -> "x";
            case 1 -> "y";
            case 2 -> "z";
            case 3 -> "w";
            default -> throw new InternalError("Invalid lane: " + lane);
        };
    }

    public static boolean isOperandF32(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp varLoadOp -> varLoadOp.varType().valueType() == JavaType.FLOAT; //recurse
            case CoreOp.VarOp varOp -> varOp.resultType().valueType() == JavaType.FLOAT;
            default -> false;
        };
    }

    //recursive
    public static boolean isArrayReference(MethodHandles.Lookup lookup, Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isArrayReference(lookup, varLoadOp); // recurse
            case CoreOp.VarOp varOp -> varOp.operands().getFirst() instanceof Op.Result varOpResult
                    && invoke(lookup(), varOpResult.op()) instanceof OpHelper.Invoke invoke
                    && invoke.named("array")
                    && !isInvokeLoadingFromOnChipMemory(lookup, invoke.op());
            default -> false;
        };
    }

    //recursive
    public static boolean isArrayReference(MethodHandles.Lookup lookup, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isArrayReference(lookup, varLoadOp.operands().getFirst());
    }

    public static boolean isVectorBinaryOperation(OpHelper.Invoke invoke) {
        return (invoke.returns(IfaceValue.Vector.class) && invoke.nameMatchesRegex("(add|sub|mul|div)"));
    }

    public static boolean isF16Local(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isF16Local(varLoadOp); //recurse
            case CoreOp.VarOp varOp ->
                    !(varOp.operands().getFirst().declaringElement() instanceof JavaOp.InvokeOp invokeOp)
                            || !invokeOp.invokeReference().name().equals("array");
            default -> false;
        };
    }

    //recursive
    public static boolean isF16Local(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isF16Local(varLoadOp.operands().getFirst());
    }

    public static boolean isInvokeFromNarrowTypeConversion(MethodHandles.Lookup lookup, JavaOp.InvokeOp invoke) {
        SequencedSet<Op.Result> uses = invoke.result().uses();
        boolean[] result = new boolean[1];
        uses.forEach(usage -> {
            if (usage.declaringElement() instanceof JavaOp.InvokeOp invokeOp2) {
                var invoke2 = invoke(lookup, invokeOp2);
                if (invoke2.nameMatchesRegex("(f16ToFloat|bfloat162float)")) {
                    result[0] = true;
                }
            }
        });
        return result[0];
    }

    public static boolean isMathLib(Optional<OpHelper.Invoke> invoke) {
        return invoke.isPresent() && !invoke.get().returnsVoid() && invoke.get().returnsClassType() && invoke.get().refIs(HATMath.class);
    }

    private HATPhaseUtils() {
        /* This utility class should not be instantiated */
    }
}
