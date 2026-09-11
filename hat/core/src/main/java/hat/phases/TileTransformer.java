/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import hat.DType;
import hat.TileContext;
import hat.TileOp;
import hat.codetypes.ConstantType;
import hat.codetypes.IndexType;
import hat.codetypes.PtrType;
import hat.codetypes.ShapeType;
import hat.codetypes.TensorType;
import hat.dialect.ArithMathOps;
import hat.dialect.TileOps;
import hat.types.Tile;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.SSA;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import jdk.incubator.code.extern.ExternalizedOp;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

/**
 * Class to perform:
 * a) Type Checks and shape propagation for Tile operations.
 * b) Build custom code model (with dialect) that supports a Tile Programming Model.
 */
public class TileTransformer {

    private static final boolean PRINT_INTERNALS = Boolean.parseBoolean(System.getProperty("PRINT_INTERNALS"));
    private static final boolean LOWER_TO_SSA = Boolean.parseBoolean(System.getProperty("LOWER_TO_SSA", "FALSE"));
    private static final boolean SIMPLE_SIGNATURE = Boolean.parseBoolean(System.getProperty("SIMPLE_SIGNATURE", "TRUE"));

    public static TileOps.ModuleOp dispatch(Class<?> klass, String methodName, List<? extends CodeType> argTypes, MethodHandles.Lookup lookup) {
        Optional<Method> method = Stream.of(klass.getDeclaredMethods())
                .filter(m -> m.getName().equals(methodName))
                .filter(m -> m.getAnnotation(Reflect.class) != null)
                .findFirst();

        CoreOp.FuncOp funcOp = Op.ofMethod(method.orElseThrow()).get();
        funcOp = processConstantFields(funcOp, lookup);

        // Verify types and shapes from the input Tile Kernel and generate a new code model (dialect for Tile)
        // A tile kernel always returns VOID
        return TileTransformer.tileModule(funcOp, JavaType.VOID, argTypes);
    }

    /**
     * Method to perform checks on the shapes and build the corresponding tensors with the correct dimensions and shapes across
     * all operations.
     *
     * @param kernel   Input Kernel
     * @param rType    Return Type
     * @param argTypes Arguments to the Tile Kernel
     * @param <O>      Op type
     * @return Module {@link TileOps.ModuleOp}
     */
    public static <O extends Op & Op.Invokable> TileOps.ModuleOp tileModule(O kernel, CodeType rType, List<? extends CodeType> argTypes) {
        Map<String, CoreOp.FuncOp> symbolTable = new LinkedHashMap<>();
        tileFunction(kernel, rType, argTypes, symbolTable);
        return TileOps.module(symbolTable.values().stream().toList());
    }

    public static <O extends Op & Op.Invokable> CoreOp.FuncOp tileFunction(O kernel, CodeType rType, List<? extends CodeType> argTypes) {
        Map<String, CoreOp.FuncOp> symbolTable = new LinkedHashMap<>();
        return tileFunction(kernel, rType, argTypes, symbolTable);
    }

    /**
     * Process the code tree to find constants that are introduced via the scope of the function being analyzed.
     *
     * @param funcOp Input function
     * @param lookup Method lookup
     * @return A new function with the replacement of FieldLoads/constant with its constant value.
     */
    public static CoreOp.FuncOp processConstantFields(CoreOp.FuncOp funcOp, MethodHandles.Lookup lookup) {
        // Preprocessing constants
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp && fieldLoadOp.operands().isEmpty()) {
                if (fieldLoadOp.resultType() instanceof PrimitiveType primitiveType && primitiveType.equals(PrimitiveType.INT)) {
                    // Found the int field. we can replace it with a constant value
                    try {
                        VarHandle field = lookup.findStaticVarHandle(lookup.lookupClass(), fieldLoadOp.fieldReference().name(), int.class);
                        // We can pass null because, at this point, we know it is a static field
                        int intValue = (int) field.get();
                        CoreOp.ConstantOp constantOp = CoreOp.constant(primitiveType, intValue);
                        Op.Result constantValue = blockBuilder.add(constantOp);
                        constantOp.setLocation(fieldLoadOp.location());
                        blockBuilder.context().mapValue(fieldLoadOp.result(), constantValue);
                    } catch (ReflectiveOperationException e) {
                        throw new RuntimeException(e);
                    }
                } else {
                    blockBuilder.add(fieldLoadOp);
                }
            } else {
                blockBuilder.add(op);
            }
            return blockBuilder;
        });
        return funcOp;
    }

    /**
     * Process the tile function. It first checks all shapes and builds the code model with custom ops for supporting the Tile Programming Model.
     *
     * @param kernel      Input Kernel
     * @param rType       Return type
     * @param argTypes    Arguments to the Tile Kernel
     * @param symbolTable Symbol Table
     * @param <O>
     * @return A new function which includes the code tree in the Tile format (dialect)
     */
    private static <O extends Op & Op.Invokable> CoreOp.FuncOp tileFunction(O kernel, CodeType rType, List<? extends CodeType> argTypes, Map<String, CoreOp.FuncOp> symbolTable) {

        String signature = (kernel instanceof CoreOp.FuncOp f) ? f.funcName() : "kernel";
        if (!SIMPLE_SIGNATURE) {
            signature = signature(signature, rType, argTypes);
        }

        if (symbolTable.containsKey(signature)) {
            return symbolTable.get(signature);
        }

        Map<Value, CodeType> valueTypeMap = new HashMap<>();
        Map<Op, Object> opData = new HashMap<>();

        // Check for types and propagate the shapes across the model
        typeCheckKernel(kernel, argTypes, valueTypeMap, opData);

        if (PRINT_INTERNALS) {
            // Print the intermediate step
            printTypeMap(kernel, valueTypeMap);
        }

        // Once we have verified the input code model to have the correct shapes and propagated those shapes for the new operations,
        // we generate a new code model. For the new code model, it uses the symbol table built in the previous process.
        return transformToTileFunction(kernel, signature, rType, valueTypeMap, opData, symbolTable);
    }

    private static <O extends Op & Op.Invokable> CoreOp.FuncOp transformToTileFunction(O kernel, String signature, CodeType rType, Map<Value, CodeType> valueTypeMap, Map<Op, Object> opData, Map<String, CoreOp.FuncOp> symbolTable) {
        // builds a function using the internal code reflection APIs
        CoreOp.FuncOp tileKernel = CoreOp.func(signature, CoreType.functionType(rType))
                .body(functionBlock -> {
                    // List of parameters for the kernel
                    List<Value> args = new ArrayList<>();
                    for (Block.Parameter param : kernel.body().entryBlock().parameters()) {
                        CodeType type = valueTypeMap.get(param);
                        if (type instanceof ConstantType c) {
                            // For all input constants, we built the Tile ConstantOp with its value
                            Op.Result r = functionBlock.add(ArithMathOps.constant(type, c.value()));
                            args.add(r);
                        } else {
                            args.add(functionBlock.parameter(type));
                        }
                    }
                    // kernel body
                    functionBlock.transformBody(kernel.body(), args, (kernelBlock, op) -> transformToTileOperation(kernelBlock, op, valueTypeMap, opData, symbolTable));
                });

        if (LOWER_TO_SSA) {
            tileKernel = lowerToSSA(tileKernel);
        }
        symbolTable.put(tileKernel.funcName(), tileKernel);
        return tileKernel;
    }

    private static Block.Builder transformToTileOperation(Block.Builder kblock, Op op, Map<Value, CodeType> valueTypeMap, Map<Op, Object> opData, Map<String, CoreOp.FuncOp> symbolTable) {
        TileBuilderInterpreter tileBuilderInterpreter = new TileBuilderInterpreter(symbolTable, kblock);
        CodeContext cc = kblock.context();
        switch (op) {

            case CoreOp.VarOp varOp -> {
                Value init = cc.getValue(op.operands().getFirst());
                Op.Result r = kblock.add(CoreOp.var(varOp.varName(), init));
                cc.mapValue(op.result(), r);
            }

            case CoreOp.ConstantOp constantOp -> {
                CodeType type = valueTypeMap.get(constantOp);
                if (type instanceof ConstantType c) {
                    Op.Result r = kblock.add(CoreOp.constant(type, c.value()));
                    cc.mapValue(op.result(), r);
                } else {
                    kblock.add(op);
                }
            }

            case JavaOp.InvokeOp iop when iop.invokeReference().refType().equals(TYPE_TILE_CONTEXT) -> {
                Value result = tileBuilderInterpreter.build(op, iop.invokeReference().name(), valueTypeMap);
                if (result != null) {
                    cc.mapValue(op.result(), result);
                }
            }
            case JavaOp.InvokeOp iop when iop.invokeReference().refType().equals(TYPE_TILE_MATH) -> {
                Value result = tileBuilderInterpreter.build(op, iop.invokeReference().name(), valueTypeMap);
                if (result != null) {
                    cc.mapValue(op.result(), result);
                }
            }
            case JavaOp.InvokeOp iop when iop.invokeReference().refType().equals(TYPE_TILE) -> {
                Value result = tileBuilderInterpreter.build(op, iop.invokeReference().name(), valueTypeMap);
                if (result != null) {
                    cc.mapValue(op.result(), result);
                }
            }
            default -> kblock.add(op);
        }
        return kblock;
    }

    /**
     * Lower the input model to SSA representation.
     *
     * @param funcOp Input code model in non-SSA format
     * @return Code model in SSA representation.
     */
    static CoreOp.FuncOp lowerToSSA(CoreOp.FuncOp funcOp) {
        CoreOp.FuncOp loweredCodeModel = funcOp.transform(CodeTransformer.LOWERING_TRANSFORMER);
        return SSA.transform(loweredCodeModel);
    }

    static class TileBuilderInterpreter {
        private final Map<String, CoreOp.FuncOp> symbolTable;
        private final Block.Builder block;

        TileBuilderInterpreter(Map<String, CoreOp.FuncOp> symbolTable, Block.Builder block) {
            this.symbolTable = symbolTable;
            this.block = block;
        }

        public Value build(Op op, String name, Map<Value, CodeType> valueTypeMap) {
            MethodHandle methodHandle;
            try {
                Optional<Method> om = Stream.of(TileBuilderInterpreter.class.getDeclaredMethods())
                        .filter(m -> m.getName().equals(name))
                        .filter(m -> m.isVarArgs()
                                ? m.getParameterCount() / 2 - 1 <= op.operands().size()
                                : m.getParameterCount() / 2 - 1 == op.operands().size())
                        .findFirst();
                methodHandle = MethodHandles.lookup().unreflect(om.orElseThrow(() -> new NoSuchMethodException(name)));
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException(e);
            }
            List<Object> arguments = new ArrayList<>();
            arguments.add(this);
            arguments.add(valueTypeMap.get(op.result()));
            arguments.add(op.result());
            for (Value o : op.operands()) {
                arguments.add(valueTypeMap.get(o));
                arguments.add(o);
            }
            try {
                return (Value) methodHandle.invokeWithArguments(arguments.toArray(Object[]::new));
            } catch (Throwable e) {
                throw new IllegalStateException(e);
            }
        }

        public Value BIDX(CodeType type, Op.Result result) {
            return block.add(TileOps.bid(0));
        }

        public Value BIDY(CodeType type, Op.Result result) {
            return block.add(TileOps.bid(1));
        }

        public Value BIDZ(CodeType type, Op.Result result) {
            return block.add(TileOps.bid(2));
        }

        public Value index(CodeType type, Op.Result result,
                           CodeType indexAType, Value indexA,
                           CodeType indexBType, Value indexB) {
            return block.add(TileOps.index(block.context().getValue(indexA), block.context().getValue(indexB)));
        }

        public Value shape(CodeType type, Op.Result result,
                           ConstantType shapeAType, Value shapeA) {
            return block.add(TileOps.shape(block.context().getValue(shapeA)));
        }

        public Value shape(CodeType type, Op.Result result,
                           ConstantType shapeAType, Value shapeA,
                           ConstantType shapeBType, Value shapeB) {
            return block.add(TileOps.shape(block.context().getValue(shapeA), block.context().getValue(shapeB)));
        }

        public Value shape(CodeType type, Op.Result result,
                           ConstantType shapeAType, Value shapeA,
                           ConstantType shapeBType, Value shapeB,
                           ConstantType shapeCType, Value shapeC) {
            return block.add(TileOps.shape(
                    block.context().getValue(shapeA),
                    block.context().getValue(shapeB),
                    block.context().getValue(shapeC)));
        }

        public Value load(CodeType type, Op.Result result,
                          PtrType ptrType, Value ptr,
                          CodeType dimensionType, Value dimension,
                          ConstantType shapeType, Value shape) {
            // Here we can perform some checks, for example, check shapes, check dimensions, etc.
            // Here we can perform some checks, for example, check shapes, check dimensions, etc.
            return block.add(TileOps.load(type, block.context().getValue(ptr), block.context().getValue(dimension), block.context().getValue(shape), ptrType.dims()));
        }

        public Value store(CodeType type, Op.Result result,
                           PtrType ptrType, Value ptr,
                           CodeType idType, Value id,
                           ConstantType tensorType, Value tensor) {
            // Here we can perform some checks, for example, check shapes, check dimensions, etc.
            return block.add(TileOps.store(block.context().getValue(ptr),
                    block.context().getValue(id),
                    block.context().getValue(tensor), ptrType.dims()));
        }

        public Value irange(CodeType resultType, Op.Result result,
                            ConstantType startIndex, Value start,
                            ConstantType endIndex, Value end) {
            return block.add(TileOps.irange(resultType, block.context().getValue(start), block.context().getValue(end)));
        }

        public Value irange(CodeType resultType, Op.Result result,
                            ConstantType endIndex, Value end) {
            CoreOp.ConstantOp constantOp = CoreOp.constant(JavaType.INT, 0);
            block.add(constantOp);
            return block.add(TileOps.irange(resultType, constantOp.result(), block.context().getValue(end)));
        }

        public Value add(CodeType type, Op.Result result,
                         CodeType typeA, Value tensorA,
                         CodeType typeB, Value tensorB) {
            tensorA = block.context().getValue(tensorA);
            tensorB = block.context().getValue(tensorB);
            if (type instanceof PtrType ptrType || type instanceof TensorType t && t.elementType() instanceof PtrType) {
                throw new IllegalStateException("Not supported yet");
            } else {
                return block.add(ArithMathOps.add(type, tensorA, tensorB));
            }
        }

        public Value sub(CodeType type, Op.Result result,
                         CodeType typeA, Value tensorA,
                         CodeType typeB, Value tensorB) {
            tensorA = block.context().getValue(tensorA);
            tensorB = block.context().getValue(tensorB);
            if (type instanceof PtrType ptrType || type instanceof TensorType t && t.elementType() instanceof PtrType) {
                throw new IllegalStateException("Not supported yet");
            } else {
                return block.add(ArithMathOps.sub(type, tensorA, tensorB));
            }
        }

        public Value mul(CodeType type, Op.Result result,
                         CodeType typeA, Value tensorA,
                         CodeType typeB, Value tensorB) {
            tensorA = block.context().getValue(tensorA);
            tensorB = block.context().getValue(tensorB);
            if (type instanceof PtrType ptrType || type instanceof TensorType t && t.elementType() instanceof PtrType) {
                throw new IllegalStateException("Not supported yet");
            } else {
                return block.add(ArithMathOps.mul(type, tensorA, tensorB));
            }
        }

        public Value cdiv(CodeType type, Op.Result result,
                          CodeType typeA, Value tensorA,
                          CodeType typeB, Value tensorB) {
            tensorA = block.context().getValue(tensorA);
            tensorB = block.context().getValue(tensorB);
            if (type instanceof PtrType ptrType || type instanceof TensorType t && t.elementType() instanceof PtrType) {
                throw new IllegalStateException("Not supported yet");
            } else {
                return block.add(ArithMathOps.cdiv(type, tensorA, tensorB));
            }
        }

        public Value ceildiv(CodeType type, Op.Result result,
                             CodeType typeA, Value valA,
                             CodeType typeB, Value valB) {
            valA = block.context().getValue(valA);
            valB = block.context().getValue(valB);
            if (type instanceof PtrType ptrType || type instanceof TensorType t && t.elementType() instanceof PtrType) {
                throw new IllegalStateException("Not supported yet");
            } else {
                return block.add(ArithMathOps.cdiv(type, valA, valB));
            }
        }

        public Value min(CodeType type, Op.Result result,
                         CodeType typeA, Value valA,
                         CodeType typeB, Value valB) {
            valA = block.context().getValue(valA);
            valB = block.context().getValue(valB);
            if (type instanceof PtrType ptrType || type instanceof TensorType t && t.elementType() instanceof PtrType) {
                throw new IllegalStateException("Not supported yet");
            } else {
                return block.add(ArithMathOps.min(type, valA, valB));
            }
        }

        public Value truediv(CodeType type, Op.Result result,
                             CodeType typeA, Value tensorA,
                             CodeType typeB, Value tensorB) {
            tensorA = block.context().getValue(tensorA);
            tensorB = block.context().getValue(tensorB);
            if (type instanceof PtrType ptrType || type instanceof TensorType t && t.elementType() instanceof PtrType) {
                throw new IllegalStateException("Not supported yet");
            } else {
                return block.add(ArithMathOps.truediv(type, tensorA, tensorB));
            }
        }

        public Value numTiles(CodeType type, Op.Result result,
                              PtrType ptrType, Value ptr,
                              ConstantType dimensionType, Value dimension,
                              ConstantType shapeType, Value shape) {
            return block.add(TileOps.numTiles(
                    block.context().getValue(ptr),
                    block.context().getValue(dimension),
                    block.context().getValue(shape)));
        }

        public Value full(CodeType typeResult, Op.Result result,
                          ConstantType shapeType, Value shape,
                          ConstantType valueType, Value val) {
            // TODO: We might need to inspect the shape to pass the dimensions as a field too.
            return block.add(TileOps.full(typeResult,
                    block.context().getValue(shape),
                    block.context().getValue(val)));
        }

        public Value asType(CodeType typeResult, Op.Result result,
                            CodeType tensorType, Value tensor,
                            CodeType toType, Value toTypeValue) {
            return block.add(TileOps.toType(typeResult, block.context().getValue(tensor)));
        }

        public Value zeros(CodeType typeResult, Op.Result result,
                           ConstantType shapeTypeA, Value shapeA,
                           ConstantType shapeTypeB, Value shapeB) {
            return block.add(TileOps.zeros(typeResult,
                    block.context().getValue(shapeA),
                    block.context().getValue(shapeB)));
        }

        public Value sum(CodeType typeResult, Op.Result result,
                         CodeType tileType, Value tile,
                         ConstantType dimensionType, Value dimension) {
            return block.add(TileOps.sum(typeResult,
                    block.context().getValue(tile),
                    block.context().getValue(dimension)));
        }

        public Value transpose(CodeType type, Op.Result result, CodeType typeMatrix, Value matrix) {
            return block.add(ArithMathOps.transpose(type, block.context().getValue(matrix)));
        }

        public Value reshape(CodeType type, Op.Result result,
                             CodeType tensorType, Value tensor,
                             ConstantType shapeType, Value shape) {
            return block.add(ArithMathOps.reshape(type, block.context().getValue(tensor)));
        }

        public Value permute(CodeType type, Op.Result result,
                             CodeType tensorType, Value tensor,
                             ConstantType shapeType, Value shape) {
            return block.add(ArithMathOps.permute(type, block.context().getValue(tensor)));
        }

        public Value mma(CodeType typeResult, Op.Result result,
                         CodeType tensorTypeA, Value tensorA,
                         CodeType tensorTypeB, Value tensorB,
                         CodeType tensorTypeC, Value tensorC) {
            return block.add(ArithMathOps.mma(block.context().getValue(tensorA),
                    block.context().getValue(tensorB),
                    block.context().getValue(tensorC)));
        }

        public Value arange(CodeType typeResult, Op.Result result,
                            ConstantType sizeType, Value size,
                            CodeType dType, Value dTypeValue) {
            return block.add(TileOps.arange(typeResult, block.context().getValue(size)));
        }

        public Value arange(CodeType typeResult, Op.Result result,
                            ConstantType sizeType, Value size) {
            return block.add(TileOps.arange(typeResult, block.context().getValue(size)));
        }

        public Value arange(CodeType typeResult, Op.Result result,
                            ConstantType sizeType, Value size,
                            ConstantType startType, Value start,
                            ConstantType stepType, Value step) {
            return block.add(TileOps.arange(typeResult,
                    block.context().getValue(size),
                    block.context().getValue(start),
                    block.context().getValue(step)));
        }

        public Value arange(CodeType typeResult, Op.Result result,
                            ConstantType sizeType, Value size,
                            ConstantType startType, Value start,
                            ConstantType stepType, Value step,
                            CodeType arangeType, Value arangeValue) {
            return block.add(TileOps.arange(typeResult,
                    block.context().getValue(size),
                    block.context().getValue(start),
                    block.context().getValue(step)));
        }
    }

    static final JavaType TYPE_TILE_CONTEXT = JavaType.type(TileContext.class);
    static final JavaType TYPE_TILE_MATH = JavaType.type(TileOp.class);
    static final JavaType TYPE_TILE = JavaType.type(Tile.class);
    static final JavaType TYPE_J_L_MATH = JavaType.type(Math.class);

    private static boolean isTypePromotionValid(CodeType fromType, CodeType toType) {
        if (fromType.equals(toType)) {
            return true;
        } else if (toType.equals(DType.TENSOR_2D_F32_TYPE) && fromType.equals(DType.TENSOR_2D_F16_TYPE)) {
            return true;
        } else if (toType.equals(DType.TENSOR_F32_TYPE) && fromType.equals(DType.TENSOR_F16_TYPE)) {
            return true;
        }
        throw new IllegalStateException("Type conversion not supported: " + fromType + " -> " + toType);
    }

    private static <O extends Op & Op.Invokable> void typeCheckKernel(O kernel, List<? extends CodeType> argTypes, Map<Value, CodeType> valueTypeMap, Map<Op, Object> opData) {
        kernel.elements().forEach(codeElement -> {
            if (!(codeElement instanceof Op op)) {
                return;
            }
            switch (op) {
                case Op.Invokable functionOp -> {
                    List<Block.Parameter> parameters = functionOp.body().entryBlock().parameters();
                    for (int i = 0; i < parameters.size(); i++) {
                        valueTypeMap.put(parameters.get(i), argTypes.get(i));
                    }
                }
                case CoreOp.VarOp _, CoreOp.VarAccessOp.VarLoadOp _ -> {
                    Value init = op.operands().getFirst();
                    // pass through this value using the valueTypeMap
                    valueTypeMap.put(op.result(), valueTypeMap.get(init));
                }
                case CoreOp.VarAccessOp.VarStoreOp varStoreOp -> {
                    Value var = varStoreOp.operands().getFirst();
                    CodeType varType = valueTypeMap.get(var);
                    Value v = op.operands().get(1);
                    CodeType vType = valueTypeMap.get(v);

                    // check type promotion for Tensors
                    if (varType instanceof ConstantType c && c.value() instanceof TensorType t1 && vType instanceof ConstantType c2 && c2.value() instanceof TensorType t2) {
                        CodeType toType = t1.elementType();
                        CodeType fromType = t2.elementType();
                        // Promotion: F16 to F32
                        if (!isTypePromotionValid(fromType, toType)) {
                            throw new IllegalArgumentException("incompatible types to be stored: " + varType + " != " + vType);
                        }
                    } else if (!varType.equals(vType)) {
                        throw new IllegalArgumentException("incompatible types to be stored: " + varType + " != " + vType);
                    }
                }
                case CoreOp.ConstantOp constantOp -> {
                    valueTypeMap.put(op.result(), new ConstantType(op.result().type(), constantOp.value()));
                }
                case JavaOp.InvokeOp iop when iop.invokeReference().refType().equals(TYPE_TILE_CONTEXT) -> {
                    CodeType t = checkWithTypeInterpreter(op, iop.invokeReference().name(), valueTypeMap);
                    valueTypeMap.put(op.result(), new ConstantType(op.result().type(), t));
                }
                case JavaOp.InvokeOp iop when iop.invokeReference().refType().equals(TYPE_TILE_MATH) -> {
                    CodeType t = checkWithTypeInterpreter(op, iop.invokeReference().name(), valueTypeMap);
                    valueTypeMap.put(op.result(), new ConstantType(op.result().type(), t));
                }
                case JavaOp.InvokeOp iop when iop.invokeReference().refType().equals(TYPE_TILE) -> {
                    CodeType t = checkWithTypeInterpreter(op, iop.invokeReference().name(), valueTypeMap);
                    valueTypeMap.put(op.result(), new ConstantType(op.result().type(), t));
                }
                case JavaOp.BinaryOp _, JavaOp.UnaryOp _ -> {
                    CodeType t = checkWithTypeInterpreter(op, externalizeOpName(op), valueTypeMap);
                    valueTypeMap.put(op.result(), t); // this is the way it should be done: shall we model a CodeType that is not constant?
                }
                case JavaOp.InvokeOp iop when iop.invokeReference().refType().equals(TYPE_J_L_MATH) -> {
                    for (Value v : op.operands()) {
                        valueTypeMap.put(op.result(), valueTypeMap.get(v));
                    }
                }
                case JavaOp.InvokeOp iop when isAssignable(MethodHandles.lookup(), iop.invokeReference().refType(), Number.class) -> {
                    if (iop.invokeReference().name().equals("valueOf")) {
                        Value v = op.operands().getFirst();
                        valueTypeMap.put(op.result(), valueTypeMap.get(v));
                    } else {
                        throw new IllegalArgumentException("incompatible types to be stored: " + iop.invokeReference());
                    }
                }
                case JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp -> {
                    if (!fieldLoadOp.operands().isEmpty()) {
                        throw new IllegalStateException("field load op not supported");
                    }
                    Field f;
                    try {
                        f = fieldLoadOp.fieldReference().resolveToField(MethodHandles.lookup());
                    } catch (ReflectiveOperationException e) {
                        throw new IllegalStateException("could not resolve field load op", e);
                    }
                    Object valueField;
                    try {
                        valueField = f.get(null);
                    } catch (IllegalAccessException e) {
                        throw new IllegalStateException("could not access field load op", e);
                    }
                    valueTypeMap.put(op.result(), new ConstantType(JavaType.type(f.getType()), valueField));
                }
                case JavaOp.CompareOp _ -> {
                }
                case CoreOp.ReturnOp _ -> {
                }
                case JavaOp.ForOp forLoopOp -> {
                    // taken from the Triton experiment
                    CodeType t = forLoopOp.initBody().yieldType();
                    if (t instanceof VarType varType && varType.valueType().equals(JavaType.INT)) {
                        for (Body b : List.of(forLoopOp.condBody(), forLoopOp.updateBody(), forLoopOp.loopBody())) {
                            valueTypeMap.put(b.entryBlock().parameters().getFirst(), JavaType.INT);
                        }
                    } else {
                        throw new IllegalArgumentException("incompatible types to be stored: " + forLoopOp);
                    }
                }
                case JavaOp.EnhancedForOp enhancedForOp -> {
                    // taken from the Triton experiment
                    CodeType t = enhancedForOp.initBody().yieldType();
                    if (t instanceof VarType varType && varType.valueType().equals(JavaType.INT)) {
                        for (Body b : List.of(enhancedForOp.loopBody())) {
                            valueTypeMap.put(b.entryBlock().parameters().getFirst(), JavaType.INT);
                        }
                    } else {
                        throw new IllegalArgumentException("incompatible types to be stored: " + enhancedForOp);
                    }
                }
                case CoreOp.YieldOp _ -> {

                }
                case JavaOp.ContinueOp _ -> {

                }
                default -> throw new IllegalStateException("Unexpected value: " + op);
            }
        });
    }

    static Type classTypeToTypeOrThrow(MethodHandles.Lookup lookup, ClassType classType) {
        try {
            return classType.resolve(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static boolean isAssignable(MethodHandles.Lookup lookup, CodeType codeType, Class<?>... classes) {
        if (codeType instanceof ClassType classType) {
            Type type = classTypeToTypeOrThrow(lookup, classType);
            return Arrays.stream(classes).anyMatch(clazz -> clazz.isAssignableFrom((Class<?>) type));
        } else if (codeType instanceof PrimitiveType) {
            return Arrays.stream(classes).anyMatch(clazz ->
                    (codeType == JavaType.FLOAT && clazz.equals(float.class))
                            || (codeType == JavaType.DOUBLE && clazz.equals(double.class))
                            || (codeType == JavaType.INT && clazz.equals(int.class))
                            || (codeType == JavaType.LONG && clazz.equals(long.class))
                            || (codeType == JavaType.SHORT && clazz.equals(short.class))
                            || (codeType == JavaType.CHAR && clazz.equals(char.class))
                            || (codeType == JavaType.BYTE && clazz.equals(byte.class))
                            || (codeType == JavaType.BOOLEAN && clazz.equals(boolean.class))
                            || (codeType == JavaType.VOID && clazz.equals(void.class))
            );
        }
        return false;
    }

    static String externalizeOpName(Op op) {
        return (op instanceof ExternalizedOp.Externalizable externalizable) ? externalizable.externalizeOpName() : op.getClass().getName();
    }

    // Transform the kernel name and arguments into the a kernel signature
    static String signature(String name, CodeType rType, List<? extends CodeType> argTypes) {
        StringBuilder sb = new StringBuilder(name);

        for (CodeType argType : argTypes) {
            sb.append("_");
            if (argType instanceof ConstantType ct) {
                sb.append(ct.value());
            } else {
                sb.append(argType);
            }
        }
        sb.append("_");
        sb.append(rType);
        return sb.toString();
    }

    public static <O extends Op & Op.Invokable> void printTypeMap(O kernel, Map<Value, CodeType> valueTypeMap) {
        AtomicInteger valueId = new AtomicInteger();
        Map<Value, Integer> valueIdMap = new LinkedHashMap<>();
        kernel.elements().forEach(codeElement -> {
            switch (codeElement) {
                case CoreOp.FuncOp _ -> {
                    // Ignore
                }
                case Op op when !op.result().type().equals(JavaType.VOID) -> {
                    valueIdMap.put(op.result(), valueId.getAndIncrement());
                }
                case Block block -> {
                    for (Block.Parameter parameter : block.parameters()) {
                        valueIdMap.put(parameter, valueId.getAndIncrement());
                    }
                }
                default -> {
                }
            }
        });

        valueIdMap.forEach((value, id) -> {
            CodeType type = valueTypeMap.get(value);
            if (type != null) {
                System.out.println("%" + id + " : " + value.type() + " -> " + type);
            }
        });
    }

    static CodeType checkWithTypeInterpreter(Op op, String name, Map<Value, CodeType> valueTypeMap) {
        MethodHandle mh;
        try {
            Optional<Method> optionalMethod = Stream.of(TileTypeInterpreter.class.getDeclaredMethods())
                    .filter(m -> m.getName().equals(name))
                    .filter(m -> m.isVarArgs() ? m.getParameterCount() <= op.operands().size() : m.getParameterCount() == op.operands().size())
                    .findFirst();
            mh = MethodHandles.lookup().unreflect(optionalMethod.orElseThrow(() -> new NoSuchMethodException(name)));
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }

        List<CodeType> operandTypes = op.operands().stream().map(valueTypeMap::get).toList();
        try {
            return (CodeType) mh.invokeWithArguments(operandTypes.toArray(Object[]::new));
        } catch (Throwable e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    static class TileTypeInterpreter {
        private TileTypeInterpreter() {
        }

        // int bid(Constant int dimension)
        public static JavaType BIDX() {
            return JavaType.INT;
        }

        public static JavaType BIDY() {
            return JavaType.INT;
        }

        public static JavaType BIDZ() {
            return JavaType.INT;
        }

        public static IndexType index(ConstantType indexA) {
            return new IndexType(indexA);
        }

        public static IndexType index(CodeType indexA, CodeType indexB) {
            return new IndexType(indexA, indexB);
        }

        public static ShapeType shape(ConstantType shapeA) {
            return new ShapeType(shapeA);
        }

        public static ShapeType shape(ConstantType shapeA, ConstantType shapeB) {
            return new ShapeType(shapeA, shapeB);
        }

        public static ShapeType shape(ConstantType shapeA, ConstantType shapeB, ConstantType shapeC) {
            return new ShapeType(shapeA, shapeB, shapeC);
        }

        public static TensorType load(PtrType ptr, CodeType dimension, ConstantType shape) {
            if (shape.value() instanceof ShapeType shapeType) {
                return new TensorType(ptr.rType(), shapeType.list());
            } else {
                return new TensorType(ptr.rType(), List.of((Integer) shape.value()));
            }
        }

        // store(arrayBuffer, id, tensor)
        public static void store(PtrType ptr, CodeType id, ConstantType tensor) {

        }

        public static JavaType irange(CodeType startIndex, CodeType endIndex) {
            return JavaType.INT;
        }

        public static JavaType irange(CodeType endIndex) {
            return JavaType.INT;
        }

        public static CodeType add(CodeType t1, CodeType t2) {
            return binary(inferConstantType(t1), inferConstantType(t2));
        }

        public static CodeType sub(CodeType t1, CodeType t2) {
            return binary(inferConstantType(t1), inferConstantType(t2));
        }

        public static CodeType mul(CodeType t1, CodeType t2) {
            return binary(inferConstantType(t1), inferConstantType(t2));
        }

        public static CodeType div(CodeType t1, CodeType t2) {
            return binary(inferConstantType(t1), inferConstantType(t2));
        }

        public static CodeType mod(CodeType t1, CodeType t2) {
            return binary(inferConstantType(t1), inferConstantType(t2));
        }

        public static JavaType ceildiv(CodeType t1, CodeType t2) {
            return JavaType.INT;
        }

        public static JavaType min(CodeType t1, CodeType t2) {
            return JavaType.INT;
        }


        public static JavaType numTiles(PtrType ptr, ConstantType dimension, ConstantType tileSize) {
            return JavaType.INT;
        }

        public static CodeType full(ConstantType dimension, ConstantType value) {
            CodeType codeType = inferConstantType(dimension);
            if (codeType instanceof ShapeType shapeType) {
                return new TensorType(JavaType.FLOAT, shapeType.list());
            } else {
                return new TensorType(JavaType.FLOAT, List.of((Integer) dimension.value()));
            }
        }

        public static CodeType zeros(ConstantType tileShapeA, ConstantType tileShapeB) {
            return new TensorType(DType.TENSOR_2D_F32_TYPE, List.of((Integer) tileShapeA.value(), (Integer) tileShapeB.value()));
        }

        public static CodeType asType(CodeType tensor, CodeType toType) {
            tensor = inferConstantType(tensor);
            if (tensor instanceof TensorType tensorType) {
                return new TensorType(toType, tensorType.shape());
            }
            throw new IllegalArgumentException("tensor must be a tensor type");
        }

        public static CodeType sum(CodeType tile, ConstantType axis) {
            // Shape after reduction
            return reduce(inferConstantType(tile), axis);
        }

        public static CodeType arange(ConstantType sizeType, CodeType type) {
            if (type instanceof ConstantType constantType && constantType.value() instanceof CodeType ct) {
                return new TensorType(ct, List.of((Integer) sizeType.value()));
            }
            throw new IllegalArgumentException("type must be a constant type");
        }

        // When dType is not passed, then we build a new Tile of type FLOAT
        public static CodeType arange(ConstantType sizeType) {
            return new TensorType(JavaType.FLOAT, List.of((Integer) sizeType.value()));
        }

        public static CodeType arange(ConstantType sizeType, ConstantType start, ConstantType step, CodeType type) {
            return arange(sizeType, type);
        }

        public static CodeType arange(ConstantType sizeType, ConstantType start, ConstantType step) {
            return arange(sizeType);
        }

        public static CodeType binary(CodeType a, CodeType b) {
            if (a instanceof TensorType ta && b instanceof TensorType tb) {
                return checkTensorTypes(ta, tb);
            } else if (a instanceof TensorType ta) {
                return new TensorType(checkScalarTypes(ta.elementType(), b), ta.shape());
            } else if (b instanceof TensorType tb) {
                return new TensorType(checkScalarTypes(a, tb.elementType()), tb.shape());
            } else {
                return checkScalarTypes(a, b);
            }
        }

        public static TensorType transpose(CodeType matrix) {
            var tensor = inferConstantType(matrix);
            if (tensor instanceof TensorType tensorType) {
                List<Integer> shape = transposeShape(tensorType.shape());
                return new TensorType(tensorType.elementType(), shape);
            } else {
                throw new IllegalArgumentException("matrix must be a tensor type, but found " + matrix.getClass());
            }
        }

        public static TensorType reshape(CodeType codeType, ConstantType reshapeDims) {
            var tensor = inferConstantType(codeType);
            if (tensor instanceof TensorType tensorType && reshapeDims.value() instanceof ShapeType shapeDims) {
                List<Integer> shape = tensorType.shape();
                List<Integer> newShape = shapeDims.list();
                int originalShape = shape.stream().reduce(1, (a, b) -> a * b);
                int newShapeSize = newShape.stream().reduce(1, (a, b) -> a * b);
                if (originalShape != newShapeSize) {
                    throw new IllegalArgumentException("reshape dimensions must be the same size: " + originalShape + " != " + newShapeSize);
                }
                return new TensorType(tensorType.elementType(), newShape);
            }
            throw new IllegalArgumentException("Reshape input must be a tensor type, but found " + tensor.getClass());
        }

        public static TensorType permute(CodeType codeType, ConstantType reshapeDims) {
            var tensor = inferConstantType(codeType);
            if (tensor instanceof TensorType tensorType && reshapeDims.value() instanceof ShapeType permuteShape) {
                List<Integer> shape = tensorType.shape();
                List<Integer> permuteList = permuteShape.list();
                List<Integer> resultShape = new ArrayList<>();
                Set<Integer> check = new HashSet<>();
                if (shape.size() != permuteList.size()) {
                    throw new IllegalStateException("permute dimensions must be the same size: " + shape.size() + " != " + permuteList.size());
                }
                for (int i = 0; i < shape.size(); i++) {
                    if (check.contains(permuteList.get(i))) {
                        throw new IllegalStateException("The permute list can't contain duplicates");
                    }
                    if (permuteList.get(i) < 0 || permuteList.get(i) > shape.size()) {
                        throw new IllegalStateException("Permute values must be defined within the input tensor list shape indices: " + permuteList.get(i));
                    }
                    check.add(permuteList.get(i));
                    resultShape.add(shape.get(permuteList.get(i)));
                }
                return new TensorType(tensorType.elementType(), resultShape);
            }
            throw new IllegalArgumentException("Reshape input must be a tensor type, but found " + tensor.getClass());
        }

        public static TensorType mma(CodeType codeTypeA, CodeType codeTypeB, CodeType codeTypeC) {
            CodeType ta = inferConstantType(codeTypeA);
            CodeType tb = inferConstantType(codeTypeB);
            CodeType accA = inferConstantType(codeTypeC);

            // to perform a mma operation, the required shapes are as follows:
            // assuming tensorA = tm x tk
            //          codeTypeB = tk x tn
            // Then, the resulting (acc) must be of shape: tm x tk
            // We also assume that MMA operations are performed with 2D tile shapes

            if (ta instanceof TensorType tensorA && tb instanceof TensorType tensorB && accA instanceof TensorType tensorC) {
                List<Integer> dimA = tensorA.shape();
                List<Integer> dimB = tensorB.shape();
                List<Integer> dimC = tensorC.shape();

                // mma operations are performed on 2D tensors
                if (isMMADimensionCompatible(dimA, dimB, dimC) && isMMAShapeCompatible(dimA, dimB, dimC)) {
                    return new TensorType(tensorA.elementType(), tensorC.shape());
                } else {
                    throw new IllegalArgumentException("Incompatible tensor shapes for performing the MMA operation");
                }
            } else {
                throw new IllegalArgumentException("input types must be a tensorA type");
            }
        }

        static boolean isMMADimensionCompatible(List<Integer> dimA, List<Integer> dimB, List<Integer> dimC) {
            return dimA.size() == 2 && dimA.size() == dimB.size() && dimA.size() == dimC.size();
        }

        static boolean isMMAShapeCompatible(List<Integer> dimA, List<Integer> dimB, List<Integer> dimC) {
            return dimC.getFirst().equals(dimA.getFirst()) && dimC.get(1).equals(dimB.get(1)) && dimA.get(1).equals(dimB.get(0));
        }

        static TensorType checkTensorTypes(TensorType t1, TensorType t2) {
            List<Integer> dimensions = checkTensorShape(t1, t2);
            CodeType elementType = checkScalarTypes(t1.elementType(), t2.elementType());
            return new TensorType(elementType, dimensions);
        }

        static List<Integer> checkTensorShape(TensorType t1, TensorType t2) {
            if (t1.shape().size() != t2.shape().size()) {
                throw new IllegalStateException("Shapes must have the same length");
            }

            List<Integer> dimensions = new ArrayList<>();
            for (int i = 0; i < t1.shape().size(); i++) {
                int dimA = t1.shape().get(i);
                int dimB = t2.shape().get(i);

                int dim;
                if (dimA == dimB) {
                    // there is no expansion
                    dim = dimA;
                } else {
                    // expand dimensions
                    if (dimA != 1 && dimB == 1) {
                        dim = dimA;
                    } else if (dimA == 1) {
                        dim = dimB;
                    } else {
                        // shape mismatch
                        throw new IllegalStateException("Shapes must have same size!");
                    }
                }
                dimensions.add(dim);
            }
            return dimensions;
        }

        static public CodeType checkScalarTypes(CodeType t1, CodeType t2) {
            if (t1 instanceof PtrType) {
                if (!t2.equals(JavaType.FLOAT)) {
                    throw new IllegalArgumentException("t1 must be of type float!");
                }
            } else if (t2 instanceof PtrType) {
                throw new IllegalArgumentException("The pointer must be the first argument");
            } else if (t1 instanceof ConstantType || t2 instanceof ConstantType) {
                return checkScalarTypes(reduceScalarType(t1), reduceScalarType(t2));
            } else if (t1 instanceof PrimitiveType && t2 instanceof ClassType classType) {
                // check for type equivalence
                if (classType.equals(ClassType.J_L_FLOAT) && t1.equals(JavaType.FLOAT)) {
                    return t1;
                } else if (classType.equals(DType.TENSOR_F32_TYPE) && t1.equals(JavaType.FLOAT)) {
                    return t1;
                } else {
                    throw new IllegalArgumentException("t1 vs t2 vs classType! " + t1 + " vs " + t2);
                }
            } else if (!t1.equals(t2)) {
                throw new IllegalArgumentException("t1 and t2 must be equal, but found `" + t1 + "` vs `" + t2 + "`");
            }
            return t1;
        }

        static CodeType reduceScalarType(CodeType codeType) {
            return codeType instanceof ConstantType constantType ? constantType.codeType() : codeType;
        }

        static CodeType reduce(CodeType tensor, ConstantType axis) {
            if (!axis.codeType().equals(JavaType.INT)) {
                throw new IllegalArgumentException("Axis must be of type int!");
            }
            // obtain the axis to perform the reduction
            int axisValue = (int) axis.value();
            if (axisValue < 0 || axisValue > 2) {
                throw new IllegalArgumentException("axis must be a positive integer between 0 and 2!");
            }

            if (tensor instanceof TensorType tensorType) {
                List<Integer> reduceShape = new ArrayList<>();
                for (int i = 0; i < tensorType.shape().size(); i++) {
                    reduceShape.add(i == axisValue ? 1 : tensorType.shape().get(i));
                }
                return new TensorType(tensorType.elementType(), reduceShape);
            }
            throw new IllegalArgumentException("tensor must be of type Constant!");
        }

        static CodeType inferConstantType(CodeType codeType) {
            if (codeType instanceof ConstantType constantType) {
                if (constantType.value() instanceof TensorType || constantType.value() instanceof ShapeType) {
                    return (CodeType) constantType.value();
                }
                return codeType;
            }
            return codeType;
        }

        static List<Integer> transposeShape(List<Integer> inputShape) {
            if (inputShape.size() < 2) {
                throw new IllegalArgumentException("tensor shape must be at least of 2D for a transpose operation");
            }
            List<Integer> transposeShape = new ArrayList<>();
            transposeShape.add(inputShape.get(1));
            transposeShape.add(inputShape.get(0));
            for (int i = 2; i < inputShape.size(); i++) {
                transposeShape.add(inputShape.get(i));
            }
            return transposeShape;
        }
    }
}
