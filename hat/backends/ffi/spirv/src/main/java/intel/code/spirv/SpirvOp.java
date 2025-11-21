/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaType;

public abstract class SpirvOp extends Op {
    static final String NAME_PREFIX = "spirv.";
    private final TypeElement type;
    private final Map<String, Object> attributes;

    SpirvOp(String opName) {
        super(opName, List.of());
        this.type = JavaType.VOID;
        this.attributes = new HashMap<>();
    }

    SpirvOp(String opName, TypeElement type, List<Value> operands) {
        super(opName, operands);
        this.type = type;
        this.attributes = new HashMap<>();
    }

    SpirvOp(String opName, TypeElement type, List<Value> operands, Map<String, Object> attributes) {
        super(opName, operands);
        this.type = type;
        this.attributes = new HashMap<>();
    }

    SpirvOp(SpirvOp that, CodeContext cc) {
        super(that, cc);
        this.type = that.type;
        this.attributes = new HashMap<>();
    }

    @Override
    public TypeElement resultType() {
        return type;
    }

    @Override
    public Map<String, Object> attributes() {
        return attributes;
    }

    public static final class ModuleOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "module";

        private final String name;

        public ModuleOp(String moduleName) {
            super(OPNAME);
            this.name = moduleName;
        }

        public ModuleOp(ModuleOp that, CodeContext cc) {
            super(that, cc);
            this.name = that.name;
        }

        @Override
        public ModuleOp transform(CodeContext cc, CodeTransformer ot) {
            return new ModuleOp(this, cc);
        }
    }

    public static final class PhiOp extends SpirvOp {
        public static record Predecessor(Block.Reference block, Value value) {}

        public static final String OPNAME = NAME_PREFIX + "phi";
        private final List<Predecessor> predecessors;


        public PhiOp(TypeElement resultType, List<Predecessor> predecessors) {
            super(OPNAME, resultType, List.of());
            this.predecessors = predecessors;
            this.attributes().put("predecessors", predecessors);
        }

        public PhiOp(PhiOp that, CodeContext cc) {
            super(that, cc);
            List<Predecessor> thisPredecessors = List.of();
            for (var predecessor : that.predecessors) {
                thisPredecessors.add(new Predecessor(cc.getSuccessorOrCreate(predecessor.block), predecessor.value()));
            }
            this.predecessors = thisPredecessors;
            this.attributes().put("predecessors", predecessors);
        }

        @Override
        public PhiOp transform(CodeContext cc, CodeTransformer ot) {
            return new PhiOp(this, cc);
        }

        public List<Predecessor> predecessors() {
            return predecessors;
        }
    }

    public static final class LoadOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "load";

        public LoadOp(TypeElement resultType, List<Value> operands) {
            super(OPNAME, resultType, operands);
        }

        public LoadOp(LoadOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public LoadOp transform(CodeContext cc, CodeTransformer ot) {
            return new LoadOp(this, cc);
        }
    }

    public static final class FieldLoadOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "fieldload";
        private final FieldRef fieldDesc;

        public FieldLoadOp(TypeElement resultType, FieldRef fieldRef, List<Value> operands) {
            super(OPNAME, resultType, operands);
            this.fieldDesc = fieldRef;
        }

        public FieldLoadOp(FieldLoadOp that, CodeContext cc) {
            super(that, cc);
            this.fieldDesc = that.fieldDesc;
        }

        @Override
        public FieldLoadOp transform(CodeContext cc, CodeTransformer ot) {
            return new FieldLoadOp(this, cc);
        }

        public FieldRef fieldDescriptor() {
            return fieldDesc;
        }
    }

    public static final class StoreOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "store";

        public StoreOp(Value dest, Value value) {
            super(NAME, JavaType.VOID, List.of(dest, value));
        }

        public StoreOp(List<Value> operands) {
            this(operands.get(0), operands.get(1));
        }

        public StoreOp(StoreOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public StoreOp transform(CodeContext cc, CodeTransformer ot) {
            return new StoreOp(this, cc);
        }
    }

    public static final class CallOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "call";
        private MethodRef descriptor;

        public CallOp(MethodRef descriptor, List<Value> operands) {
            super(nameString(descriptor), descriptor.type().returnType(), operands);
            this.descriptor = descriptor;
        }

        public CallOp(CallOp that, CodeContext cc) {
            super(that, cc);
            this.descriptor = that.descriptor;
        }

        @Override
        public CallOp transform(CodeContext cc, CodeTransformer ot) {
            return new CallOp(this, cc);
        }

        public MethodRef callDescriptor() {
            return descriptor;
        }

        private static String nameString(MethodRef descriptor) {
            return OPNAME + " @" + descriptor.refType() + "::" + descriptor.name();
        }
    }

    public static final class ArrayLengthOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "arraylength";

        public ArrayLengthOp(TypeElement resultType, List<Value> operands) {
            super(NAME, resultType, operands);
        }

        public ArrayLengthOp(ArrayLengthOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ArrayLengthOp transform(CodeContext cc, CodeTransformer ot) {
            return new ArrayLengthOp(this, cc);
        }
    }

    public static final class ConstantOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "constant";
        private final Object value;

        public ConstantOp(TypeElement resultType, Object value) {
                super(OPNAME, resultType, List.of());
                this.value = value;
        }

        public ConstantOp(ConstantOp that, CodeContext cc) {
            super(that, cc);
            this.value = that.value;  // TODO: need to copy
        }

        @Override
        public ConstantOp transform(CodeContext cc, CodeTransformer ot) {
            return new ConstantOp(this, cc);
        }

        public Object value() {
            return value;
        }
    }

    public static final class ConvertOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "sconvert";

        public ConvertOp(TypeElement resultType, List<Value> operands) {
                super(OPNAME, resultType, operands);
        }

        public ConvertOp(ConvertOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ConvertOp transform(CodeContext cc, CodeTransformer ot) {
            return new ConvertOp(this, cc);
        }
    }

    public static final class BitwiseAndOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "and";

        public BitwiseAndOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public BitwiseAndOp(BitwiseAndOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public BitwiseAndOp transform(CodeContext cc, CodeTransformer ot) {
            return new BitwiseAndOp(this, cc);
        }
    }


    public static final class IAddOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "iadd";

        public IAddOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IAddOp(IAddOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public IAddOp transform(CodeContext cc, CodeTransformer ot) {
            return new IAddOp(this, cc);
        }
    }

    public static final class FAddOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fadd";

        public FAddOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FAddOp(FAddOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FAddOp transform(CodeContext cc, CodeTransformer ot) {
            return new FAddOp(this, cc);
        }
    }

    public static final class ISubOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "isub";

        public ISubOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public ISubOp(ISubOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ISubOp transform(CodeContext cc, CodeTransformer ot) {
            return new ISubOp(this, cc);
        }
    }

    public static final class FSubOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fsub";

        public FSubOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FSubOp(FSubOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FSubOp transform(CodeContext cc, CodeTransformer ot) {
            return new FSubOp(this, cc);
        }
    }

    public static final class IMulOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "imul";

        public IMulOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IMulOp(IMulOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public IMulOp transform(CodeContext cc, CodeTransformer ot) {
            return new IMulOp(this, cc);
        }
    }

    public static final class FMulOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fmul";

        public FMulOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FMulOp(FMulOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FMulOp transform(CodeContext cc, CodeTransformer ot) {
            return new FMulOp(this, cc);
        }
    }

    public static final class IDivOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "idiv";

        public IDivOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IDivOp(IDivOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public IDivOp transform(CodeContext cc, CodeTransformer ot) {
            return new IDivOp(this, cc);
        }
    }

    public static final class FDivOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fdiv";

        public FDivOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FDivOp(FDivOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FDivOp transform(CodeContext cc, CodeTransformer ot) {
            return new FDivOp(this, cc);
        }
    }

    public static final class ModOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "mod";

        public ModOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public ModOp(ModOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ModOp transform(CodeContext cc, CodeTransformer ot) {
            return new ModOp(this, cc);
        }
    }

    public static final class SNegateOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "sneg";

        public SNegateOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public SNegateOp(SNegateOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public SNegateOp transform(CodeContext cc, CodeTransformer ot) {
            return new SNegateOp(this, cc);
        }
    }

    public static final class FNegateOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fneg";

        public FNegateOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FNegateOp(FNegateOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FNegateOp transform(CodeContext cc, CodeTransformer ot) {
            return new FNegateOp(this, cc);
        }
    }

    public static final class IEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "iequal";

        public IEqualOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IEqualOp(IEqualOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public IEqualOp transform(CodeContext cc, CodeTransformer ot) {
            return new IEqualOp(this, cc);
        }
    }

    public static final class FEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fequal";

        public FEqualOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FEqualOp(FEqualOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FEqualOp transform(CodeContext cc, CodeTransformer ot) {
            return new FEqualOp(this, cc);
        }
    }

    public static final class INotEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "inotequal";

        public INotEqualOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public INotEqualOp(INotEqualOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public INotEqualOp transform(CodeContext cc, CodeTransformer ot) {
            return new INotEqualOp(this, cc);
        }
    }


    public static final class FNotEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fnotequal";

        public FNotEqualOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FNotEqualOp(FNotEqualOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FNotEqualOp transform(CodeContext cc, CodeTransformer ot) {
            return new FNotEqualOp(this, cc);
        }
    }

    public static final class PtrNotEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "ptrnotequal";

        public PtrNotEqualOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public PtrNotEqualOp(PtrNotEqualOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public PtrNotEqualOp transform(CodeContext cc, CodeTransformer ot) {
            return new PtrNotEqualOp(this, cc);
        }
    }

    public static final class LtOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "lt";

        public LtOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public LtOp(LtOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public LtOp transform(CodeContext cc, CodeTransformer ot) {
            return new LtOp(this, cc);
        }
    }

    public static final class GtOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "gt";

        public GtOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public GtOp(GtOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public GtOp transform(CodeContext cc, CodeTransformer ot) {
            return new GtOp(this, cc);
        }
    }

    public static final class GeOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "ge";

        public GeOp(TypeElement resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public GeOp(GeOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public GeOp transform(CodeContext cc, CodeTransformer ot) {
            return new GeOp(this, cc);
        }
    }

    public static final class BranchOp extends SpirvOp implements Op.BlockTerminating {
        public static final String NAME = NAME_PREFIX + "br";
        private final Block.Reference successor;

        public BranchOp(Block.Reference successor) {
            super(NAME, JavaType.VOID, List.of());
            this.successor = successor;
        }

        public BranchOp(BranchOp that, CodeContext cc) {
            super(that, cc);
            this.successor = cc.getSuccessorOrCreate(that.successor);
        }

        @Override
        public BranchOp transform(CodeContext cc, CodeTransformer ot) {
            return new BranchOp(this, cc);
        }

        public Block.Reference branch() {
            return successor;
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(successor);
        }
    }

    public static final class ConditionalBranchOp extends SpirvOp implements Op.BlockTerminating {
        public static final String NAME = NAME_PREFIX + "brcond";
        private final Block.Reference trueBlock;
        private final Block.Reference falseBlock;

        public ConditionalBranchOp(Block.Reference trueBlock, Block.Reference falseBlock, List<Value> operands) {
                super(NAME, JavaType.VOID, operands);
                this.trueBlock = trueBlock;
                this.falseBlock = falseBlock;
        }

        public ConditionalBranchOp(ConditionalBranchOp that, CodeContext cc) {
            super(that, cc);
            this.trueBlock = cc.getSuccessorOrCreate(that.trueBlock);
            this.falseBlock = cc.getSuccessorOrCreate(that.falseBlock);
        }

        @Override
        public ConditionalBranchOp transform(CodeContext cc, CodeTransformer ot) {
            return new ConditionalBranchOp(this, cc);
        }

        public Block.Reference trueBranch() {
            return trueBlock;
        }

        public Block.Reference falseBranch() {
            return falseBlock;
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(trueBlock, falseBlock);
        }
    }

    public static final class VariableOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "variable";
        private final String varName;
        private final TypeElement varType;

        public VariableOp(String varName, TypeElement type, TypeElement varType) {
            super(OPNAME + " @" + varName, type, List.of());
            this.varName = varName;
            this.varType = varType;
        }

        public VariableOp(VariableOp that, CodeContext cc) {
            super(that, cc);
            this.varName = that.varName;
            this.varType = that.varType;
        }

        @Override
        public VariableOp transform(CodeContext cc, CodeTransformer ot) {
            return new VariableOp(this, cc);
        }

        public TypeElement varType() {
            return varType;
        }

        public String varName() {
            return varName;
        }
    }

    public static final class CompositeExtractOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "compositeExtract";

        public CompositeExtractOp(TypeElement resultType, List<Value> operands) {
                super(OPNAME, resultType, operands);
        }

        public CompositeExtractOp(CompositeExtractOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public CompositeExtractOp transform(CodeContext cc, CodeTransformer ot) {
            return new CompositeExtractOp(this, cc);
        }
    }

    public static final class InBoundsAccessChainOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "inBoundsAccessChain";

        public InBoundsAccessChainOp(TypeElement resultType, List<Value> operands) {
                super(OPNAME, resultType, operands);
        }

        public InBoundsAccessChainOp(InBoundsAccessChainOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public InBoundsAccessChainOp transform(CodeContext cc, CodeTransformer ot) {
            return new InBoundsAccessChainOp(this, cc);
        }
    }

    public static final class ReturnOp extends SpirvOp implements Op.Terminating {
        public static final String OPNAME = "return";

        public ReturnOp(TypeElement resultType) {
            super(OPNAME, resultType, List.of());
        }

        public ReturnOp(ReturnOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ReturnOp transform(CodeContext cc, CodeTransformer ot) {
            return new ReturnOp(this, cc);
        }
    }

    public static final class ReturnValueOp extends SpirvOp implements Op.Terminating {
        public static final String OPNAME = "return";

        public ReturnValueOp(TypeElement resultType, List<Value> operands) {
            super(OPNAME, resultType, operands);
        }

        public ReturnValueOp(ReturnValueOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ReturnValueOp transform(CodeContext cc, CodeTransformer ot) {
            return new ReturnValueOp(this, cc);
        }
    }

    public static final class FunctionParameterOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "function parameter";

        public FunctionParameterOp(TypeElement resultType, List<Value> operands) {
            super(OPNAME, resultType, operands);
        }

        public FunctionParameterOp(FunctionParameterOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public FunctionParameterOp transform(CodeContext cc, CodeTransformer ot) {
            return new FunctionParameterOp(this, cc);
        }
    }

    public static final class FuncOp extends SpirvOp implements Op.Invokable {
        public static enum Control {
            INLINE,
            DONTINLINE,
            PURE,
            CONST,
            NONE
        }

        public static final String OPNAME = NAME_PREFIX + "function";
        private final String functionName;
        private final FunctionType functionType;
        private final Body body;


        public FuncOp(String name, FunctionType functionType, Body.Builder builder) {
            super(OPNAME + "_" + name);
            this.functionName = name;
            this.functionType = functionType;
            this.body = builder.build(this);
        }

        public FuncOp(FuncOp that, CodeContext cc) {
            super(that, cc);
            this.functionName = that.functionName;
            this.functionType = that.functionType;
            this.body = that.body;
        }

        public FuncOp transform(CodeTransformer ot) {
            return transform(CodeContext.create(), ot);
        }

        @Override
        public FuncOp transform(CodeContext cc, CodeTransformer ot) {
            return new FuncOp(this, this.functionName, cc, ot);
        }

        FuncOp(FuncOp that, String functionName, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.functionType = that.functionType;
            this.functionName = functionName;
            Body.Builder bb = that.body.transform(cc, ot);
            this.body = bb.build(this);
        }

        @Override
        public Body body() {
            return body;
        }

        public String functionName() {
            return functionName;
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        @Override
        public FunctionType invokableType() {
            return functionType;
        }
    }
}