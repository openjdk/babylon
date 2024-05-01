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
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.CodeType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;

public class SpirvOps {
    private static final String NAME_PREFIX = "spirv.";

    public static final class ModuleOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "module";

        private final String name;

        public ModuleOp(String moduleName) {
            super(OPNAME);
            this.name = moduleName;
        }

        public ModuleOp(ModuleOp that, CopyContext cc) {
            super(that, cc);
            this.name = that.name;
        }

        @Override
        public ModuleOp transform(CopyContext cc, OpTransformer ot) {
            return new ModuleOp(this, cc);
        }
    }

    public static final class LoadOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "load";

        public LoadOp(CodeType resultType, List<Value> operands) {
            super(OPNAME, resultType, operands);
        }

        public LoadOp(LoadOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LoadOp transform(CopyContext cc, OpTransformer ot) {
            return new LoadOp(this, cc);
        }
    }

    public static final class FieldLoadOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "fieldload";
        private final FieldRef fieldDesc;

        public FieldLoadOp(CodeType resultType, FieldRef fieldRef, List<Value> operands) {
            super(OPNAME, resultType, operands);
            this.fieldDesc = fieldRef;
        }

        public FieldLoadOp(FieldLoadOp that, CopyContext cc) {
            super(that, cc);
            this.fieldDesc = that.fieldDesc;
        }

        @Override
        public FieldLoadOp transform(CopyContext cc, OpTransformer ot) {
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

        public StoreOp(StoreOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public StoreOp transform(CopyContext cc, OpTransformer ot) {
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

        public CallOp(CallOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public CallOp transform(CopyContext cc, OpTransformer ot) {
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

        public ArrayLengthOp(CodeType resultType, List<Value> operands) {
            super(NAME, resultType, operands);
        }

        public ArrayLengthOp(ArrayLengthOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ArrayLengthOp transform(CopyContext cc, OpTransformer ot) {
            return new ArrayLengthOp(this, cc);
        }
    }

    public static final class ConstantOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "constant";
        private final Object value;

        public ConstantOp(CodeType resultType, Object value) {
                super(OPNAME, resultType, List.of());
                this.value = value;
        }

        public ConstantOp(ConstantOp that, CopyContext cc) {
            super(that, cc);
            this.value = that.value;
        }

        @Override
        public ConstantOp transform(CopyContext cc, OpTransformer ot) {
            return new ConstantOp(this, cc);
        }

        public Object value() {
            return value;
        }
    }

    public static final class ConvertOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "sconvert";

        public ConvertOp(CodeType resultType, List<Value> operands) {
                super(OPNAME, resultType, operands);
        }

        public ConvertOp(ConvertOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ConvertOp transform(CopyContext cc, OpTransformer ot) {
            return new ConvertOp(this, cc);
        }
    }

    public static final class IAddOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "iadd";

        public IAddOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IAddOp(IAddOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public IAddOp transform(CopyContext cc, OpTransformer ot) {
            return new IAddOp(this, cc);
        }
    }

    public static final class FAddOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fadd";

        public FAddOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FAddOp(FAddOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public FAddOp transform(CopyContext cc, OpTransformer ot) {
            return new FAddOp(this, cc);
        }
    }

    public static final class ISubOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "isub";

        public ISubOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public ISubOp(ISubOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ISubOp transform(CopyContext cc, OpTransformer ot) {
            return new ISubOp(this, cc);
        }
    }

    public static final class FSubOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fsub";

        public FSubOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FSubOp(FSubOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public FSubOp transform(CopyContext cc, OpTransformer ot) {
            return new FSubOp(this, cc);
        }
    }

    public static final class IMulOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "imul";

        public IMulOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IMulOp(IMulOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public IMulOp transform(CopyContext cc, OpTransformer ot) {
            return new IMulOp(this, cc);
        }
    }

    public static final class FMulOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fmul";

        public FMulOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FMulOp(FMulOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public FMulOp transform(CopyContext cc, OpTransformer ot) {
            return new FMulOp(this, cc);
        }
    }

    public static final class IDivOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "idiv";

        public IDivOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IDivOp(IDivOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public IDivOp transform(CopyContext cc, OpTransformer ot) {
            return new IDivOp(this, cc);
        }
    }

    public static final class FDivOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fdiv";

        public FDivOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FDivOp(FDivOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public FDivOp transform(CopyContext cc, OpTransformer ot) {
            return new FDivOp(this, cc);
        }
    }

    public static final class ModOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "mod";

        public ModOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public ModOp(ModOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ModOp transform(CopyContext cc, OpTransformer ot) {
            return new ModOp(this, cc);
        }
    }

    public static final class IEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "iequal";

        public IEqualOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public IEqualOp(IEqualOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public IEqualOp transform(CopyContext cc, OpTransformer ot) {
            return new IEqualOp(this, cc);
        }
    }

    public static final class FEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fequal";

        public FEqualOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FEqualOp(FEqualOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public FEqualOp transform(CopyContext cc, OpTransformer ot) {
            return new FEqualOp(this, cc);
        }
    }

    public static final class INotEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "inotequal";

        public INotEqualOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public INotEqualOp(INotEqualOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public INotEqualOp transform(CopyContext cc, OpTransformer ot) {
            return new INotEqualOp(this, cc);
        }
    }


    public static final class FNotEqualOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "fnotequal";

        public FNotEqualOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public FNotEqualOp(FNotEqualOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public FNotEqualOp transform(CopyContext cc, OpTransformer ot) {
            return new FNotEqualOp(this, cc);
        }
    }

    public static final class LtOp extends SpirvOp {
        public static final String NAME = NAME_PREFIX + "lt";

        public LtOp(CodeType resultType, List<Value> operands) {
                super(NAME, resultType, operands);
        }

        public LtOp(LtOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LtOp transform(CopyContext cc, OpTransformer ot) {
            return new LtOp(this, cc);
        }
    }

    public static final class BranchOp extends SpirvOp implements Op.BlockTerminating {
        public static final String NAME = NAME_PREFIX + "br";
        private final Block.Reference successor;

        public BranchOp(Block.Reference successor) {
            super(NAME, JavaType.VOID, List.of());
            this.successor = successor;
        }

        public BranchOp(BranchOp that, CopyContext cc) {
            super(that, cc);
            this.successor = that.successor;
        }

        @Override
        public BranchOp transform(CopyContext cc, OpTransformer ot) {
            return new BranchOp(this, cc);
        }

        public Block branch() {
            return successor.targetBlock();
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

        public ConditionalBranchOp(ConditionalBranchOp that, CopyContext cc) {
            super(that, cc);
            this.trueBlock = that.trueBlock;
            this.falseBlock = that.falseBlock;
        }

        @Override
        public ConditionalBranchOp transform(CopyContext cc, OpTransformer ot) {
            return new ConditionalBranchOp(this, cc);
        }

        public Block trueBranch() {
            return trueBlock.targetBlock();
        }

        public Block falseBranch() {
            return falseBlock.targetBlock();
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(trueBlock, falseBlock);
        }
    }

    public static final class VariableOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "variable";
        private final String varName;
        private final CodeType varType;

        public VariableOp(String varName, CodeType type, CodeType varType) {
            super(OPNAME + " @" + varName, type, List.of());
            this.varName = varName;
            this.varType = varType;
        }

        public VariableOp(VariableOp that, CopyContext cc) {
            super(that, cc);
            this.varName = that.varName;
            this.varType = that.varType;
        }

        @Override
        public VariableOp transform(CopyContext cc, OpTransformer ot) {
            return new VariableOp(this, cc);
        }

        public CodeType varType() {
            return varType;
        }

        public String varName() {
            return varName;
        }
    }

    public static final class CompositeExtractOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "compositeExtract";

        public CompositeExtractOp(CodeType resultType, List<Value> operands) {
                super(OPNAME, resultType, operands);
        }

        public CompositeExtractOp(CompositeExtractOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public CompositeExtractOp transform(CopyContext cc, OpTransformer ot) {
            return new CompositeExtractOp(this, cc);
        }
    }

    public static final class InBoundAccessChainOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "inBoundAccessChain";

        public InBoundAccessChainOp(CodeType resultType, List<Value> operands) {
                super(OPNAME, resultType, operands);
        }

        public InBoundAccessChainOp(InBoundAccessChainOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public InBoundAccessChainOp transform(CopyContext cc, OpTransformer ot) {
            return new InBoundAccessChainOp(this, cc);
        }
    }

    public static final class ReturnOp extends SpirvOp implements Op.Terminating {
        public static final String OPNAME = "return";

        public ReturnOp(CodeType resultType, List<Value> operands) {
            super(OPNAME, resultType, operands);
        }

        public ReturnOp(ReturnOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ReturnOp transform(CopyContext cc, OpTransformer ot) {
            return new ReturnOp(this, cc);
        }
    }

    public static final class FunctionParameterOp extends SpirvOp {
        public static final String OPNAME = NAME_PREFIX + "function parameter";

        public FunctionParameterOp(CodeType resultType, List<Value> operands) {
            super(OPNAME, resultType, operands);
        }

        public FunctionParameterOp(FunctionParameterOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public FunctionParameterOp transform(CopyContext cc, OpTransformer ot) {
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

        public FuncOp(FuncOp that, CopyContext cc) {
            super(that, cc);
            this.functionName = that.functionName;
            this.functionType = that.functionType;
            this.body = that.body;
        }

        @Override
        public FuncOp transform(CopyContext cc, OpTransformer ot) {
            return new FuncOp(this, cc);
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