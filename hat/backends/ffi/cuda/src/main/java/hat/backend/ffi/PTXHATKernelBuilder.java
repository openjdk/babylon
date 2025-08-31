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
package hat.backend.ffi;

import hat.ifacemapper.BoundSchema;
import hat.optools.*;
import hat.codebuilders.CodeBuilder;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.foreign.MemoryLayout;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class PTXHATKernelBuilder extends CodeBuilder<PTXHATKernelBuilder> {

    Map<Value, PTXRegister> varToRegMap;
    List<String> paramNames;
    List<Block.Parameter> paramObjects;
    Map<Field, PTXRegister> fieldToRegMap;

    HashMap<PTXRegister.Type, Integer> ordinalMap;

    PTXRegister returnReg;
    private int addressSize;

    public enum Field {
        NTID_X ("ntid.x", false),
        CTAID_X ("ctaid.x", false),
        TID_X ("tid.x", false),
        KC_X ("x", false),
        KC_ADDR("kc", true),
        KC_MAXX ("maxX", false);

        private final String name;
        private final boolean destination;

        Field(String name, boolean destination) {
            this.name = name;
            this.destination = destination;
        }
        public String toString() {
            return this.name;
        }
        public boolean isDestination() {return this.destination;}
    }

    public PTXHATKernelBuilder(int addressSize) {
        varToRegMap = new HashMap<>();
        paramNames = new ArrayList<>();
        fieldToRegMap = new HashMap<>();
        paramObjects = new ArrayList<>();
        ordinalMap = new HashMap<>();
        this.addressSize = addressSize;
    }

    public PTXHATKernelBuilder() {
        this(32);
    }

    public void ptxHeader(int major, int minor, String target, int addressSize) {
        this.addressSize = addressSize;
        version().space().major(major).dot().minor(minor).nl();
        target().space().target(target).nl();
        addressSize().space().size(addressSize);
    }

    public void functionHeader(String funcName, boolean entry, TypeElement yieldType) {
        if (entry) {
            visible().space().entry().space();
        } else {
            func().space();
        }
        if (!yieldType.toString().equals("void")) {
            returnReg = new PTXRegister(getOrdinal(getResultType(yieldType)), getResultType(yieldType));
            returnReg.name("%retReg");
            oparen().dot().param().space().paramType(yieldType);
            space().regName(returnReg).cparen().space();
        }
        funcName(funcName);
    }

    public PTXHATKernelBuilder parameters(List<FuncOpWrapper.ParamTable.Info> infoList) {
        paren(_ -> nl().commaNlSeparated(infoList, (info) -> {
            ptxIndent().dot().param().space().paramType(info.javaType);
            space().regName(info.varOp.varName());
            paramNames.add(info.varOp.varName());
        }).nl()).nl();
        return this;
    }

    public void blockBody(Block block, Stream<OpWrapper<?>> ops) {
        if (block.index() == 0) {
            for (Block.Parameter p : block.parameters()) {
                ptxIndent().ld().dot().param();
                resultType(p.type(), false).ptxIndent().space();
                reg(p, getResultType(p.type())).commaSpace().osbrace().regName(paramNames.get(p.index())).csbrace().semicolon().nl();
                paramObjects.add(p);
            }
        }
        nl();
        block(block);
        colon().nl();
        ops.forEach(op -> {
            if (op instanceof InvokeOpWrapper invoke && !invoke.isIfaceBufferMethod()) {
                ptxIndent().convert(op).nl();
            } else {
                ptxIndent().convert(op).semicolon().nl();
            }
        });
    }

    public void ptxRegisterDecl() {
        for (PTXRegister.Type t : ordinalMap.keySet()) {
            ptxIndent().reg().space();
            if (t.equals(PTXRegister.Type.U32)) {
                b32();
            } else if (t.equals(PTXRegister.Type.U64)) {
                b64();
            } else {
                dot().regType(t);
            }
            ptxIndent().regTypePrefix(t).oabrace().intVal(ordinalMap.get(t)).cabrace().semicolon().nl();
        }
        nl();
    }

    public void functionPrologue() {
        obrace().nl();
    }

    public void functionEpilogue() {
        cbrace();
    }

    public static class PTXPtrOp extends Op {
        public String fieldName;
        public static final String NAME = "ptxPtr";
        final TypeElement resultType;
        public BoundSchema<?> boundSchema;

        PTXPtrOp(TypeElement resultType, String fieldName, List<Value> operands, BoundSchema<?> boundSchema) {
            super(NAME, operands);
            this.resultType = resultType;
            this.fieldName = fieldName;
            this.boundSchema = boundSchema;
        }

        PTXPtrOp(PTXPtrOp that, CopyContext cc) {
            super(that, cc);
            this.resultType = that.resultType;
            this.fieldName = that.fieldName;
            this.boundSchema = that.boundSchema;
        }

        @Override
        public PTXPtrOp transform(CopyContext cc, OpTransformer ot) {
            return new PTXPtrOp(this, cc);
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }


    public PTXHATKernelBuilder convert(OpWrapper<?> wrappedOp) {
        switch (wrappedOp) {
            case FieldLoadOpWrapper op -> fieldLoad(op);
            case FieldStoreOpWrapper op -> fieldStore(op);
            case BinaryArithmeticOrLogicOperation op -> binaryOperation(op);
            case BinaryTestOpWrapper op -> binaryTest(op);
            case ConvOpWrapper op -> conv(op);
            case ConstantOpWrapper op -> constant(op);
            case YieldOpWrapper op -> javaYield(op);
            case InvokeOpWrapper op -> methodCall(op);
            case VarDeclarationOpWrapper op -> varDeclaration(op);
            case VarFuncDeclarationOpWrapper op -> varFuncDeclaration(op);
            case ReturnOpWrapper op -> ret(op);
            case JavaBreakOpWrapper op -> javaBreak(op);
            default -> {
                switch (wrappedOp.op){
                    case CoreOp.BranchOp op -> branch(op);
                    case CoreOp.ConditionalBranchOp op -> condBranch(op);
                    case JavaOp.NegOp op -> neg(op);
                    case PTXPtrOp op -> ptxPtr(op);
                    default -> throw new IllegalStateException("op translation doesn't exist");
                }
            }
        }
        return this;
    }

    public void ptxPtr(PTXPtrOp op) {
        PTXRegister source;
        int offset = (int) op.boundSchema.groupLayout().byteOffset(MemoryLayout.PathElement.groupElement(op.fieldName));

        if (op.fieldName.equals("array")) {
            source = new PTXRegister(incrOrdinal(addressType()), addressType());
            add().s64().space().regName(source).commaSpace().reg(op.operands().get(0)).commaSpace().reg(op.operands().get(1)).ptxNl();
        } else {
            source = getReg(op.operands().getFirst());
        }

        if (op.resultType.toString().equals("void")) {
            st().global().dot().regType(op.operands().getLast()).space().address(source.name(), offset).commaSpace().reg(op.operands().getLast());
        } else {
            ld().global().resultType(op.resultType(), true).space().reg(op.result(), getResultType(op.resultType())).commaSpace().address(source.name(), offset);
        }
    }

    public void fieldLoad(FieldLoadOpWrapper fieldLoadOpWrapper) {
        if (FieldAccessOpWrapper.fieldName(fieldLoadOpWrapper.op).equals(Field.KC_X.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(fieldLoadOpWrapper.op.result());
            } else {
                mov().u32().space().resultReg(fieldLoadOpWrapper, PTXRegister.Type.U32).commaSpace().fieldReg(Field.KC_X);
            }
        } else if (FieldAccessOpWrapper.fieldName(fieldLoadOpWrapper.op).equals(Field.KC_MAXX.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(fieldLoadOpWrapper.op.operands().getFirst());
            }
            ld().global().u32().space().fieldReg(Field.KC_MAXX, fieldLoadOpWrapper.op.result()).commaSpace()
                    .address(fieldToRegMap.get(Field.KC_ADDR).name(), 4);
        } else {
            ld().global().u32().space().resultReg(fieldLoadOpWrapper, PTXRegister.Type.U64).commaSpace().reg(fieldLoadOpWrapper.op.operands().getFirst());
        }
    }

    public void loadKcX(Value value) {
        cvta().to().global().size().space().fieldReg(Field.KC_ADDR).commaSpace()
                .reg(paramObjects.get(paramNames.indexOf(Field.KC_ADDR.toString())), addressType()).ptxNl();
        mov().u32().space().fieldReg(Field.NTID_X).commaSpace().percent().regName(Field.NTID_X.toString()).ptxNl();
        mov().u32().space().fieldReg(Field.CTAID_X).commaSpace().percent().regName(Field.CTAID_X.toString()).ptxNl();
        mov().u32().space().fieldReg(Field.TID_X).commaSpace().percent().regName(Field.TID_X.toString()).ptxNl();
        mad().lo().s32().space().fieldReg(Field.KC_X, value).commaSpace().fieldReg(Field.CTAID_X)
                .commaSpace().fieldReg(Field.NTID_X).commaSpace().fieldReg(Field.TID_X).ptxNl();
        st().global().u32().space().address(fieldToRegMap.get(Field.KC_ADDR).name()).commaSpace().fieldReg(Field.KC_X);
    }

    public void fieldStore(FieldStoreOpWrapper op) {
        // TODO: fix
        st().global().u64().space().resultReg(op, PTXRegister.Type.U64).commaSpace().reg(op.op.operands().getFirst());
    }

    PTXHATKernelBuilder symbol(Op op) {
        return switch (op) {
            case JavaOp.ModOp _ -> rem();
            case JavaOp.MulOp _ -> mul();
            case JavaOp.DivOp _ -> div();
            case JavaOp.AddOp _ -> add();
            case JavaOp.SubOp _ -> sub();
            case JavaOp.LtOp _ -> lt();
            case JavaOp.GtOp _ -> gt();
            case JavaOp.LeOp _ -> le();
            case JavaOp.GeOp _ -> ge();
            case JavaOp.NeqOp _ -> ne();
            case JavaOp.EqOp _ -> eq();
            case JavaOp.OrOp _ -> or();
            case JavaOp.AndOp _ -> and();
            case JavaOp.XorOp _ -> xor();
            case JavaOp.LshlOp _ -> shl();
            case JavaOp.AshrOp _, JavaOp.LshrOp _ -> shr();
            default -> throw new IllegalStateException("Unexpected value");
        };
    }

    public void binaryOperation(BinaryArithmeticOrLogicOperation op) {
        symbol(op.op);
        if (getResultType(op.op.resultType()).getBasicType().equals(PTXRegister.Type.BasicType.FLOATING)
                && (op.op instanceof JavaOp.DivOp || op.op instanceof JavaOp.MulOp)) {
            rn();
        } else if (!getResultType(op.op.resultType()).getBasicType().equals(PTXRegister.Type.BasicType.FLOATING)
                && op.op instanceof JavaOp.MulOp) {
            lo();
        }
        resultType(op.op.resultType(), true).space();
        resultReg(op, getResultType(op.op.resultType()));
        commaSpace();
        reg(op.op.operands().getFirst());
        commaSpace();
        reg(op.op.operands().get(1));
    }

    public void binaryTest(BinaryTestOpWrapper op) {
        setp().dot();
        symbol(op.op).resultType(op.op.operands().getFirst().type(), true).space();
        resultReg(op, PTXRegister.Type.PREDICATE);
        commaSpace();
        reg(op.op.operands().getFirst());
        commaSpace();
        reg(op.op.operands().get(1));
    }

    public void conv(ConvOpWrapper op) {
        if (op.op.resultType().equals(JavaType.LONG)) {
            if (isIndex(op)) {
                mul().wide().s32().space().resultReg(op, PTXRegister.Type.U64).commaSpace()
                        .reg(op.op.operands().getFirst()).commaSpace().intVal(4);
            } else {
                cvt().u64().dot().regType(op.op.operands().getFirst()).space()
                        .resultReg(op, PTXRegister.Type.U64).commaSpace().reg(op.op.operands().getFirst()).ptxNl();
            }
        } else if (op.op.resultType().equals(JavaType.FLOAT)) {
            cvt().rn().f32().dot().regType(op.op.operands().getFirst()).space()
                    .resultReg(op, PTXRegister.Type.F32).commaSpace().reg(op.op.operands().getFirst());
        } else if (op.op.resultType().equals(JavaType.DOUBLE)) {
            cvt();
            if (op.op.operands().getFirst().type().equals(JavaType.INT)) {
                rn();
            }
            f64().dot().regType(op.op.operands().getFirst()).space()
                    .resultReg(op, PTXRegister.Type.F64).commaSpace().reg(op.op.operands().getFirst());
        } else if (op.op.resultType().equals(JavaType.INT)) {
            cvt();
            if (op.op.operands().getFirst().type().equals(JavaType.DOUBLE) || op.op.operands().getFirst().type().equals(JavaType.FLOAT)) {
                rzi();
            } else {
                rn();
            }
            s32().dot().regType(op.op.operands().getFirst()).space()
                    .resultReg(op, PTXRegister.Type.S32).commaSpace().reg(op.op.operands().getFirst());
        } else {
            cvt().rn().s32().dot().regType(op.op.operands().getFirst()).space()
                    .resultReg(op, PTXRegister.Type.S32).commaSpace().reg(op.op.operands().getFirst());
        }
    }





    public static class PTXRegister {
        private String name;
        private final Type type;

        public enum Type {
            S8 (8, BasicType.SIGNED, "s8", "%s"),
            S16 (16, BasicType.SIGNED, "s16", "%s"),
            S32 (32, BasicType.SIGNED, "s32", "%s"),
            S64 (64, BasicType.SIGNED, "s64", "%sd"),
            U8 (8, BasicType.UNSIGNED, "u8", "%r"),
            U16 (16, BasicType.UNSIGNED, "u16", "%r"),
            U32 (32, BasicType.UNSIGNED, "u32", "%r"),
            U64 (64, BasicType.UNSIGNED, "u64", "%rd"),
            F16 (16, BasicType.FLOATING, "f16", "%f"),
            F16X2 (16, BasicType.FLOATING, "f16", "%f"),
            F32 (32, BasicType.FLOATING, "f32", "%f"),
            F64 (64, BasicType.FLOATING, "f64", "%fd"),
            B8 (8, BasicType.BIT, "b8", "%b"),
            B16 (16, BasicType.BIT, "b16", "%b"),
            B32 (32, BasicType.BIT, "b32", "%b"),
            B64 (64, BasicType.BIT, "b64", "%bd"),
            B128 (128, BasicType.BIT, "b128", "%b"),
            PREDICATE (1, BasicType.PREDICATE, "pred", "%p");

            public enum BasicType {
                SIGNED,
                UNSIGNED,
                FLOATING,
                BIT,
                PREDICATE
            }

            private final int size;
            private final BasicType basicType;
            private final String name;
            private final String regPrefix;

            Type(int size, BasicType type, String name, String regPrefix) {
                this.size = size;
                this.basicType = type;
                this.name = name;
                this.regPrefix = regPrefix;
            }

            public int getSize() {
                return this.size;
            }

            public BasicType getBasicType() {
                return this.basicType;
            }

            public String getName() {
                return this.name;
            }

            public String getRegPrefix() {
                return this.regPrefix;
            }
        }

        public PTXRegister(int num, Type type) {
            this.type = type;
            this.name = type.regPrefix + num;
        }

        public String name() {
            return this.name;
        }

        public void name(String name) {
            this.name = name;
        }

        public Type type() {
            return this.type;
        }
    }


    private boolean isIndex(ConvOpWrapper op) {
        for (Op.Result r : op.op.result().uses()) {
            if (r.op() instanceof PTXPtrOp) return true;
        }
        return false;
    }

    public void constant(ConstantOpWrapper op) {
        mov().resultType(op.op.resultType(), false).space().resultReg(op, getResultType(op.op.resultType())).commaSpace();
        if (op.op.resultType().toString().equals("float")) {
            if (op.op.value().toString().equals("0.0")) {
                floatVal("00000000");
            } else {
                floatVal(Integer.toHexString(Float.floatToIntBits(Float.parseFloat(op.op.value().toString()))).toUpperCase());
            }
        } else {
            append(op.op.value().toString());
        }
    }

    public void javaYield(YieldOpWrapper op) {
        exit();
    }

    // S32Array and S32Array2D functions can be deleted after schema is done
    public void methodCall(InvokeOpWrapper op) {
        switch (op.methodRef().toString()) {
            // S32Array functions
            case "hat.buffer.S32Array::array(long)int" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                add().s64().space().regName(temp).commaSpace().reg(op.op.operands().getFirst()).commaSpace().reg(op.op.operands().get(1)).ptxNl();
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(temp.name(), 4);
            }
            case "hat.buffer.S32Array::array(long, int)void" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                add().s64().space().regName(temp).commaSpace().reg(op.op.operands().getFirst()).commaSpace().reg(op.op.operands().get(1)).ptxNl();
                st().global().u32().space().address(temp.name(), 4).commaSpace().reg(op.op.operands().get(2));
            }
            case "hat.buffer.S32Array::length()int" -> {
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(getReg(op.op.operands().getFirst()).name());
            }
            // S32Array2D functions
            case "hat.buffer.S32Array2D::array(long, int)void" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                add().s64().space().regName(temp).commaSpace().reg(op.op.operands().getFirst()).commaSpace().reg(op.op.operands().get(1)).ptxNl();
                st().global().u32().space().address(temp.name(), 8).commaSpace().reg(op.op.operands().get(2));
            }
            case "hat.buffer.S32Array2D::width()int" -> {
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(getReg(op.op.operands().getFirst()).name());
            }
            case "hat.buffer.S32Array2D::height()int" -> {
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(getReg(op.op.operands().getFirst()).name(), 4);
            }
            // Java Math function
            case "java.lang.Math::sqrt(double)double" -> {
                sqrt().rn().f64().space().resultReg(op, PTXRegister.Type.F64).commaSpace().reg(op.op.operands().getFirst()).semicolon();
            }
            default -> {
                obrace().nl().ptxIndent();
                for (int i = 0; i < op.op.operands().size(); i++) {
                    dot().param().space().paramType(op.op.operands().get(i).type()).space().param().intVal(i).ptxNl();
                    st().dot().param().paramType(op.op.operands().get(i).type()).space().osbrace().param().intVal(i).csbrace().commaSpace().reg(op.op.operands().get(i)).ptxNl();
                }
                dot().param().space().paramType(op.op.resultType()).space().retVal().ptxNl();
                call().uni().space().oparen().retVal().cparen().commaSpace().append(op.method().getName()).commaSpace();
                final int[] counter = {0};
                paren(_ -> commaSeparated(op.op.operands(), _ -> param().intVal(counter[0]++))).ptxNl();
                ld().dot().param().paramType(op.op.resultType()).space().resultReg(op, getResultType(op.op.resultType())).commaSpace().osbrace().retVal().csbrace();
                ptxNl().cbrace();
            }
        }
    }

    public void varDeclaration(VarDeclarationOpWrapper op) {
        ld().dot().param().resultType(op.op.resultType(), false).space().resultReg(op, addressType()).commaSpace().reg(op.op.operands().getFirst());
    }

    public void varFuncDeclaration(VarFuncDeclarationOpWrapper op) {
        ld().dot().param().resultType(op.op.resultType(), false).space().resultReg(op, addressType()).commaSpace().reg(op.op.operands().getFirst());
    }

    public void ret(ReturnOpWrapper op) {
        if (!op.op.operands().isEmpty()) {
            st().dot().param();
            if (returnReg.type().equals(PTXRegister.Type.U32)) {
                b32();
            } else if (returnReg.type().equals(PTXRegister.Type.U64)) {
                b64();
            } else {
                dot().regType(returnReg.type());
            }
            space().osbrace().regName(returnReg).csbrace().commaSpace().reg(op.op.operands().getFirst()).ptxNl();
        }
        ret();
    }

    public void javaBreak(JavaBreakOpWrapper op) {
        brkpt();
    }

    public void branch(CoreOp.BranchOp op) {
        loadBlockParams(op.successors().getFirst());
        bra().space().block(op.successors().getFirst().targetBlock());
    }

    public void condBranch(CoreOp.ConditionalBranchOp op) {
        loadBlockParams(op.successors().getFirst());
        loadBlockParams(op.successors().getLast());
        at().reg(op.operands().getFirst()).space()
                .bra().space().block(op.successors().getFirst().targetBlock()).ptxNl();
        bra().space().block(op.successors().getLast().targetBlock());
    }

    public void neg(JavaOp.NegOp op) {
        neg().resultType(op.resultType(), true).space().reg(op.result(), getResultType(op.resultType())).commaSpace().reg(op.operands().getFirst());
    }

    /*
     * Helper functions for printing blocks and variables
     */

    public void loadBlockParams(Block.Reference block) {
        for (int i = 0; i < block.arguments().size(); i++) {
            Block.Parameter p = block.targetBlock().parameters().get(i);
            mov().resultType(p.type(), false).space().reg(p, getResultType(p.type()))
                    .commaSpace().reg(block.arguments().get(i)).ptxNl();
        }
    }

    public PTXHATKernelBuilder block(Block block) {
        return append("block_").intVal(block.index());
    }

    public PTXHATKernelBuilder fieldReg(Field ref) {
        if (fieldToRegMap.containsKey(ref)) {
            return regName(fieldToRegMap.get(ref));
        }
        if (ref.isDestination()) {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(addressType()), addressType()));
        } else {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32));
        }
        return regName(fieldToRegMap.get(ref));
    }

    public PTXHATKernelBuilder fieldReg(Field ref, Value value) {
        if (fieldToRegMap.containsKey(ref)) {
            return regName(fieldToRegMap.get(ref));
        }
        if (ref.isDestination()) {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(getOrdinal(addressType()), addressType()));
            return reg(value, addressType());
        } else {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(getOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32));
            return reg(value, PTXRegister.Type.U32);
        }
    }

    public Field getFieldObj(String fieldName) {
        for (Field f : fieldToRegMap.keySet()) {
            if (f.toString().equals(fieldName)) return f;
        }
        throw new IllegalStateException("no existing field");
    }

    public PTXHATKernelBuilder resultReg(OpWrapper<?> opWrapper, PTXRegister.Type type) {
        return append(addReg(opWrapper.op.result(), type));
    }

    public PTXHATKernelBuilder reg(Value val, PTXRegister.Type type) {
        if (varToRegMap.containsKey(val)) {
            return regName(getReg(val));
        } else {
            return append(addReg(val, type));
        }
    }

    public PTXHATKernelBuilder reg(Value val) {
        return regName(getReg(val));
    }

    public PTXRegister getReg(Value val) {
        if (varToRegMap.get(val) == null && val instanceof Op.Result result && result.op() instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
            return fieldToRegMap.get(getFieldObj(fieldLoadOp.fieldDescriptor().name()));
        }
        if (varToRegMap.containsKey(val)) {
            return varToRegMap.get(val);
        } else {
            throw new IllegalStateException("var to reg mapping doesn't exist");
        }
    }

    public String addReg(Value val, PTXRegister.Type type) {
        if (varToRegMap.containsKey(val)) {
            return varToRegMap.get(val).name();
        }
        varToRegMap.put(val, new PTXRegister(incrOrdinal(type), type));
        return varToRegMap.get(val).name();
    }

    public Integer getOrdinal(PTXRegister.Type type) {
        ordinalMap.putIfAbsent(type, 1);
        return ordinalMap.get(type);
    }

    public Integer incrOrdinal(PTXRegister.Type type) {
        ordinalMap.putIfAbsent(type, 1);
        int out = ordinalMap.get(type);
        ordinalMap.put(type, out + 1);
        return out;
    }

    public PTXHATKernelBuilder size() {
        return (addressSize == 32) ? u32() : u64();
    }

    public PTXRegister.Type addressType() {
        return (addressSize == 32) ? PTXRegister.Type.U32 : PTXRegister.Type.U64;
    }

    public PTXHATKernelBuilder resultType(TypeElement type, boolean signedResult) {
        PTXRegister.Type res = getResultType(type);
        if (signedResult && (res == PTXRegister.Type.U32)) return s32();
        return dot().append(getResultType(type).getName());
    }

    public PTXHATKernelBuilder paramType(TypeElement type) {
        PTXRegister.Type res = getResultType(type);
        if (res == PTXRegister.Type.U32) return b32();
        if (res == PTXRegister.Type.U64) return b64();
        return dot().append(getResultType(type).getName());
    }

    public PTXRegister.Type getResultType(TypeElement type) {
        switch (type.toString()) {
            case "float" -> {
                return PTXRegister.Type.F32;
            }
            case "double" -> {
                return PTXRegister.Type.F64;
            }
            case "int" -> {
                return PTXRegister.Type.U32;
            }
            case "boolean" -> {
                return PTXRegister.Type.PREDICATE;
            }
            default -> {
                return PTXRegister.Type.U64;
            }
        }
    }

    /*
     * Basic CodeBuilder functions
     */

    // used for parameter list
    // prints out items separated by a comma then new line
    // Don't know why this was overriding with the same code grf.
   /* @Override
    public <I> PTXHATKernelBuilder commaNlSeparated(Iterable<I> iterable, Consumer<I> c) {
        StreamCounter.of(iterable, (counter, t) -> {
            if (counter.isNotFirst()) {
                comma().nl();
            }
            c.accept(t);
        });
        return self();
    }
*/
    public PTXHATKernelBuilder address(String address) {
        return osbrace().append(address).csbrace();
    }

    public PTXHATKernelBuilder address(String address, int offset) {
        osbrace().append(address);
        if (offset == 0) {
            return csbrace();
        } else if (offset > 0) {
            plus();
        }
        return intVal(offset).csbrace();
    }

    public PTXHATKernelBuilder ptxNl() {
        return semicolon().nl().ptxIndent();
    }

    public PTXHATKernelBuilder commaSpace() {
        return comma().space();
    }

    public PTXHATKernelBuilder param() {
        return append("param");
    }

    public PTXHATKernelBuilder global() {
        return dot().append("global");
    }

    public PTXHATKernelBuilder rn() {
        return dot().append("rn");
    }

    public PTXHATKernelBuilder rm() {
        return dot().append("rm");
    }

    public PTXHATKernelBuilder rzi() {
        return dot().append("rzi");
    }

    public PTXHATKernelBuilder to() {
        return dot().append("to");
    }

    public PTXHATKernelBuilder lo() {
        return dot().append("lo");
    }

    public PTXHATKernelBuilder wide() {
        return dot().append("wide");
    }

    public PTXHATKernelBuilder uni() {
        return dot().append("uni");
    }

    public PTXHATKernelBuilder sat() {
        return dot().append("sat");
    }

    public PTXHATKernelBuilder ftz() {
        return dot().append("ftz");
    }

    public PTXHATKernelBuilder approx() {
        return dot().append("approx");
    }

    public PTXHATKernelBuilder mov() {
        return append("mov");
    }

    public PTXHATKernelBuilder setp() {
        return append("setp");
    }

    public PTXHATKernelBuilder selp() {
        return append("selp");
    }

    public PTXHATKernelBuilder ld() {
        return append("ld");
    }

    public PTXHATKernelBuilder st() {
        return append("st");
    }

    public PTXHATKernelBuilder cvt() {
        return append("cvt");
    }

    public PTXHATKernelBuilder bra() {
        return append("bra");
    }

    public PTXHATKernelBuilder ret() {
        return append("ret");
    }

    public PTXHATKernelBuilder rem() {
        return append("rem");
    }

    public PTXHATKernelBuilder mul() {
        return append("mul");
    }

    public PTXHATKernelBuilder div() {
        return append("div");
    }

    public PTXHATKernelBuilder rcp() {
        return append("rcp");
    }

    public PTXHATKernelBuilder add() {
        return append("add");
    }

    public PTXHATKernelBuilder sub() {
        return append("sub");
    }

    public PTXHATKernelBuilder lt() {
        return append("lt");
    }

    public PTXHATKernelBuilder gt() {
        return append("gt");
    }

    public PTXHATKernelBuilder le() {
        return append("le");
    }

    public PTXHATKernelBuilder ge() {
        return append("ge");
    }

    public PTXHATKernelBuilder geu() {
        return append("geu");
    }

    public PTXHATKernelBuilder ne() {
        return append("ne");
    }

    public PTXHATKernelBuilder eq() {
        return append("eq");
    }

    public PTXHATKernelBuilder xor() {
        return append("xor");
    }

    public PTXHATKernelBuilder or() {
        return append("or");
    }

    public PTXHATKernelBuilder and() {
        return append("and");
    }

    public PTXHATKernelBuilder cvta() {
        return append("cvta");
    }

    public PTXHATKernelBuilder mad() {
        return append("mad");
    }

    public PTXHATKernelBuilder fma() {
        return append("fma");
    }

    public PTXHATKernelBuilder sqrt() {
        return append("sqrt");
    }

    public PTXHATKernelBuilder abs() {
        return append("abs");
    }

    public PTXHATKernelBuilder ex2() {
        return append("ex2");
    }

    public PTXHATKernelBuilder shl() {
        return append("shl");
    }

    public PTXHATKernelBuilder shr() {
        return append("shr");
    }

    public PTXHATKernelBuilder neg() {
        return append("neg");
    }

    public PTXHATKernelBuilder call() {
        return append("call");
    }

    public PTXHATKernelBuilder exit() {
        return append("exit");
    }

    public PTXHATKernelBuilder brkpt() {
        return append("brkpt");
    }

    public PTXHATKernelBuilder ptxIndent() {
        return append("    ");
    }

    public PTXHATKernelBuilder u32() {
        return dot().append(PTXRegister.Type.U32.getName());
    }

    public PTXHATKernelBuilder s32() {
        return dot().append(PTXRegister.Type.S32.getName());
    }

    public PTXHATKernelBuilder f32() {
        return dot().append(PTXRegister.Type.F32.getName());
    }

    public PTXHATKernelBuilder b32() {
        return dot().append(PTXRegister.Type.B32.getName());
    }

    public PTXHATKernelBuilder u64() {
        return dot().append(PTXRegister.Type.U64.getName());
    }

    public PTXHATKernelBuilder s64() {
        return dot().append(PTXRegister.Type.S64.getName());
    }

    public PTXHATKernelBuilder f64() {
        return dot().append(PTXRegister.Type.F64.getName());
    }

    public PTXHATKernelBuilder b64() {
        return dot().append(PTXRegister.Type.B64.getName());
    }

    public PTXHATKernelBuilder version() {
        return dot().append("version");
    }

    public PTXHATKernelBuilder target() {
        return dot().append("target");
    }

    public PTXHATKernelBuilder addressSize() {
        return dot().append("address_size");
    }

    public PTXHATKernelBuilder major(int major) {
        return intVal(major);
    }

    public PTXHATKernelBuilder minor(int minor) {
        return intVal(minor);
    }

    public PTXHATKernelBuilder target(String target) {
        return append(target);
    }

    public PTXHATKernelBuilder size(int addressSize) {
        return intVal(addressSize);
    }

    public PTXHATKernelBuilder funcName(String funcName) {
        return append(funcName);
    }

    public PTXHATKernelBuilder visible() {
        return dot().append("visible");
    }

    public PTXHATKernelBuilder entry() {
        return dot().append("entry");
    }

    public PTXHATKernelBuilder func() {
        return dot().append("func");
    }

    public PTXHATKernelBuilder oabrace() {
        return append("<");
    }

    public PTXHATKernelBuilder cabrace() {
        return append(">");
    }

    public PTXHATKernelBuilder regName(PTXRegister reg) {
        return append(reg.name());
    }

    public PTXHATKernelBuilder regName(String regName) {
        return append(regName);
    }

    public PTXHATKernelBuilder regType(Value val) {
        return append(getReg(val).type().getName());
    }

    public PTXHATKernelBuilder regType(PTXRegister.Type t) {
        return append(t.getName());
    }

    public PTXHATKernelBuilder regTypePrefix(PTXRegister.Type t) {
        return append(t.getRegPrefix());
    }

    public PTXHATKernelBuilder reg() {
        return dot().append("reg");
    }

    public PTXHATKernelBuilder retVal() {
        return append("retval");
    }

    public PTXHATKernelBuilder temp() {
        return append("temp");
    }

    public PTXHATKernelBuilder intVal(int i) {
        return append(String.valueOf(i));
    }

    public PTXHATKernelBuilder floatVal(String s) {
        return append("0f").append(s);
    }

    public PTXHATKernelBuilder doubleVal(String s) {
        return append("0d").append(s);
    }
}