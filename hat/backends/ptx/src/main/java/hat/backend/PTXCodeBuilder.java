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
package hat.backend;

import hat.ifacemapper.Schema;
import hat.optools.*;
import hat.text.CodeBuilder;
import hat.util.StreamCounter;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class PTXCodeBuilder extends CodeBuilder<PTXCodeBuilder> {

    Map<Value, PTXRegister> varToRegMap;
    List<String> params;
    Map<String, Block.Parameter> paramMap;
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

    public PTXCodeBuilder(int addressSize) {
        varToRegMap = new HashMap<>();
        params = new ArrayList<>();
        fieldToRegMap = new HashMap<>();
        paramMap = new HashMap<>();
        ordinalMap = new HashMap<>();
        this.addressSize = addressSize;
    }

    public PTXCodeBuilder() {
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

    public PTXCodeBuilder parameters(List<FuncOpWrapper.ParamTable.Info> infoList) {
        paren(_ -> nl().commaNlSeparated(infoList, (info) -> {
            ptxIndent().dot().param().space().paramType(info.javaType);
            space().regName(info.varOp.varName());
            params.add(info.varOp.varName());
        }).nl()).nl();
        return this;
    }

    public void blockBody(Block block, Stream<OpWrapper<?>> ops) {
        if (block.index() == 0) {
            for (Block.Parameter p : block.parameters()) {
                ptxIndent().ld().dot().param();
                resultType(p.type(), false).ptxIndent().space();
                reg(p, getResultType(p.type())).commaSpace().osbrace().regName(params.get(p.index())).csbrace().semicolon().nl();
                paramMap.putIfAbsent(params.get(p.index()), p);
            }
        }
        nl();
        block(block);
        colon().nl();
        ops.forEach(op -> {
            if (op instanceof InvokeOpWrapper invoke && !invoke.isIfaceBufferMethod()) {
                ptxIndent().obrace().nl().ptxIndent().convert(op).ptxNl();
                cbrace().nl();
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

    public PTXCodeBuilder convert(OpWrapper<?> wrappedOp) {
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
                switch (wrappedOp.op()){
                    case CoreOp.BranchOp op -> branch(op);
                    case CoreOp.ConditionalBranchOp op -> condBranch(op);
                    case CoreOp.NegOp op -> neg(op);
                    case PTXPtrOp op -> ptxPtr(op);
                    default -> throw new IllegalStateException("op translation doesn't exist");
                }
            }
        }
        return this;
    }

    public void ptxPtr(PTXPtrOp op) {
        PTXRegister source;
        int offset = 0;

        // TODO: account for nested schema
        // calculate offset
        for (Schema.FieldNode fieldNode : op.schema.rootIfaceType.fields) {
            if (fieldNode.name.equals(op.fieldName)) {
                break;
            }
            switch (fieldNode) {
                case Schema.SchemaNode.Padding f -> {
                    StringBuilder padding = new StringBuilder();
                    Consumer<String> consumer = a -> padding.append(a.replaceAll("[^0-9]", ""));
                    f.toText("", consumer);
                    offset += Integer.parseInt(padding.toString());
                }
                case Schema.FieldNode.PrimitiveFixedArray f -> offset += f.len * 4;
                case Schema.FieldNode.IfaceFixedArray f -> offset += f.len * 4;
                default -> offset += 4;
            }
        }

        if (op.fieldName.equals("array")) {
            source = new PTXRegister(incrOrdinal(addressType()), addressType());
            add().s64().space().regName(source).commaSpace().reg(op.operands().get(0)).commaSpace().reg(op.operands().get(1)).ptxNl();
        } else {
            source = getReg(op.operands().getFirst());
        }

        if (op.resultType.toString().equals("void")) {
            st().global().u32().space().address(source.name(), offset).commaSpace().reg(op.operands().get(2));
        } else {
            ld().global().u32().space().reg(op.result(), PTXRegister.Type.U32).commaSpace().address(source.name(), offset);
        }
    }

    public void fieldLoad(FieldLoadOpWrapper op) {
        if (op.fieldName().equals(Field.KC_X.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(op.result());
            } else {
                mov().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().fieldReg(Field.KC_X);
            }
        } else if (op.fieldName().equals(Field.KC_MAXX.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(op.operandNAsValue(0));
            }
            ld().global().u32().space().fieldReg(Field.KC_MAXX, op.result()).commaSpace()
                    .address(fieldToRegMap.get(Field.KC_ADDR).name(), 4);
        } else {
            ld().global().u32().space().resultReg(op, PTXRegister.Type.U64).commaSpace().reg(op.operandNAsValue(0));
        }
    }

    public void loadKcX(Value value) {
        cvta().to().global().size().space().fieldReg(Field.KC_ADDR).commaSpace()
                .reg(paramMap.get("kc"), addressType()).ptxNl();
        mov().u32().space().fieldReg(Field.NTID_X).commaSpace().percent().regName(Field.NTID_X.toString()).ptxNl();
        mov().u32().space().fieldReg(Field.CTAID_X).commaSpace().percent().regName(Field.CTAID_X.toString()).ptxNl();
        mov().u32().space().fieldReg(Field.TID_X).commaSpace().percent().regName(Field.TID_X.toString()).ptxNl();
        mad().lo().s32().space().fieldReg(Field.KC_X, value).commaSpace().fieldReg(Field.CTAID_X)
                .commaSpace().fieldReg(Field.NTID_X).commaSpace().fieldReg(Field.TID_X).ptxNl();
        st().global().u32().space().address(fieldToRegMap.get(Field.KC_ADDR).name()).commaSpace().fieldReg(Field.KC_X);
    }

    public void fieldStore(FieldStoreOpWrapper op) {
        // TODO: fix
        st().global().u64().space().resultReg(op, PTXRegister.Type.U64).commaSpace().reg(op.operandNAsValue(0));
    }

    PTXCodeBuilder symbol(Op op) {
        return switch (op) {
            case CoreOp.ModOp _ -> rem();
            case CoreOp.MulOp _ -> mul();
            case CoreOp.DivOp _ -> div();
            case CoreOp.AddOp _ -> add();
            case CoreOp.SubOp _ -> sub();
            case CoreOp.LtOp _ -> lt();
            case CoreOp.GtOp _ -> gt();
            case CoreOp.LeOp _ -> le();
            case CoreOp.GeOp _ -> ge();
            case CoreOp.NeqOp _ -> ne();
            case CoreOp.EqOp _ -> eq();
            case CoreOp.OrOp _ -> or();
            case CoreOp.AndOp _ -> and();
            case CoreOp.XorOp _ -> xor();
            case CoreOp.LshlOp _ -> shl();
            case CoreOp.AshrOp _, CoreOp.LshrOp _ -> shr();
            default -> throw new IllegalStateException("Unexpected value");
        };
    }

    public void binaryOperation(BinaryArithmeticOrLogicOperation op) {
        symbol(op.op());
        if (op.resultType().toString().equals("float") && op.op() instanceof CoreOp.DivOp) rn();
        if (!op.resultType().toString().equals("float") && op.op() instanceof CoreOp.MulOp) lo();
        resultType(op.resultType(), true).space();
        resultReg(op, getResultType(op.resultType()));
        commaSpace();
        reg(op.operandNAsValue(0));
        commaSpace();
        reg(op.operandNAsValue(1));
    }

    public void binaryTest(BinaryTestOpWrapper op) {
        setp().dot();
        symbol(op.op()).resultType(op.operandNAsValue(0).type(), true).space();
        resultReg(op, PTXRegister.Type.PREDICATE);
        commaSpace();
        reg(op.operandNAsValue(0));
        commaSpace();
        reg(op.operandNAsValue(1));
    }

    public void conv(ConvOpWrapper op) {
        if (op.resultJavaType().equals(JavaType.LONG)) {
            if (isIndex(op)) {
                mul().wide().s32().space().resultReg(op, PTXRegister.Type.U64).commaSpace()
                        .reg(op.operandNAsValue(0)).commaSpace().intVal(4);
            } else {
                cvt().rn().u64().dot().regType(op.operandNAsValue(0)).space()
                        .resultReg(op, PTXRegister.Type.U64).commaSpace().reg(op.operandNAsValue(0)).ptxNl();
            }
        } else if (op.resultJavaType().equals(JavaType.FLOAT)) {
            cvt().rn().f32().dot().regType(op.operandNAsValue(0)).space()
                    .resultReg(op, PTXRegister.Type.F32).commaSpace().reg(op.operandNAsValue(0));
        } else if (op.resultJavaType().equals(JavaType.DOUBLE)) {
            cvt().rn().f64().dot().regType(op.operandNAsValue(0)).space()
                    .resultReg(op, PTXRegister.Type.F64).commaSpace().reg(op.operandNAsValue(0));
        } else if (op.resultJavaType().equals(JavaType.INT)) {
            cvt().rn().s32().dot().regType(op.operandNAsValue(0)).space()
                    .resultReg(op, PTXRegister.Type.S32).commaSpace().reg(op.operandNAsValue(0));
        } else {
            cvt().rn().s32().dot().regType(op.operandNAsValue(0)).space()
                    .resultReg(op, PTXRegister.Type.S32).commaSpace().reg(op.operandNAsValue(0));
        }
    }

    private boolean isIndex(ConvOpWrapper op) {
        for (Op.Result r : op.result().uses()) {
            if (r.op() instanceof PTXPtrOp) return true;
        }
        return false;
    }

    public void constant(ConstantOpWrapper op) {
        mov().resultType(op.resultType(), false).space().resultReg(op, getResultType(op.resultType())).commaSpace();
        if (op.resultType().toString().equals("float")) {
            append("0f");
            append(Integer.toHexString(Float.floatToIntBits(Float.parseFloat(op.op().value().toString()))).toUpperCase());
            if (op.op().value().toString().equals("0.0")) append("0000000");
        } else {
            append(op.op().value().toString());
        }
    }

    public void javaYield(YieldOpWrapper op) {
        exit();
    }

    public void methodCall(InvokeOpWrapper op) {
        switch (op.methodRef().toString()) {
            // S32Array functions
            case "hat.buffer.S32Array::array(long)int" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                add().s64().space().regName(temp).commaSpace().reg(op.operandNAsValue(0)).commaSpace().reg(op.operandNAsValue(1)).ptxNl();
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(temp.name(), 4);
            }
            case "hat.buffer.S32Array::array(long, int)void" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                add().s64().space().regName(temp).commaSpace().reg(op.operandNAsValue(0)).commaSpace().reg(op.operandNAsValue(1)).ptxNl();
                st().global().u32().space().address(temp.name(), 4).commaSpace().reg(op.operandNAsValue(2));
            }
            case "hat.buffer.S32Array::length()int" -> {
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(getReg(op.operandNAsValue(0)).name());
            }
            // S32Array2D functions
            case "hat.buffer.S32Array2D::array(long, int)void" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                add().s64().space().regName(temp).commaSpace().reg(op.operandNAsValue(0)).commaSpace().reg(op.operandNAsValue(1)).ptxNl();
                st().global().u32().space().address(temp.name(), 8).commaSpace().reg(op.operandNAsValue(2));
            }
            case "hat.buffer.S32Array2D::width()int" -> {
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(getReg(op.operandNAsValue(0)).name());
            }
            case "hat.buffer.S32Array2D::height()int" -> {
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(getReg(op.operandNAsValue(0)).name(), 4);
            }
            // Java Math functions
            case "java.lang.Math::sqrt(double)double" -> {
                sqrt().rn().f32().space().resultReg(op, PTXRegister.Type.F32).commaSpace().getReg(op.operandNAsValue(0)).name();
            }
            // TODO: add these
            case "java.lang.Math::exp(double)double" -> {
                /*
                mov.f32         %f2, 0f3F000000;
                mov.f32         %f3, 0f3BBB989D;
                fma.rn.f32      %f4, %f1, %f3, %f2;
                mov.f32         %f5, 0f3FB8AA3B;
                mov.f32         %f6, 0f437C0000;
                cvt.sat.f32.f32         %f7, %f4;
                mov.f32         %f8, 0f4B400001;
                fma.rm.f32      %f9, %f7, %f6, %f8;
                add.f32         %f10, %f9, 0fCB40007F;
                neg.f32         %f11, %f10;
                fma.rn.f32      %f12, %f1, %f5, %f11;
                mov.f32         %f13, 0f32A57060;
                fma.rn.f32      %f14, %f1, %f13, %f12;
                mov.b32         %r6, %f9;
                shl.b32         %r7, %r6, 23;
                mov.b32         %f15, %r7;
                ex2.approx.ftz.f32      %f16, %f14;
                mul.f32         %f17, %f16, %f15;
                 */
            }
            case "java.lang.Math::log(double)double" -> {
                /*
                mul.f32         %f6, %f5, 0f4B000000;
                setp.lt.f32     %p2, %f5, 0f00800000;
                selp.f32        %f1, %f6, %f5, %p2;
                selp.f32        %f7, 0fC1B80000, 0f00000000, %p2;
                mov.b32         %r6, %f1;
                add.s32         %r7, %r6, -1059760811;
                and.b32         %r8, %r7, -8388608;
                sub.s32         %r9, %r6, %r8;
                mov.b32         %f8, %r9;
                cvt.rn.f32.s32  %f9, %r8;
                mov.f32         %f10, 0f34000000;
                fma.rn.f32      %f11, %f9, %f10, %f7;
                add.f32         %f12, %f8, 0fBF800000;
                mov.f32         %f13, 0f3E1039F6;
                mov.f32         %f14, 0fBE055027;
                fma.rn.f32      %f15, %f14, %f12, %f13;
                mov.f32         %f16, 0fBDF8CDCC;
                fma.rn.f32      %f17, %f15, %f12, %f16;
                mov.f32         %f18, 0f3E0F2955;
                fma.rn.f32      %f19, %f17, %f12, %f18;
                mov.f32         %f20, 0fBE2AD8B9;
                fma.rn.f32      %f21, %f19, %f12, %f20;
                mov.f32         %f22, 0f3E4CED0B;
                fma.rn.f32      %f23, %f21, %f12, %f22;
                mov.f32         %f24, 0fBE7FFF22;
                fma.rn.f32      %f25, %f23, %f12, %f24;
                mov.f32         %f26, 0f3EAAAA78;
                fma.rn.f32      %f27, %f25, %f12, %f26;
                mov.f32         %f28, 0fBF000000;
                fma.rn.f32      %f29, %f27, %f12, %f28;
                mul.f32         %f30, %f12, %f29;
                fma.rn.f32      %f31, %f30, %f12, %f12;
                mov.f32         %f32, 0f3F317218;
                fma.rn.f32      %f35, %f11, %f32, %f31;
                setp.lt.u32     %p3, %r6, 2139095040;
                @%p3 bra        $L__BB0_3;

                mov.f32         %f33, 0f7F800000;
                fma.rn.f32      %f35, %f1, %f33, %f33;

        $L__BB0_3:
                cvta.to.global.u64      %rd4, %rd1;
                setp.eq.f32     %p4, %f1, 0f00000000;
                selp.f32        %f34, 0fFF800000, %f35, %p4;
                 */
                ld().global().u32().space().resultReg(op, PTXRegister.Type.U32).commaSpace().address(getReg(op.operandNAsValue(0)).name(), 4);
            }
            default -> {
                for (int i = 0; i < op.operands().size(); i++) {
                    dot().param().space().paramType(op.operandNAsValue(i).type()).space().param().intVal(i).ptxNl();
                    st().dot().param().paramType(op.operandNAsValue(i).type()).space().osbrace().param().intVal(i).csbrace().commaSpace().reg(op.operandNAsValue(i)).ptxNl();
                }
                dot().param().space().paramType(op.resultType()).space().retVal().ptxNl();
                call().uni().space().oparen().retVal().cparen().commaSpace().append(op.method().getName()).commaSpace();
                final int[] counter = {0};
                paren(_ -> commaSeparated(op.operands(), _ -> param().intVal(counter[0]++))).ptxNl();
                ld().dot().param().paramType(op.resultType()).space().resultReg(op, getResultType(op.resultType())).commaSpace().osbrace().retVal().csbrace();
            }
        }
    }

    public void varDeclaration(VarDeclarationOpWrapper op) {
        ld().dot().param().resultType(op.resultType(), false).space().resultReg(op, addressType()).commaSpace().reg(op.operandNAsValue(0));
    }

    public void varFuncDeclaration(VarFuncDeclarationOpWrapper op) {
        ld().dot().param().resultType(op.resultType(), false).space().resultReg(op, addressType()).commaSpace().reg(op.operandNAsValue(0));
    }

    public void ret(ReturnOpWrapper op) {
        if (op.hasOperands()) {
            st().dot().param();
            if (returnReg.type().equals(PTXRegister.Type.U32)) {
                b32();
            } else if (returnReg.type().equals(PTXRegister.Type.U64)) {
                b64();
            } else {
                dot().regType(returnReg.type());
            }
            space().osbrace().regName(returnReg).csbrace().commaSpace().reg(op.operandNAsValue(0)).ptxNl();
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

    public void neg(CoreOp.NegOp op) {
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

    public PTXCodeBuilder block(Block block) {
        return append("block_").intVal(block.index());
    }

    public PTXCodeBuilder fieldReg(Field ref) {
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

    public PTXCodeBuilder fieldReg(Field ref, Value value) {
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

    public PTXCodeBuilder resultReg(OpWrapper<?> opWrapper, PTXRegister.Type type) {
        return append(addReg(opWrapper.result(), type));
    }

    public PTXCodeBuilder reg(Value val, PTXRegister.Type type) {
        if (varToRegMap.containsKey(val)) {
            return regName(getReg(val));
        } else {
            return append(addReg(val, type));
        }
    }

    public PTXCodeBuilder reg(Value val) {
        return regName(getReg(val));
    }

    public PTXRegister getReg(Value val) {
        if (varToRegMap.get(val) == null && val instanceof Op.Result result && result.op() instanceof CoreOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
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

    public PTXCodeBuilder size() {
        return (addressSize == 32) ? u32() : u64();
    }

    public PTXRegister.Type addressType() {
        return (addressSize == 32) ? PTXRegister.Type.U32 : PTXRegister.Type.U64;
    }

    public PTXCodeBuilder resultType(TypeElement type, boolean signedResult) {
        PTXRegister.Type res = getResultType(type);
        if (signedResult && (res == PTXRegister.Type.U32)) return s32();
        return dot().append(getResultType(type).getName());
    }

    public PTXCodeBuilder paramType(TypeElement type) {
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
    public <I> PTXCodeBuilder commaNlSeparated(Iterable<I> iterable, Consumer<I> c) {
        StreamCounter.of(iterable, (counter, t) -> {
            if (counter.isNotFirst()) {
                comma().nl();
            }
            c.accept(t);
        });
        return self();
    }

    public PTXCodeBuilder address(String address) {
        return osbrace().append(address).csbrace();
    }

    public PTXCodeBuilder address(String address, int offset) {
        osbrace().append(address);
        if (offset == 0) {
            return csbrace();
        } else if (offset > 0) {
            plus();
        }
        return intVal(offset).csbrace();
    }

    public void ptxNl() {
        semicolon().nl().ptxIndent();
    }

    public PTXCodeBuilder commaSpace() {
        return comma().space();
    }

    public PTXCodeBuilder param() {
        return append("param");
    }

    public PTXCodeBuilder global() {
        return dot().append("global");
    }

    public PTXCodeBuilder rn() {
        return dot().append("rn");
    }

    public PTXCodeBuilder to() {
        return dot().append("to");
    }

    public PTXCodeBuilder lo() {
        return dot().append("lo");
    }

    public PTXCodeBuilder wide() {
        return dot().append("wide");
    }

    public PTXCodeBuilder uni() {
        return dot().append("uni");
    }

    public PTXCodeBuilder mov() {
        return append("mov");
    }

    public PTXCodeBuilder setp() {
        return append("setp");
    }

    public PTXCodeBuilder ld() {
        return append("ld");
    }

    public PTXCodeBuilder st() {
        return append("st");
    }

    public PTXCodeBuilder cvt() {
        return append("cvt");
    }

    public PTXCodeBuilder bra() {
        return append("bra");
    }

    public PTXCodeBuilder ret() {
        return append("ret");
    }

    public PTXCodeBuilder rem() {
        return append("rem");
    }

    public PTXCodeBuilder mul() {
        return append("mul");
    }

    public PTXCodeBuilder div() {
        return append("div");
    }

    public PTXCodeBuilder add() {
        return append("add");
    }

    public PTXCodeBuilder sub() {
        return append("sub");
    }

    public PTXCodeBuilder lt() {
        return append("lt");
    }

    public PTXCodeBuilder gt() {
        return append("gt");
    }

    public PTXCodeBuilder le() {
        return append("le");
    }

    public PTXCodeBuilder ge() {
        return append("ge");
    }

    public PTXCodeBuilder ne() {
        return append("ne");
    }

    public PTXCodeBuilder eq() {
        return append("eq");
    }

    public PTXCodeBuilder xor() {
        return append("xor");
    }

    public PTXCodeBuilder or() {
        return append("or");
    }

    public PTXCodeBuilder and() {
        return append("and");
    }

    public PTXCodeBuilder cvta() {
        return append("cvta");
    }

    public PTXCodeBuilder mad() {
        return append("mad");
    }

    public PTXCodeBuilder sqrt() {
        return append("sqrt");
    }

    public PTXCodeBuilder shl() {
        return append("shl");
    }

    public PTXCodeBuilder shr() {
        return append("shr");
    }

    public PTXCodeBuilder neg() {
        return append("neg");
    }

    public PTXCodeBuilder call() {
        return append("call");
    }

    public PTXCodeBuilder exit() {
        return append("exit");
    }

    public PTXCodeBuilder brkpt() {
        return append("brkpt");
    }

    public PTXCodeBuilder ptxIndent() {
        return append("    ");
    }

    public PTXCodeBuilder u32() {
        return dot().append(PTXRegister.Type.U32.getName());
    }

    public PTXCodeBuilder s32() {
        return dot().append(PTXRegister.Type.S32.getName());
    }

    public PTXCodeBuilder f32() {
        return dot().append(PTXRegister.Type.F32.getName());
    }

    public PTXCodeBuilder b32() {
        return dot().append(PTXRegister.Type.B32.getName());
    }

    public PTXCodeBuilder u64() {
        return dot().append(PTXRegister.Type.U64.getName());
    }

    public PTXCodeBuilder s64() {
        return dot().append(PTXRegister.Type.S64.getName());
    }

    public PTXCodeBuilder f64() {
        return dot().append(PTXRegister.Type.F64.getName());
    }

    public PTXCodeBuilder b64() {
        return dot().append(PTXRegister.Type.B64.getName());
    }

    public PTXCodeBuilder version() {
        return dot().append("version");
    }

    public PTXCodeBuilder target() {
        return dot().append("target");
    }

    public PTXCodeBuilder addressSize() {
        return dot().append("address_size");
    }

    public PTXCodeBuilder major(int major) {
        return intVal(major);
    }

    public PTXCodeBuilder minor(int minor) {
        return intVal(minor);
    }

    public PTXCodeBuilder target(String target) {
        return append(target);
    }

    public PTXCodeBuilder size(int addressSize) {
        return intVal(addressSize);
    }

    public PTXCodeBuilder funcName(String funcName) {
        return append(funcName);
    }

    public PTXCodeBuilder visible() {
        return dot().append("visible");
    }

    public PTXCodeBuilder entry() {
        return dot().append("entry");
    }

    public PTXCodeBuilder func() {
        return dot().append("func");
    }

    public PTXCodeBuilder oabrace() {
        return append("<");
    }

    public PTXCodeBuilder cabrace() {
        return append(">");
    }

    public PTXCodeBuilder regName(PTXRegister reg) {
        return append(reg.name());
    }

    public PTXCodeBuilder regName(String regName) {
        return append(regName);
    }

    public PTXCodeBuilder regType(Value val) {
        return append(getReg(val).type().getName());
    }

    public PTXCodeBuilder regType(PTXRegister.Type t) {
        return append(t.getName());
    }

    public PTXCodeBuilder regTypePrefix(PTXRegister.Type t) {
        return append(t.getRegPrefix());
    }

    public PTXCodeBuilder reg() {
        return dot().append("reg");
    }

    public PTXCodeBuilder retVal() {
        return append("retval");
    }

    public PTXCodeBuilder intVal(int i) {
        return append(String.valueOf(i));
    }
}