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

import java.lang.foreign.MemoryLayout;
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

    public PTXCodeBuilder(int addressSize) {
        varToRegMap = new HashMap<>();
        paramNames = new ArrayList<>();
        fieldToRegMap = new HashMap<>();
        paramObjects = new ArrayList<>();
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
                if (invoke.methodRef().refType().toString().equals("java.lang.Math")) {
                    ptxIndent().convert(op).nl();
                } else {
                    ptxIndent().obrace().nl().ptxIndent().convert(op).ptxNl();
                    cbrace().nl();
                }
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

        offset = (int) op.boundSchema.groupLayout().byteOffset(MemoryLayout.PathElement.groupElement(op.fieldName));

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
        if (getResultType(op.resultType()).getBasicType().equals(PTXRegister.Type.BasicType.FLOATING)
                && (op.op() instanceof CoreOp.DivOp || op.op() instanceof CoreOp.MulOp)) {
            rn();
        } else if (!getResultType(op.resultType()).getBasicType().equals(PTXRegister.Type.BasicType.FLOATING)
                && op.op() instanceof CoreOp.MulOp) {
            lo();
        }
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
                cvt().u64().dot().regType(op.operandNAsValue(0)).space()
                        .resultReg(op, PTXRegister.Type.U64).commaSpace().reg(op.operandNAsValue(0)).ptxNl();
            }
        } else if (op.resultJavaType().equals(JavaType.FLOAT)) {
            cvt().rn().f32().dot().regType(op.operandNAsValue(0)).space()
                    .resultReg(op, PTXRegister.Type.F32).commaSpace().reg(op.operandNAsValue(0));
        } else if (op.resultJavaType().equals(JavaType.DOUBLE)) {
            cvt();
            if (op.operandNAsValue(0).type().equals(JavaType.INT)) {
                rn();
            }
            f64().dot().regType(op.operandNAsValue(0)).space()
                    .resultReg(op, PTXRegister.Type.F64).commaSpace().reg(op.operandNAsValue(0));
        } else if (op.resultJavaType().equals(JavaType.INT)) {
            cvt();
            if (op.operandNAsValue(0).type().equals(JavaType.DOUBLE) || op.operandNAsValue(0).type().equals(JavaType.FLOAT)) {
                rzi();
            } else {
                rn();
            }
            s32().dot().regType(op.operandNAsValue(0)).space()
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
            if (op.op().value().toString().equals("0.0")) {
                floatVal("00000000");
            } else {
                floatVal(Integer.toHexString(Float.floatToIntBits(Float.parseFloat(op.op().value().toString()))).toUpperCase());
            }
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
                sqrt().rn().f64().space().resultReg(op, PTXRegister.Type.F64).commaSpace().reg(op.operandNAsValue(0));
                semicolon();
            }
            case "java.lang.Math::exp(double)double" -> {
                if (getReg(op.operandNAsValue(0)).type().equals(PTXRegister.Type.F32)) {
                    PTXRegister[] fRegs = new PTXRegister[15];
                    for (int i = 0; i < fRegs.length; i++) {
                        fRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.F32), PTXRegister.Type.F32);
                    }
                    PTXRegister[] bRegs = new PTXRegister[2];
                    for (int i = 0; i < bRegs.length; i++) {
                        bRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32);
                    }

                    mov().f32().space().regName(fRegs[0]).commaSpace().floatVal("3F000000").ptxNl();
                    mov().f32().space().regName(fRegs[1]).commaSpace().floatVal("3BBB989D").ptxNl();
                    fma().rn().f32().space().regName(fRegs[2]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().regName(fRegs[1]).commaSpace().regName(fRegs[0]).ptxNl();
                    mov().f32().space().regName(fRegs[3]).commaSpace().floatVal("3FB8AA3B").ptxNl();
                    mov().f32().space().regName(fRegs[4]).commaSpace().floatVal("437C0000").ptxNl();
                    cvt().sat().f32().f32().space().regName(fRegs[5]).commaSpace().regName(fRegs[2]).ptxNl();
                    mov().f32().space().regName(fRegs[6]).commaSpace().floatVal("4B400001").ptxNl();
                    fma().rm().f32().space().regName(fRegs[7]).commaSpace().regName(fRegs[5]).commaSpace().regName(fRegs[4]).commaSpace().regName(fRegs[6]).ptxNl();
                    add().f32().space().regName(fRegs[8]).commaSpace().regName(fRegs[7]).commaSpace().floatVal("CB40007F").ptxNl();
                    neg().f32().space().regName(fRegs[9]).commaSpace().regName(fRegs[8]).ptxNl();
                    fma().rn().f32().space().regName(fRegs[10]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().regName(fRegs[3]).commaSpace().regName(fRegs[9]).ptxNl();
                    mov().f32().space().regName(fRegs[11]).commaSpace().floatVal("32A57060").ptxNl();
                    fma().rn().f32().space().regName(fRegs[12]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().regName(fRegs[11]).commaSpace().regName(fRegs[10]).ptxNl();
                    mov().b32().space().regName(bRegs[0]).commaSpace().regName(fRegs[7]).ptxNl();
                    shl().b32().space().regName(bRegs[1]).commaSpace().regName(bRegs[0]).commaSpace().intVal(23).ptxNl();
                    mov().b32().space().regName(fRegs[13]).commaSpace().regName(bRegs[1]).ptxNl();
                    ex2().approx().ftz().f32().space().regName(fRegs[14]).commaSpace().regName(fRegs[12]).ptxNl();
                    mul().f32().space().resultReg(op, PTXRegister.Type.F32).commaSpace().regName(fRegs[14]).commaSpace().regName(fRegs[13]);
                    semicolon();
                } else {
                    PTXRegister[] fRegs = new PTXRegister[2];
                    for (int i = 0; i < fRegs.length; i++) {
                        fRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.F32), PTXRegister.Type.F32);
                    }
                    PTXRegister[] fdRegs = new PTXRegister[34];
                    for (int i = 0; i < fdRegs.length; i++) {
                        fdRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.F64), PTXRegister.Type.F64);
                    }
                    PTXRegister[] bRegs = new PTXRegister[15];
                    for (int i = 0; i < bRegs.length; i++) {
                        bRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32);
                    }
                    PTXRegister[] pRegs = new PTXRegister[3];
                    for (int i = 0; i < pRegs.length; i++) {
                        pRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.PREDICATE), PTXRegister.Type.PREDICATE);
                    }

                    mov().f64().space().regName(fdRegs[0]).commaSpace().doubleVal("4338000000000000").ptxNl();
                    mov().f64().space().regName(fdRegs[1]).commaSpace().doubleVal("3FF71547652B82FE").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[2]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().regName(fdRegs[1]).commaSpace().regName(fdRegs[0]).ptxNl();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().regName(bRegs[0]).commaSpace().percent().temp().cbrace().commaSpace().regName(fdRegs[2]).ptxNl();
                    cbrace().nl().ptxIndent();
                    mov().f64().space().regName(fdRegs[3]).commaSpace().doubleVal("C338000000000000").ptxNl();
                    add().rn().f64().space().regName(fdRegs[4]).commaSpace().regName(fdRegs[2]).commaSpace().regName(fdRegs[3]).ptxNl();
                    mov().f64().space().regName(fdRegs[5]).commaSpace().doubleVal("BFE62E42FEFA39EF").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[6]).commaSpace().regName(fdRegs[4]).commaSpace().regName(fdRegs[5]).commaSpace().reg(op.operandNAsValue(0)).ptxNl();
                    mov().f64().space().regName(fdRegs[7]).commaSpace().doubleVal("BC7ABC9E3B39803F").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[8]).commaSpace().regName(fdRegs[4]).commaSpace().regName(fdRegs[7]).commaSpace().regName(fdRegs[6]).ptxNl();
                    mov().f64().space().regName(fdRegs[9]).commaSpace().doubleVal("3E928AF3FCA213EA").ptxNl();
                    mov().f64().space().regName(fdRegs[10]).commaSpace().doubleVal("3E5ADE1569CE2BDF").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[11]).commaSpace().regName(fdRegs[10]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[9]).ptxNl();
                    mov().f64().space().regName(fdRegs[12]).commaSpace().doubleVal("3EC71DEE62401315").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[13]).commaSpace().regName(fdRegs[11]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[12]).ptxNl();
                    mov().f64().space().regName(fdRegs[14]).commaSpace().doubleVal("3EFA01997C89EB71").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[15]).commaSpace().regName(fdRegs[13]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[14]).ptxNl();
                    mov().f64().space().regName(fdRegs[16]).commaSpace().doubleVal("3F2A01A014761F65").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[17]).commaSpace().regName(fdRegs[15]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[16]).ptxNl();
                    mov().f64().space().regName(fdRegs[18]).commaSpace().doubleVal("3F56C16C1852B7AF").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[19]).commaSpace().regName(fdRegs[17]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[18]).ptxNl();
                    mov().f64().space().regName(fdRegs[20]).commaSpace().doubleVal("3F81111111122322").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[21]).commaSpace().regName(fdRegs[19]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[20]).ptxNl();
                    mov().f64().space().regName(fdRegs[22]).commaSpace().doubleVal("3FA55555555502A1").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[23]).commaSpace().regName(fdRegs[21]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[22]).ptxNl();
                    mov().f64().space().regName(fdRegs[24]).commaSpace().doubleVal("3FC5555555555511").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[25]).commaSpace().regName(fdRegs[23]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[24]).ptxNl();
                    mov().f64().space().regName(fdRegs[26]).commaSpace().doubleVal("3FE000000000000B").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[27]).commaSpace().regName(fdRegs[25]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[26]).ptxNl();
                    mov().f64().space().regName(fdRegs[28]).commaSpace().doubleVal("3FF0000000000000").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[29]).commaSpace().regName(fdRegs[27]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[28]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[30]).commaSpace().regName(fdRegs[29]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[28]).ptxNl();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().regName(bRegs[1]).commaSpace().percent().temp().cbrace().commaSpace().regName(fdRegs[30]).ptxNl();
                    cbrace().nl().ptxIndent();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().percent().temp().commaSpace().regName(bRegs[2]).cbrace().commaSpace().regName(fdRegs[30]).ptxNl();
                    cbrace().nl().ptxIndent();
                    shl().b32().space().regName(bRegs[3]).commaSpace().regName(bRegs[0]).commaSpace().intVal(20).ptxNl();
                    add().s32().space().regName(bRegs[4]).commaSpace().regName(bRegs[2]).commaSpace().regName(bRegs[3]).ptxNl();
                    mov().b64().space().resultReg(op, PTXRegister.Type.F64).commaSpace().obrace().regName(bRegs[1]).commaSpace().regName(bRegs[4]).cbrace().ptxNl();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().percent().temp().commaSpace().regName(bRegs[5]).cbrace().commaSpace().reg(op.operandNAsValue(0)).ptxNl();
                    cbrace().nl().ptxIndent();
                    mov().b32().space().regName(fRegs[0]).commaSpace().regName(bRegs[5]).ptxNl();
                    abs().f32().space().regName(fRegs[1]).commaSpace().regName(fRegs[0]).ptxNl();
                    setp().dot().lt().f32().space().regName(pRegs[0]).commaSpace().regName(fRegs[1]).commaSpace().floatVal("4086232B").ptxNl();
                    at().regName(pRegs[0]).space().bra().space().append("endExp").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).ptxNl();
                    setp().dot().lt().f64().space().regName(pRegs[1]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().doubleVal("0000000000000000").ptxNl();
                    add().f64().space().regName(fdRegs[31]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().doubleVal("7FF0000000000000").ptxNl();
                    selp().f64().space().resultReg(op, PTXRegister.Type.F64).commaSpace().doubleVal("0000000000000000").commaSpace().regName(fdRegs[31]).commaSpace().regName(pRegs[1]).ptxNl();
                    setp().dot().geu().f32().space().regName(pRegs[2]).commaSpace().regName(fRegs[1]).commaSpace().floatVal("40874800").ptxNl();
                    at().regName(pRegs[2]).space().bra().space().append("endExp").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).ptxNl();
                    shr().u32().space().regName(bRegs[6]).commaSpace().regName(bRegs[0]).commaSpace().intVal(31).ptxNl();
                    add().s32().space().regName(bRegs[7]).commaSpace().regName(bRegs[0]).commaSpace().regName(bRegs[6]).ptxNl();
                    shr().s32().space().regName(bRegs[8]).commaSpace().regName(bRegs[7]).commaSpace().intVal(1).ptxNl();
                    shl().b32().space().regName(bRegs[9]).commaSpace().regName(bRegs[8]).commaSpace().intVal(20).ptxNl();
                    add().s32().space().regName(bRegs[10]).commaSpace().regName(bRegs[2]).commaSpace().regName(bRegs[9]).ptxNl();
                    mov().b64().space().regName(fdRegs[32]).commaSpace().obrace().regName(bRegs[1]).commaSpace().regName(bRegs[10]).cbrace().ptxNl();
                    sub().s32().space().regName(bRegs[11]).commaSpace().regName(bRegs[0]).commaSpace().regName(bRegs[8]).ptxNl();
                    shl().b32().space().regName(bRegs[12]).commaSpace().regName(bRegs[11]).commaSpace().intVal(20).ptxNl();
                    add().s32().space().regName(bRegs[13]).commaSpace().regName(bRegs[12]).commaSpace().intVal(1072693248).ptxNl();
                    mov().u32().space().regName(bRegs[14]).commaSpace().intVal(0).ptxNl();
                    mov().b64().space().regName(fdRegs[33]).commaSpace().obrace().regName(bRegs[14]).commaSpace().regName(bRegs[13]).cbrace().ptxNl();
                    mul().f64().space().resultReg(op, PTXRegister.Type.F64).commaSpace().regName(fdRegs[32]).commaSpace().regName(fdRegs[33]).semicolon().nl();
                    append("endExp").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).colon();
                }
            }
            case "java.lang.Math::log(double)double" -> {
                if (getReg(op.operandNAsValue(0)).type().equals(PTXRegister.Type.F32)) {
                    PTXRegister[] fRegs = new PTXRegister[30];
                    for (int i = 0; i < fRegs.length; i++) {
                        fRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.F32), PTXRegister.Type.F32);
                    }
                    PTXRegister[] bRegs = new PTXRegister[4];
                    for (int i = 0; i < bRegs.length; i++) {
                        bRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32);
                    }
                    PTXRegister[] pRegs = new PTXRegister[3];
                    for (int i = 0; i < pRegs.length; i++) {
                        pRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.PREDICATE), PTXRegister.Type.PREDICATE);
                    }

                    mul().f32().space().regName(fRegs[0]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().floatVal("4B000000").ptxNl();
                    setp().dot().lt().f32().space().regName(pRegs[0]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().floatVal("00800000").ptxNl();
                    selp().f32().space().regName(fRegs[1]).commaSpace().regName(fRegs[0]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().regName(pRegs[0]).ptxNl();
                    selp().f32().space().regName(fRegs[2]).commaSpace().floatVal("C1B80000").commaSpace().floatVal("00000000").commaSpace().regName(pRegs[0]).ptxNl();
                    mov().b32().space().regName(bRegs[0]).commaSpace().regName(fRegs[1]).ptxNl();
                    add().s32().space().regName(bRegs[1]).commaSpace().regName(bRegs[0]).commaSpace().intVal(-1059760811).ptxNl();
                    and().b32().space().regName(bRegs[2]).commaSpace().regName(bRegs[1]).commaSpace().intVal(-8388608).ptxNl();
                    sub().s32().space().regName(bRegs[3]).commaSpace().regName(bRegs[0]).commaSpace().regName(bRegs[2]).ptxNl();
                    mov().b32().space().regName(fRegs[3]).commaSpace().regName(bRegs[3]).ptxNl();
                    cvt().rn().f32().s32().space().regName(fRegs[4]).commaSpace().regName(bRegs[2]).ptxNl();
                    mov().f32().space().regName(fRegs[5]).commaSpace().floatVal("34000000").ptxNl();
                    fma().rn().f32().space().regName(fRegs[6]).commaSpace().regName(fRegs[4]).commaSpace().regName(fRegs[5]).commaSpace().regName(fRegs[2]).ptxNl();
                    add().f32().space().regName(fRegs[7]).commaSpace().regName(fRegs[3]).commaSpace().floatVal("BF800000").ptxNl();
                    mov().f32().space().regName(fRegs[8]).commaSpace().floatVal("3E1039F6").ptxNl();
                    mov().f32().space().regName(fRegs[9]).commaSpace().floatVal("BE055027").ptxNl();
                    fma().rn().f32().space().regName(fRegs[10]).commaSpace().regName(fRegs[9]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[8]).ptxNl();
                    mov().f32().space().regName(fRegs[11]).commaSpace().floatVal("BDF8CDCC").ptxNl();
                    fma().rn().f32().space().regName(fRegs[12]).commaSpace().regName(fRegs[10]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[11]).ptxNl();
                    mov().f32().space().regName(fRegs[13]).commaSpace().floatVal("3E0F2955").ptxNl();
                    fma().rn().f32().space().regName(fRegs[14]).commaSpace().regName(fRegs[12]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[13]).ptxNl();
                    mov().f32().space().regName(fRegs[15]).commaSpace().floatVal("BE2AD8B9").ptxNl();
                    fma().rn().f32().space().regName(fRegs[16]).commaSpace().regName(fRegs[14]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[15]).ptxNl();
                    mov().f32().space().regName(fRegs[17]).commaSpace().floatVal("3E4CED0B").ptxNl();
                    fma().rn().f32().space().regName(fRegs[18]).commaSpace().regName(fRegs[16]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[17]).ptxNl();
                    mov().f32().space().regName(fRegs[19]).commaSpace().floatVal("BE7FFF22").ptxNl();
                    fma().rn().f32().space().regName(fRegs[20]).commaSpace().regName(fRegs[18]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[19]).ptxNl();
                    mov().f32().space().regName(fRegs[21]).commaSpace().floatVal("3EAAAA78").ptxNl();
                    fma().rn().f32().space().regName(fRegs[22]).commaSpace().regName(fRegs[20]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[21]).ptxNl();
                    mov().f32().space().regName(fRegs[23]).commaSpace().floatVal("BF000000").ptxNl();
                    fma().rn().f32().space().regName(fRegs[24]).commaSpace().regName(fRegs[22]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[23]).ptxNl();
                    mul().f32().space().regName(fRegs[25]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[24]).ptxNl();
                    fma().rn().f32().space().regName(fRegs[26]).commaSpace().regName(fRegs[25]).commaSpace().regName(fRegs[7]).commaSpace().regName(fRegs[7]).ptxNl();
                    mov().f32().space().regName(fRegs[27]).commaSpace().floatVal("3F317218").ptxNl();
                    fma().rn().f32().space().regName(fRegs[29]).commaSpace().regName(fRegs[6]).commaSpace().regName(fRegs[27]).commaSpace().regName(fRegs[26]).ptxNl();
                    setp().dot().lt().u32().space().regName(pRegs[1]).commaSpace().regName(bRegs[0]).commaSpace().intVal(2139095040).ptxNl();
                    at().regName(pRegs[1]).space().bra().space().append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).ptxNl();
                    mov().f32().space().regName(fRegs[28]).commaSpace().floatVal("7F800000").ptxNl();
                    fma().rn().f32().space().regName(fRegs[29]).commaSpace().regName(fRegs[2]).commaSpace().regName(fRegs[28]).commaSpace().regName(fRegs[28]).semicolon().nl();
                    append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).colon().nl().ptxIndent();
                    setp().dot().eq().f32().space().regName(pRegs[2]).commaSpace().regName(fRegs[2]).commaSpace().floatVal("00000000").ptxNl();
                    selp().f32().space().resultReg(op, PTXRegister.Type.F32).commaSpace().floatVal("FF800000").commaSpace().regName(fRegs[29]).commaSpace().regName(pRegs[2]);
                    semicolon();
                } else {
                    PTXRegister[] fRegs = new PTXRegister[1];
                    for (int i = 0; i < fRegs.length; i++) {
                        fRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.F32), PTXRegister.Type.F32);
                    }
                    PTXRegister[] fdRegs = new PTXRegister[47];
                    for (int i = 0; i < fdRegs.length; i++) {
                        fdRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.F64), PTXRegister.Type.F64);
                    }
                    PTXRegister[] bRegs = new PTXRegister[15];
                    for (int i = 0; i < bRegs.length; i++) {
                        bRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32);
                    }
                    PTXRegister[] pRegs = new PTXRegister[4];
                    for (int i = 0; i < pRegs.length; i++) {
                        pRegs[i] = new PTXRegister(incrOrdinal(PTXRegister.Type.PREDICATE), PTXRegister.Type.PREDICATE);
                    }

                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().percent().temp().commaSpace().regName(bRegs[11]).cbrace().commaSpace().reg(op.operandNAsValue(0)).ptxNl();
                    cbrace().nl().ptxIndent();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().regName(bRegs[12]).commaSpace().percent().temp().cbrace().commaSpace().reg(op.operandNAsValue(0)).ptxNl();
                    cbrace().nl().ptxIndent();
                    setp().dot().gt().s32().space().regName(pRegs[0]).commaSpace().regName(bRegs[11]).commaSpace().intVal(1048575).ptxNl();
                    mov().u32().space().regName(bRegs[13]).commaSpace().intVal(-1023).ptxNl();
                    at().regName(pRegs[0]).space().bra().space().append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(0).ptxNl();
                    mul().f64().space().reg(op.operandNAsValue(0)).commaSpace().reg(op.operandNAsValue(0)).commaSpace().doubleVal("4350000000000000").ptxNl();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().percent().temp().commaSpace().regName(bRegs[11]).cbrace().commaSpace().reg(op.operandNAsValue(0)).ptxNl();
                    cbrace().nl().ptxIndent();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().regName(bRegs[12]).commaSpace().percent().temp().cbrace().commaSpace().reg(op.operandNAsValue(0)).ptxNl();
                    cbrace().nl().ptxIndent();
                    mov().u32().space().regName(bRegs[13]).commaSpace().intVal(-1077).semicolon().nl();
                    append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(0).colon().nl().ptxIndent();
                    add().s32().space().regName(bRegs[0]).commaSpace().regName(bRegs[11]).commaSpace().intVal(-1).ptxNl();
                    setp().dot().lt().u32().space().regName(pRegs[1]).commaSpace().regName(bRegs[0]).commaSpace().intVal(2146435071).ptxNl();
                    at().regName(pRegs[1]).space().bra().space().append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(2).ptxNl();
                    bra().uni().space().append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(1).semicolon().nl();
                    append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(2).colon().nl().ptxIndent();
                    shr().u32().space().regName(bRegs[2]).commaSpace().regName(bRegs[11]).commaSpace().intVal(20).ptxNl();
                    add().s32().space().regName(bRegs[14]).commaSpace().regName(bRegs[13]).commaSpace().regName(bRegs[2]).ptxNl();
                    and().b32().space().regName(bRegs[3]).commaSpace().regName(bRegs[11]).commaSpace().intVal(-2146435073).ptxNl();
                    or().b32().space().regName(bRegs[4]).commaSpace().regName(bRegs[3]).commaSpace().intVal(1072693248).ptxNl();
                    mov().b64().space().regName(fdRegs[46]).commaSpace().obrace().regName(bRegs[12]).commaSpace().regName(bRegs[4]).cbrace().ptxNl();
                    setp().dot().lt().s32().space().regName(pRegs[3]).commaSpace().regName(bRegs[4]).commaSpace().intVal(1073127583).ptxNl();
                    at().regName(pRegs[3]).space().bra().space().append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(4).ptxNl();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().regName(bRegs[5]).commaSpace().percent().temp().cbrace().commaSpace().regName(fdRegs[46]).ptxNl();
                    cbrace().nl().ptxIndent();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().percent().temp().commaSpace().regName(bRegs[6]).cbrace().commaSpace().regName(fdRegs[46]).ptxNl();
                    cbrace().nl().ptxIndent();
                    add().s32().space().regName(bRegs[7]).commaSpace().regName(bRegs[6]).commaSpace().intVal(-1048576).ptxNl();
                    mov().b64().space().regName(fdRegs[46]).commaSpace().obrace().regName(bRegs[5]).commaSpace().regName(bRegs[7]).cbrace().ptxNl();
                    add().s32().space().regName(bRegs[14]).commaSpace().regName(bRegs[14]).commaSpace().intVal(1).semicolon().nl();
                    append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(4).colon().nl().ptxIndent();
                    add().f64().space().regName(fdRegs[2]).commaSpace().regName(fdRegs[46]).commaSpace().doubleVal("3FF0000000000000").ptxNl();
                    mov().f64().space().regName(fdRegs[3]).commaSpace().doubleVal("3FF0000000000000").ptxNl();
                    rcp().approx().ftz().f64().space().regName(fdRegs[4]).commaSpace().regName(fdRegs[2]).ptxNl();
                    neg().f64().space().regName(fdRegs[5]).commaSpace().regName(fdRegs[2]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[6]).commaSpace().regName(fdRegs[5]).commaSpace().regName(fdRegs[4]).commaSpace().regName(fdRegs[3]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[7]).commaSpace().regName(fdRegs[6]).commaSpace().regName(fdRegs[6]).commaSpace().regName(fdRegs[6]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[8]).commaSpace().regName(fdRegs[7]).commaSpace().regName(fdRegs[4]).commaSpace().regName(fdRegs[4]).ptxNl();
                    add().f64().space().regName(fdRegs[9]).commaSpace().regName(fdRegs[46]).commaSpace().doubleVal("BFF0000000000000").ptxNl();
                    mul().f64().space().regName(fdRegs[10]).commaSpace().regName(fdRegs[9]).commaSpace().regName(fdRegs[8]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[11]).commaSpace().regName(fdRegs[9]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[10]).ptxNl();
                    mul().f64().space().regName(fdRegs[12]).commaSpace().regName(fdRegs[11]).commaSpace().regName(fdRegs[11]).ptxNl();
                    mov().f64().space().regName(fdRegs[13]).commaSpace().doubleVal("3ED0EE258B7A8B04").ptxNl();
                    mov().f64().space().regName(fdRegs[14]).commaSpace().doubleVal("3EB1380B3AE80F1E").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[15]).commaSpace().regName(fdRegs[14]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[13]).ptxNl();
                    mov().f64().space().regName(fdRegs[16]).commaSpace().doubleVal("3EF3B2669F02676F").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[17]).commaSpace().regName(fdRegs[15]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[16]).ptxNl();
                    mov().f64().space().regName(fdRegs[18]).commaSpace().doubleVal("3F1745CBA9AB0956").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[19]).commaSpace().regName(fdRegs[17]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[18]).ptxNl();
                    mov().f64().space().regName(fdRegs[20]).commaSpace().doubleVal("3F3C71C72D1B5154").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[21]).commaSpace().regName(fdRegs[19]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[20]).ptxNl();
                    mov().f64().space().regName(fdRegs[22]).commaSpace().doubleVal("3F624924923BE72D").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[23]).commaSpace().regName(fdRegs[21]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[22]).ptxNl();
                    mov().f64().space().regName(fdRegs[24]).commaSpace().doubleVal("3F8999999999A3C4").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[25]).commaSpace().regName(fdRegs[23]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[24]).ptxNl();
                    mov().f64().space().regName(fdRegs[26]).commaSpace().doubleVal("3FB5555555555554").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[27]).commaSpace().regName(fdRegs[25]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[26]).ptxNl();
                    sub().f64().space().regName(fdRegs[28]).commaSpace().regName(fdRegs[9]).commaSpace().regName(fdRegs[11]).ptxNl();
                    add().f64().space().regName(fdRegs[29]).commaSpace().regName(fdRegs[28]).commaSpace().regName(fdRegs[28]).ptxNl();
                    neg().f64().space().regName(fdRegs[30]).commaSpace().regName(fdRegs[11]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[31]).commaSpace().regName(fdRegs[30]).commaSpace().regName(fdRegs[9]).commaSpace().regName(fdRegs[29]).ptxNl();
                    mul().f64().space().regName(fdRegs[32]).commaSpace().regName(fdRegs[8]).commaSpace().regName(fdRegs[31]).ptxNl();
                    mul().f64().space().regName(fdRegs[33]).commaSpace().regName(fdRegs[12]).commaSpace().regName(fdRegs[27]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[34]).commaSpace().regName(fdRegs[33]).commaSpace().regName(fdRegs[11]).commaSpace().regName(fdRegs[32]).ptxNl();
                    xor().b32().space().regName(bRegs[8]).commaSpace().regName(bRegs[14]).commaSpace().intVal(-2147483648).ptxNl();
                    mov().u32().space().regName(bRegs[9]).commaSpace().intVal(-2147483648).ptxNl();
                    mov().u32().space().regName(bRegs[10]).commaSpace().intVal(1127219200).ptxNl();
                    mov().b64().space().regName(fdRegs[35]).commaSpace().obrace().regName(bRegs[8]).commaSpace().regName(bRegs[10]).cbrace().ptxNl();
                    mov().b64().space().regName(fdRegs[36]).commaSpace().obrace().regName(bRegs[9]).commaSpace().regName(bRegs[10]).cbrace().ptxNl();
                    sub().f64().space().regName(fdRegs[37]).commaSpace().regName(fdRegs[35]).commaSpace().regName(fdRegs[36]).ptxNl();
                    mov().f64().space().regName(fdRegs[38]).commaSpace().doubleVal("3FE62E42FEFA39EF").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[39]).commaSpace().regName(fdRegs[37]).commaSpace().regName(fdRegs[38]).commaSpace().regName(fdRegs[11]).ptxNl();
                    neg().f64().space().regName(fdRegs[40]).commaSpace().regName(fdRegs[37]).ptxNl();
                    fma().rn().f64().space().regName(fdRegs[41]).commaSpace().regName(fdRegs[40]).commaSpace().regName(fdRegs[38]).commaSpace().regName(fdRegs[39]).ptxNl();
                    sub().f64().space().regName(fdRegs[42]).commaSpace().regName(fdRegs[41]).commaSpace().regName(fdRegs[11]).ptxNl();
                    sub().f64().space().regName(fdRegs[43]).commaSpace().regName(fdRegs[34]).commaSpace().regName(fdRegs[42]).ptxNl();
                    mov().f64().space().regName(fdRegs[44]).commaSpace().doubleVal("3C7ABC9E3B39803F").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[45]).commaSpace().regName(fdRegs[37]).commaSpace().regName(fdRegs[44]).commaSpace().regName(fdRegs[43]).ptxNl();
                    add().f64().space().resultReg(op, PTXRegister.Type.F64).commaSpace().regName(fdRegs[39]).commaSpace().regName(fdRegs[45]).ptxNl();
                    bra().uni().space().append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(5).semicolon().nl();
                    append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(1).colon().nl().ptxIndent();
                    mov().f64().space().regName(fdRegs[0]).commaSpace().doubleVal("7FF0000000000000").ptxNl();
                    fma().rn().f64().space().regName(fdRegs[1]).commaSpace().reg(op.operandNAsValue(0)).commaSpace().regName(fdRegs[0]).commaSpace().regName(fdRegs[0]).ptxNl();
                    obrace().nl().ptxIndent();
                    reg().space().b32().space().percent().temp().ptxNl();
                    mov().b64().space().obrace().percent().temp().commaSpace().regName(bRegs[1]).cbrace().commaSpace().reg(op.operandNAsValue(0)).ptxNl();
                    cbrace().nl().ptxIndent();
                    mov().b32().space().regName(fRegs[0]).commaSpace().regName(bRegs[1]).ptxNl();
                    setp().dot().eq().f32().space().regName(pRegs[2]).commaSpace().regName(fRegs[0]).commaSpace().floatVal("00000000").ptxNl();
                    selp().f64().space().resultReg(op, PTXRegister.Type.F64).commaSpace().doubleVal("FFF0000000000000").commaSpace().regName(fdRegs[1]).commaSpace().regName(pRegs[2]).semicolon().nl();
                    append("endLog").intVal(op.op().location().line()).underscore().intVal(op.op().location().column()).underscore().intVal(5).colon();
                }
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

    public PTXCodeBuilder rm() {
        return dot().append("rm");
    }

    public PTXCodeBuilder rzi() {
        return dot().append("rzi");
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

    public PTXCodeBuilder sat() {
        return dot().append("sat");
    }

    public PTXCodeBuilder ftz() {
        return dot().append("ftz");
    }

    public PTXCodeBuilder approx() {
        return dot().append("approx");
    }

    public PTXCodeBuilder mov() {
        return append("mov");
    }

    public PTXCodeBuilder setp() {
        return append("setp");
    }

    public PTXCodeBuilder selp() {
        return append("selp");
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

    public PTXCodeBuilder rcp() {
        return append("rcp");
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

    public PTXCodeBuilder geu() {
        return append("geu");
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

    public PTXCodeBuilder fma() {
        return append("fma");
    }

    public PTXCodeBuilder sqrt() {
        return append("sqrt");
    }

    public PTXCodeBuilder abs() {
        return append("abs");
    }

    public PTXCodeBuilder ex2() {
        return append("ex2");
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

    public PTXCodeBuilder temp() {
        return append("temp");
    }

    public PTXCodeBuilder intVal(int i) {
        return append(String.valueOf(i));
    }

    public PTXCodeBuilder floatVal(String s) {
        return append("0f").append(s);
    }

    public PTXCodeBuilder doubleVal(String s) {
        return append("0d").append(s);
    }
}