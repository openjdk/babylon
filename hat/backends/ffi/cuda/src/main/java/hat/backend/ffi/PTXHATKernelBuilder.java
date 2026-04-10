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

import optkl.FuncOpParams;
import optkl.ParamVar;
import optkl.codebuilders.CodeBuilder;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static optkl.OpHelper.FieldAccess.fieldAccess;
import static optkl.OpHelper.Invoke;

import static optkl.OpHelper.Invoke.invoke;


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
        version().sp().major(major).dot().minor(minor).nl();
        target().sp().target(target).nl();
        addressSize().sp().size(addressSize);
    }

    public void functionHeader(String funcName, boolean entry, CodeType yieldType) {
        if (entry) {
            visible().sp().entry().sp();
        } else {
            func().sp();
        }
        if (!yieldType.toString().equals("void")) {
            returnReg = new PTXRegister(getOrdinal(getResultType(yieldType)), getResultType(yieldType));
            returnReg.name("%retReg");
            oparen().dot().param().sp().paramType(yieldType);
            sp().regName(returnReg).cparen().sp();
        }
        funcName(funcName);
    }

    public PTXHATKernelBuilder parameters(List<FuncOpParams.Info> infoList) {
        paren(_ ->
                nl()
                        .commaNlSeparated(
                        infoList,
                        info -> {
                            ptxIndent().dot().param().sp().paramType(info.javaType);
                            sp().regName(info.varOp.varName());
                            paramNames.add(info.varOp.varName());
                        }
                        ).nl()).nl();
        return this;
    }

    public void blockBody(MethodHandles.Lookup lookup,Block block, Stream<Op> ops) {
        if (block.index() == 0) {
            for (Block.Parameter p : block.parameters()) {
                ptxIndent().ld().dot().param();
                resultType(p.type(), false).ptxIndent().sp();
                reg(p, getResultType(p.type())).csp().osbrace().regName(paramNames.get(p.index())).csbrace().semicolon().nl();
                paramObjects.add(p);
            }
        }
        nl();
        block(block);
        colon().nl();
        ops.forEach(op -> {
            if (invoke(lookup,op) instanceof Invoke invoke && !invoke.isMappableIface()) {
                ptxIndent().convert(lookup,op).nl();
            } else {
                ptxIndent().convert(lookup,op).semicolon().nl();
            }
        });
    }

    public void ptxRegisterDecl() {
        for (PTXRegister.Type t : ordinalMap.keySet()) {
            ptxIndent().reg().sp();
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


    public PTXHATKernelBuilder convert(MethodHandles.Lookup lookup,Op op) {
        switch (op) {
            case JavaOp.FieldAccessOp.FieldLoadOp $ -> fieldLoad(lookup,$);
            case JavaOp.FieldAccessOp.FieldStoreOp $ -> fieldStore($);
            case JavaOp.BinaryOp $ -> binaryOperation($);
            case JavaOp.CompareOp $ -> compareOperation($);
            case JavaOp.ConvOp $ -> conv($);
            case CoreOp.ConstantOp $ -> constant($);
            case CoreOp.YieldOp $ -> javaYield($);
            case JavaOp.InvokeOp $ -> methodCall(invoke(lookup,$));
            case CoreOp.VarOp $ when ParamVar.of($) != null -> varFuncDeclaration($);
            case CoreOp.VarOp $ -> varDeclaration($);
            case CoreOp.ReturnOp $ -> ret($);
            case JavaOp.BreakOp $ -> javaBreak($);
            default -> { // Why are  these switch ops not just inlined above?
                switch (op){
                    case CoreOp.BranchOp $ -> branch($);
                    case CoreOp.ConditionalBranchOp $ -> condBranch($);
                    case JavaOp.NegOp $ -> neg($);
                    case PTXPtrOp $ -> ptxPtr($);
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
            addKeyword().s64().sp().regName(source).csp().reg(op.operands().get(0)).csp().reg(op.operands().get(1)).ptxNl();
        } else {
            source = getReg(op.operands().getFirst());
        }

        if (op.resultType.toString().equals("void")) {
            st().global().dot().regType(op.operands().getLast()).sp().address(source.name(), offset).csp().reg(op.operands().getLast());
        } else {
            ld().global().resultType(op.resultType(), true).sp().reg(op.result(), getResultType(op.resultType())).csp().address(source.name(), offset);
        }
    }

    public void fieldLoad(MethodHandles.Lookup lookup,JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {

        var fieldAccess = fieldAccess(lookup,fieldLoadOp);
        if (fieldAccess.named(Field.KC_X.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(fieldLoadOp.result());
            } else {
                mov().u32().sp().resultReg(fieldLoadOp, PTXRegister.Type.U32).csp().fieldReg(Field.KC_X);
            }
        } else if (fieldAccess.named(Field.KC_MAXX.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(fieldLoadOp.operands().getFirst());
            }
            ld().global().u32().sp().fieldReg(Field.KC_MAXX, fieldLoadOp.result()).csp()
                    .address(fieldToRegMap.get(Field.KC_ADDR).name(), 4);
        } else {
            ld().global().u32().sp().resultReg(fieldLoadOp, PTXRegister.Type.U64).csp().reg(fieldLoadOp.operands().getFirst());
        }
    }

    public void loadKcX(Value value) {
        cvta().to().global().size().sp().fieldReg(Field.KC_ADDR).csp()
                .reg(paramObjects.get(paramNames.indexOf(Field.KC_ADDR.toString())), addressType()).ptxNl();
        mov().u32().sp().fieldReg(Field.NTID_X).csp().percent().regName(Field.NTID_X.toString()).ptxNl();
        mov().u32().sp().fieldReg(Field.CTAID_X).csp().percent().regName(Field.CTAID_X.toString()).ptxNl();
        mov().u32().sp().fieldReg(Field.TID_X).csp().percent().regName(Field.TID_X.toString()).ptxNl();
        mad().lo().s32().sp().fieldReg(Field.KC_X, value).csp().fieldReg(Field.CTAID_X)
                .csp().fieldReg(Field.NTID_X).csp().fieldReg(Field.TID_X).ptxNl();
        st().global().u32().sp().address(fieldToRegMap.get(Field.KC_ADDR).name()).csp().fieldReg(Field.KC_X);
    }

    public void fieldStore(JavaOp.FieldAccessOp.FieldStoreOp op) {
        // TODO: fix
        st().global().u64().sp().resultReg(op, PTXRegister.Type.U64).csp().reg(op.operands().getFirst());
    }
    // this might be duplication of CodeBuilder symbol....
@Override public
PTXHATKernelBuilder symbol(Op op) {
        return switch (op) {
            case JavaOp.ModOp _ -> remKw();
            case JavaOp.MulOp _ -> mulKeword();
            case JavaOp.DivOp _ -> divKeyword();
            case JavaOp.AddOp _ -> addKeyword();
            case JavaOp.SubOp _ -> subKeyword();
            case JavaOp.LtOp _ -> ltKeyword();
            case JavaOp.GtOp _ -> gtKeyword();
            case JavaOp.LeOp _ -> le();
            case JavaOp.GeOp _ -> ge();
            case JavaOp.NeqOp _ -> neKeyword();
            case JavaOp.EqOp _ -> eqKeyword();
            case JavaOp.OrOp _ -> or();
            case JavaOp.AndOp _ -> and();
            case JavaOp.XorOp _ -> xor();
            case JavaOp.LshlOp _ -> shl();
            case JavaOp.AshrOp _, JavaOp.LshrOp _ -> shr();
            default -> throw new IllegalStateException("Unexpected value");
        };
    }

    public void binaryOperation(JavaOp.BinaryOp op) {
        symbol(op);
        if (getResultType(op.resultType()).getBasicType().equals(PTXRegister.Type.BasicType.FLOATING)
                && (op instanceof JavaOp.DivOp || op instanceof JavaOp.MulOp)) {
            rn();
        } else if (!getResultType(op.resultType()).getBasicType().equals(PTXRegister.Type.BasicType.FLOATING)
                && op instanceof JavaOp.MulOp) {
            lo();
        }
        resultType(op.resultType(), true).sp();
        resultReg(op, getResultType(op.resultType()));
        csp();
        reg(op.operands().getFirst());
        csp();
        reg(op.operands().get(1));
    }

    public void compareOperation(JavaOp.CompareOp op) {
        setp().dot();
        symbol(op).resultType(op.operands().getFirst().type(), true).sp();
        resultReg(op, PTXRegister.Type.PREDICATE);
        csp();
        reg(op.operands().getFirst());
        csp();
        reg(op.operands().get(1));
    }

    public void conv(JavaOp.ConvOp op) {
        if (op.resultType().equals(JavaType.LONG)) {
            if (isIndex(op)) {
                mulKeword().wide().s32().sp().resultReg(op, PTXRegister.Type.U64).csp()
                        .reg(op.operands().getFirst()).csp().intVal(4);
            } else {
                cvt().u64().dot().regType(op.operands().getFirst()).sp()
                        .resultReg(op, PTXRegister.Type.U64).csp().reg(op.operands().getFirst()).ptxNl();
            }
        } else if (op.resultType().equals(JavaType.FLOAT)) {
            cvt().rn().f32().dot().regType(op.operands().getFirst()).sp()
                    .resultReg(op, PTXRegister.Type.F32).csp().reg(op.operands().getFirst());
        } else if (op.resultType().equals(JavaType.DOUBLE)) {
            cvt();
            if (op.operands().getFirst().type().equals(JavaType.INT)) {
                rn();
            }
            f64().dot().regType(op.operands().getFirst()).sp()
                    .resultReg(op, PTXRegister.Type.F64).csp().reg(op.operands().getFirst());
        } else if (op.resultType().equals(JavaType.INT)) {
            cvt();
            if (op.operands().getFirst().type().equals(JavaType.DOUBLE) || op.operands().getFirst().type().equals(JavaType.FLOAT)) {
                rzi();
            } else {
                rn();
            }
            s32().dot().regType(op.operands().getFirst()).sp()
                    .resultReg(op, PTXRegister.Type.S32).csp().reg(op.operands().getFirst());
        } else {
            cvt().rn().s32().dot().regType(op.operands().getFirst()).sp()
                    .resultReg(op, PTXRegister.Type.S32).csp().reg(op.operands().getFirst());
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


    private boolean isIndex(JavaOp.ConvOp op) {
        for (Op.Result r : op.result().uses()) {
            if (r.op() instanceof PTXPtrOp) return true;
        }
        return false;
    }

    public void constant(CoreOp.ConstantOp op) {
        mov().resultType(op.resultType(), false).sp().resultReg(op, getResultType(op.resultType())).csp();
        if (op.resultType().toString().equals("float")) {
            if (op.value().toString().equals("0.0")) {
                floatVal("00000000");
            } else {
                floatVal(Integer.toHexString(Float.floatToIntBits(Float.parseFloat(op.value().toString()))).toUpperCase());
            }
        } else {
            constant(op.value().toString());
        }
    }

    public void javaYield(CoreOp.YieldOp op) {
        exit();
    }

    // S32Array and S32Array2D functions can be deleted after schema is done
    public void methodCall(Invoke invoke) {
       // Invoke invoke = Invoke.invokeOpHelper(MethodHandles.lookup(),invokeOp);
        switch (invoke.op().invokeReference().toString()) {
            // S32Array functions
            case "hat.buffer.S32Array::array(long)int" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                addKeyword().s64().sp().regName(temp).csp().reg(invoke.op().operands().getFirst()).csp().reg(invoke.op().operands().get(1)).ptxNl();
                ld().global().u32().sp().resultReg(invoke.op(), PTXRegister.Type.U32).csp().address(temp.name(), 4);
            }
            case "hat.buffer.S32Array::array(long, int)void" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                addKeyword().s64().sp().regName(temp).csp().reg(invoke.op().operands().getFirst()).csp().reg(invoke.op().operands().get(1)).ptxNl();
                st().global().u32().sp().address(temp.name(), 4).csp().reg(invoke.op().operands().get(2));
            }
            case "hat.buffer.S32Array::length()int" -> {
                ld().global().u32().sp().resultReg(invoke.op(), PTXRegister.Type.U32).csp().address(getReg(invoke.op().operands().getFirst()).name());
            }
            // S32Array2D functions
            case "hat.buffer.S32Array2D::array(long, int)void" -> {
                PTXRegister temp = new PTXRegister(incrOrdinal(addressType()), addressType());
                addKeyword().s64().sp().regName(temp).csp().reg(invoke.op().operands().getFirst()).csp().reg(invoke.op().operands().get(1)).ptxNl();
                st().global().u32().sp().address(temp.name(), 8).csp().reg(invoke.op().operands().get(2));
            }
            case "hat.buffer.S32Array2D::width()int" -> {
                ld().global().u32().sp().resultReg(invoke.op(), PTXRegister.Type.U32).csp().address(getReg(invoke.op().operands().getFirst()).name());
            }
            case "hat.buffer.S32Array2D::height()int" -> {
                ld().global().u32().sp().resultReg(invoke.op(), PTXRegister.Type.U32).csp().address(getReg(invoke.op().operands().getFirst()).name(), 4);
            }
            // Java Math function
            case "java.lang.Math::sqrt(double)double" -> {
                sqrt().rn().f64().sp().resultReg(invoke.op(), PTXRegister.Type.F64).csp().reg(invoke.op().operands().getFirst()).semicolon();
            }
            default -> {
                obrace().nl().ptxIndent();
                for (int i = 0; i < invoke.op().operands().size(); i++) {
                    dot().param().sp().paramType(invoke.op().operands().get(i).type()).sp().param().intVal(i).ptxNl();
                    st().dot().param().paramType(invoke.op().operands().get(i).type()).sp().osbrace().param().intVal(i).csbrace().csp().reg(invoke.op().operands().get(i)).ptxNl();
                }
                dot().param().sp().paramType(invoke.op().resultType()).sp().retVal().ptxNl();
                call().uni().sp().oparen().retVal().cparen().csp().id(invoke.name()).csp();
                final int[] counter = {0};
                paren(_ ->
                        commaSpaceSeparated(
                                invoke.op().operands(),
                                _ -> param().intVal(counter[0]++)
                        )
                ).ptxNl();
                ld().dot().param().paramType(invoke.op().resultType()).sp().resultReg(invoke.op(), getResultType(invoke.op().resultType())).csp().osbrace().retVal().csbrace();
                ptxNl().cbrace();
            }
        }
    }

    public void varDeclaration(CoreOp.VarOp op) {
        ld().dot().param().resultType(op.resultType(), false).sp().resultReg(op, addressType()).csp().reg(op.operands().getFirst());
    }

    public void varFuncDeclaration(CoreOp.VarOp op) {
        ld().dot().param().resultType(op.resultType(), false).sp().resultReg(op, addressType()).csp().reg(op.operands().getFirst());
    }

    public void ret(CoreOp.ReturnOp op) {
        if (!op.operands().isEmpty()) {
            st().dot().param();
            if (returnReg.type().equals(PTXRegister.Type.U32)) {
                b32();
            } else if (returnReg.type().equals(PTXRegister.Type.U64)) {
                b64();
            } else {
                dot().regType(returnReg.type());
            }
            sp().osbrace().regName(returnReg).csbrace().csp().reg(op.operands().getFirst()).ptxNl();
        }
        ret();
    }

    public void javaBreak(JavaOp.BreakOp op) {
        brkpt();
    }

    public void branch(CoreOp.BranchOp op) {
        loadBlockParams(op.successors().getFirst());
        bra().sp().block(op.successors().getFirst().targetBlock());
    }

    public void condBranch(CoreOp.ConditionalBranchOp op) {
        loadBlockParams(op.successors().getFirst());
        loadBlockParams(op.successors().getLast());
        at().reg(op.operands().getFirst()).sp()
                .bra().sp().block(op.successors().getFirst().targetBlock()).ptxNl();
        bra().sp().block(op.successors().getLast().targetBlock());
    }

    public void neg(JavaOp.NegOp op) {
        neg().resultType(op.resultType(), true).sp().reg(op.result(), getResultType(op.resultType())).csp().reg(op.operands().getFirst());
    }

    /*
     * Helper functions for printing blocks and variables
     */

    public void loadBlockParams(Block.Reference block) {
        for (int i = 0; i < block.arguments().size(); i++) {
            Block.Parameter p = block.targetBlock().parameters().get(i);
            mov().resultType(p.type(), false).sp().reg(p, getResultType(p.type()))
                    .csp().reg(block.arguments().get(i)).ptxNl();
        }
    }

    public PTXHATKernelBuilder block(Block block) {
        return type("block_").intVal(block.index());
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

    public PTXHATKernelBuilder resultReg(Op op, PTXRegister.Type type) {
        return id(addReg(op.result(), type));
    }

    public PTXHATKernelBuilder reg(Value val, PTXRegister.Type type) {
        if (varToRegMap.containsKey(val)) {
            return regName(getReg(val));
        } else {
            return id(addReg(val, type));
        }
    }

    public PTXHATKernelBuilder reg(Value val) {
        return regName(getReg(val));
    }

    public PTXRegister getReg(Value val) {
        if (varToRegMap.get(val) == null && val instanceof Op.Result result && result.op() instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
            return fieldToRegMap.get(getFieldObj(fieldLoadOp.fieldReference().name()));
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

    public PTXHATKernelBuilder resultType(CodeType type, boolean signedResult) {
        PTXRegister.Type res = getResultType(type);
        if (signedResult && (res == PTXRegister.Type.U32)) return s32();
        return dot().type(getResultType(type).getName());
    }

    public PTXHATKernelBuilder paramType(CodeType type) {
        PTXRegister.Type res = getResultType(type);
        if (res == PTXRegister.Type.U32) return b32();
        if (res == PTXRegister.Type.U64) return b64();
        return dot().type(getResultType(type).getName());
    }

    public PTXRegister.Type getResultType(CodeType type) {
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
        return osbrace().constant(address).csbrace();
    }

    public PTXHATKernelBuilder address(String address, int offset) {
        osbrace().constant(address);
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


    public PTXHATKernelBuilder param() {
        return keyword("param");
    }

    public PTXHATKernelBuilder global() {
        return dot().keyword("global");
    }

    public PTXHATKernelBuilder rn() {
        return dot().keyword("rn");
    }

    public PTXHATKernelBuilder rm() {
        return dot().keyword("rm");
    }

    public PTXHATKernelBuilder rzi() {
        return dot().keyword("rzi");
    }

    public PTXHATKernelBuilder to() {
        return dot().keyword("to");
    }

    public PTXHATKernelBuilder lo() {
        return dot().keyword("lo");
    }

    public PTXHATKernelBuilder wide() {
        return dot().keyword("wide");
    }

    public PTXHATKernelBuilder uni() {
        return dot().keyword("uni");
    }

    public PTXHATKernelBuilder sat() {
        return dot().keyword("sat");
    }

    public PTXHATKernelBuilder ftz() {
        return dot().keyword("ftz");
    }

    public PTXHATKernelBuilder approx() {
        return dot().keyword("approx");
    }

    public PTXHATKernelBuilder mov() {
        return keyword("mov");
    }

    public PTXHATKernelBuilder setp() {
        return keyword("setp");
    }

    public PTXHATKernelBuilder selp() {
        return keyword("selp");
    }

    public PTXHATKernelBuilder ld() {
        return keyword("ld");
    }

    public PTXHATKernelBuilder st() {
        return keyword("st");
    }

    public PTXHATKernelBuilder cvt() {
        return keyword("cvt");
    }

    public PTXHATKernelBuilder bra() {
        return keyword("bra");
    }

    public PTXHATKernelBuilder ret() {
        return keyword("ret");
    }

    public PTXHATKernelBuilder remKw() {
        return keyword("rem");
    }

    public PTXHATKernelBuilder mulKeword() {
        return keyword("mul");
    }

    public PTXHATKernelBuilder divKeyword() {
        return keyword("div");
    }

    public PTXHATKernelBuilder rcp() {
        return keyword("rcp");
    }

    public PTXHATKernelBuilder addKeyword() {
        return keyword("add");
    }

    public PTXHATKernelBuilder subKeyword() {
        return keyword("sub");
    }

    public PTXHATKernelBuilder ltKeyword() {
        return keyword("lt");
    }

    public PTXHATKernelBuilder gtKeyword() {
        return keyword("gt");
    }

    public PTXHATKernelBuilder le() {
        return keyword("le");
    }

    public PTXHATKernelBuilder ge() {
        return keyword("ge");
    }

    public PTXHATKernelBuilder geu() {
        return keyword("geu");
    }

    public PTXHATKernelBuilder neKeyword() {
        return keyword("ne");
    }

    public PTXHATKernelBuilder eqKeyword() {
        return keyword("eq");
    }

    public PTXHATKernelBuilder xor() {
        return keyword("xor");
    }

    public PTXHATKernelBuilder or() {
        return keyword("or");
    }

    public PTXHATKernelBuilder and() {
        return keyword("and");
    }

    public PTXHATKernelBuilder cvta() {
        return keyword("cvta");
    }

    public PTXHATKernelBuilder mad() {
        return keyword("mad");
    }

    public PTXHATKernelBuilder fma() {
        return keyword("fma");
    }

    public PTXHATKernelBuilder sqrt() {
        return keyword("sqrt");
    }

    public PTXHATKernelBuilder abs() {
        return keyword("abs");
    }

    public PTXHATKernelBuilder ex2() {
        return keyword("ex2");
    }

    public PTXHATKernelBuilder shl() {
        return keyword("shl");
    }

    public PTXHATKernelBuilder shr() {
        return keyword("shr");
    }

    public PTXHATKernelBuilder neg() {
        return keyword("neg");
    }

    public PTXHATKernelBuilder call() {
        return keyword("call");
    }

    public PTXHATKernelBuilder exit() {
        return keyword("exit");
    }

    public PTXHATKernelBuilder brkpt() {
        return keyword("brkpt");
    }

    public PTXHATKernelBuilder ptxIndent() {
        return sp().sp().sp().sp();
    }

    public PTXHATKernelBuilder u32() {
        return dot().type(PTXRegister.Type.U32.getName());
    }

    public PTXHATKernelBuilder s32() {
        return dot().type(PTXRegister.Type.S32.getName());
    }

    public PTXHATKernelBuilder f32() {
        return dot().type(PTXRegister.Type.F32.getName());
    }

    public PTXHATKernelBuilder b32() {
        return dot().type(PTXRegister.Type.B32.getName());
    }

    public PTXHATKernelBuilder u64() {
        return dot().type(PTXRegister.Type.U64.getName());
    }

    public PTXHATKernelBuilder s64() {
        return dot().type(PTXRegister.Type.S64.getName());
    }

    public PTXHATKernelBuilder f64() {
        return dot().type(PTXRegister.Type.F64.getName());
    }

    public PTXHATKernelBuilder b64() {
        return dot().type(PTXRegister.Type.B64.getName());
    }

    public PTXHATKernelBuilder version() {
        return dot().keyword("version");
    }

    public PTXHATKernelBuilder target() {
        return dot().keyword("target");
    }

    public PTXHATKernelBuilder addressSize() {
        return dot().keyword("address_size");
    }

    public PTXHATKernelBuilder major(int major) {
        return intVal(major);
    }

    public PTXHATKernelBuilder minor(int minor) {
        return intVal(minor);
    }

    public PTXHATKernelBuilder target(String target) {
        return keyword(target);
    }

    public PTXHATKernelBuilder size(int addressSize) {
        return intVal(addressSize);
    }



    public PTXHATKernelBuilder visible() {
        return dot().keyword("visible");
    }

    public PTXHATKernelBuilder entry() {
        return dot().keyword("entry");
    }

    public PTXHATKernelBuilder func() {
        return dot().keyword("func");
    }

    public PTXHATKernelBuilder oabrace() {
        return symbol("<");
    }

    public PTXHATKernelBuilder cabrace() {
        return symbol(">");
    }

    public PTXHATKernelBuilder regName(PTXRegister reg) {
        return id(reg.name());
    }

    public PTXHATKernelBuilder regName(String regName) {
        return id(regName);
    }

    public PTXHATKernelBuilder regType(Value val) {
        return keyword(getReg(val).type().getName());
    }

    public PTXHATKernelBuilder regType(PTXRegister.Type t) {
        return keyword(t.getName());
    }

    public PTXHATKernelBuilder regTypePrefix(PTXRegister.Type t) {
        return keyword(t.getRegPrefix());
    }

    public PTXHATKernelBuilder reg() {
        return dot().keyword("reg");
    }

    public PTXHATKernelBuilder retVal() {
        return keyword("retval");
    }

    public PTXHATKernelBuilder intVal(int i) {
        return constant(String.valueOf(i));
    }

    public PTXHATKernelBuilder floatVal(String s) {
        return constant("0f").constant(s);
    }

    public PTXHATKernelBuilder doubleVal(String s) {
        return constant("0d").constant(s);
    }
}