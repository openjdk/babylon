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
import java.lang.reflect.code.op.ExtendedOp;
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
        append(".version ").append(String.valueOf(major)).dot().append(String.valueOf(minor)).nl();
        append(".target ").append(target).nl();
        append(".address_size ").append(String.valueOf(addressSize));
        this.addressSize = addressSize;
    }

    public void functionHeader(String funcName, boolean entry, TypeElement yieldType) {
        if (entry) {
            append(".visible .entry ");
        } else {
            append(".func ");
        }
        if (!yieldType.toString().equals("void")) {
            returnReg = new PTXRegister(getOrdinal(resultType(yieldType)), resultType(yieldType));
            returnReg.name("%retReg");
            oparen().param().space().printParamType(yieldType);
            space().append(returnReg.name()).cparen().space();
        }
        append(funcName);
    }

    public PTXCodeBuilder parameters(List<FuncOpWrapper.ParamTable.Info> infoList) {
        paren(_ -> nl().commaNlSeparated(infoList, (info) -> {
            ptxIndent().param().space().printParamType(info.javaType);
            space().append(info.varOp.varName());
            params.add(info.varOp.varName());
        }).nl()).nl();
        return this;
    }

    public void blockBody(Block block, Stream<OpWrapper<?>> ops) {
        if (block.index() == 0) {
            for (Block.Parameter p : block.parameters()) {
                ptxIndent().ld().param();
                printResultType(p.type(), false).ptxIndent().space();
                printAndAddVar(p, resultType(p.type())).commaSpace().osbrace().append(params.get(p.index())).csbrace().semicolon().nl();
                paramMap.putIfAbsent(params.get(p.index()), p);
            }
        }
        nl();
        printBlock(block);
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
            ptxIndent().append(".reg").space();
            if (t.equals(PTXRegister.Type.U32)) {
                b32();
            } else if (t.equals(PTXRegister.Type.U64)) {
                b64();
            } else {
                dot().append(t.toString());
            }
            ptxIndent().append(t.getRegPrefix()).lt().append(String.valueOf(ordinalMap.get(t))).gt().semicolon().nl();
        }
        nl();
    }

    public void functionPrologue() {
        append("{").nl();
    }

    public void functionEpilogue() {
        append("}");
    }

    public PTXCodeBuilder convert(OpWrapper<?> wrappedOp) {
        switch (wrappedOp) {
            case VarLoadOpWrapper op -> varLoad(op);
            case VarStoreOpWrapper op -> varStore(op);
            case FieldLoadOpWrapper op -> fieldLoad(op);
            case FieldStoreOpWrapper op -> fieldStore(op);
            case BinaryArithmeticOrLogicOperation op -> binaryOperation(op);
            case BinaryTestOpWrapper op -> binaryTest(op);
            case ConvOpWrapper op -> conv(op);
            case ConstantOpWrapper op -> constant(op);
            case YieldOpWrapper op -> javaYield(op);
            case FuncCallOpWrapper op -> funcCall(op);
            case InvokeOpWrapper op -> methodCall(op);
            case VarDeclarationOpWrapper op -> varDeclaration(op);
            case VarFuncDeclarationOpWrapper op -> varFuncDeclaration(op);
            case ReturnOpWrapper op -> ret(op);
            case JavaBreakOpWrapper op -> javaBreak(op);
            default -> {
                switch (wrappedOp.op()){
                    case CoreOp.BranchOp op -> branch(op);
                    case CoreOp.ConditionalBranchOp op -> condBranch(op);
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
            // append("mul.wide").s32().space().printAndAddVar(op.result(), PTXRegister.Type.U64).commaSpace()
            //         .printVar(op.operands().getFirst()).commaSpace().append("4").ptxNl();
            source = new PTXRegister(incrOrdinal(addrType()), addrType());
            append("add").s64().space().append(source.name()).commaSpace().printVar(op.operands().get(0)).commaSpace().printVar(op.operands().get(1)).ptxNl();
        } else {
            source = getVar(op.operands().getFirst());
        }

        if (op.resultType.toString().equals("void")) {
            st().global().u32().space().address(source.name(), offset).commaSpace().printVar(op.operands().get(2));
        } else {
            ld().global().u32().space().printAndAddVar(op.result(), PTXRegister.Type.U32).commaSpace().address(source.name(), offset);
        }
    }

    public void varLoad(VarLoadOpWrapper op) {
        ld().printResultType(op.resultType(), false).space().printResult(op, addrType()).commaSpace().printVar(op.operandNAsValue(0));
    }

    public void varStore(VarStoreOpWrapper op) {
        st().printResultType(op.resultType(), false).space().printResult(op, addrType()).commaSpace().printVar(op.operandNAsValue(0));
    }

    public void fieldLoad(FieldLoadOpWrapper op) {
        if (op.fieldName().equals(Field.KC_X.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(op.result());
            } else {
                mov().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().printField(Field.KC_X);
            }
        } else if (op.fieldName().equals(Field.KC_MAXX.toString())) {
            if (!fieldToRegMap.containsKey(Field.KC_X)) {
                loadKcX(op.operandNAsValue(0));
            }
            ld().global().u32().space().printFieldAndVal(Field.KC_MAXX, op.result()).commaSpace()
                    .address(fieldToRegMap.get(Field.KC_ADDR).name(), 4);
        } else {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U64).commaSpace().printVar(op.operandNAsValue(0));
//            ld().global().u32().space().printResult(op, PTXRegister.Type.U64).commaSpace().printFieldAndVal(op.fieldName(), op.result());
        }
    }

    public void loadKcX(Value value) {
        append("cvta.to").global().addrSize().space().printField(Field.KC_ADDR).commaSpace()
                .printAndAddVar(paramMap.get("kc"), addrType()).ptxNl();
        mov().u32().space().printField(Field.NTID_X).commaSpace().append("%ntid.x").ptxNl();
        mov().u32().space().printField(Field.CTAID_X).commaSpace().append("%ctaid.x").ptxNl();
        mov().u32().space().printField(Field.TID_X).commaSpace().append("%tid.x").ptxNl();
        append("mad.lo").s32().space().printFieldAndVal(Field.KC_X, value).commaSpace().printField(Field.CTAID_X)
                .commaSpace().printField(Field.NTID_X).commaSpace().printField(Field.TID_X).ptxNl();
        st().global().u32().space().address(fieldToRegMap.get(Field.KC_ADDR).name()).commaSpace().printField(Field.KC_X);
    }

    public void fieldStore(FieldStoreOpWrapper op) {
        // TODO: fix
        st().global().u64().space().printResult(op, PTXRegister.Type.U64).commaSpace().printVar(op.operandNAsValue(0));
    }

    PTXCodeBuilder symbol(Op op) {
        return switch (op) {
            case CoreOp.ModOp _ -> append("rem");
            case CoreOp.MulOp _ -> append("mul");
            case CoreOp.DivOp _ -> append("div");
            case CoreOp.AddOp _ -> append("add");
            case CoreOp.SubOp _ -> append("sub");
            case CoreOp.LtOp _ -> append("lt");
            case CoreOp.GtOp _ -> append("gt");
            case CoreOp.LeOp _ -> append("le");
            case CoreOp.GeOp _ -> append("ge");
            case CoreOp.NeqOp _ -> append("ne");
            case CoreOp.EqOp _ -> append("eq");
            case CoreOp.XorOp _ -> append("xor");
            case ExtendedOp.JavaConditionalAndOp _ -> append("&&");
            case ExtendedOp.JavaConditionalOrOp _ -> append("||");
            default -> throw new IllegalStateException("Unexpected value");
        };
    }

    public void binaryOperation(BinaryArithmeticOrLogicOperation op) {
        symbol(op.op());
        if (op.resultType().toString().equals("float") && op.op() instanceof CoreOp.DivOp) dot().append("rn");
        if (!op.resultType().toString().equals("float") && op.op() instanceof CoreOp.MulOp) dot().append("lo");
        printResultType(op.resultType(), true).space();
        printResult(op, resultType(op.resultType()));
        commaSpace();
        printVar(op.operandNAsValue(0));
        commaSpace();
        printVar(op.operandNAsValue(1));
    }

    public void binaryTest(BinaryTestOpWrapper op) {
        append("setp").dot();
        symbol(op.op()).printResultType(op.operandNAsValue(0).type(), true).space();
        printResult(op, PTXRegister.Type.PREDICATE);
        commaSpace();
        printVar(op.operandNAsValue(0));
        commaSpace();
        printVar(op.operandNAsValue(1));
    }

    //TODO: fix?? (i think this is multiplying the idx by 4)
    public void conv(ConvOpWrapper op) {
        if (op.resultJavaType().equals(JavaType.LONG)) {
            // PTXRegister temp = new PTXRegister(incrOrdinal(addrType()), addrType());
            // append("cvt.rn").u64().dot().append(getVar(op.operandNAsValue(0)).type().toString()).space()
            //         .printResult(op, PTXRegister.Type.U64).commaSpace().printVar(op.operandNAsValue(0)).ptxNl();
            // append("mul.wide").s32().space().printResult(op, PTXRegister.Type.U64).commaSpace()
            //         .printVar(op.operandNAsValue(0)).commaSpace().append("4");
            append("mul.wide").s32().space().printResult(op, PTXRegister.Type.U64).commaSpace()
                    .printVar(op.operandNAsValue(0)).commaSpace().append("4");
        } else if (op.resultJavaType().equals(JavaType.FLOAT)) {
            append("cvt.rn").f32().dot().append(getVar(op.operandNAsValue(0)).type().toString()).space()
                    .printResult(op, PTXRegister.Type.F32).commaSpace().printVar(op.operandNAsValue(0));
        } else if (op.resultJavaType().equals(JavaType.DOUBLE)) {
            append("cvt.rn").f64().dot().append(getVar(op.operandNAsValue(0)).type().toString()).space()
                    .printResult(op, PTXRegister.Type.F64).commaSpace().printVar(op.operandNAsValue(0));
        } else if (op.resultJavaType().equals(JavaType.INT)) {
            append("cvt.rn").s32().dot().append(getVar(op.operandNAsValue(0)).type().toString()).space()
                    .printResult(op, PTXRegister.Type.S32).commaSpace().printVar(op.operandNAsValue(0));
        } else {
            append("cvt.rn").s32().dot().append(getVar(op.operandNAsValue(0)).type().toString()).space()
                    .printResult(op, PTXRegister.Type.S32).commaSpace().printVar(op.operandNAsValue(0));
        }
    }

    public void constant(ConstantOpWrapper op) {
        mov().printResultType(op.resultType(), false).space().printResult(op, resultType(op.resultType())).commaSpace();
        if (op.resultType().toString().equals("float")) {
            append("0f");
            append(Integer.toHexString(Float.floatToIntBits(Float.parseFloat(op.op().value().toString()))).toUpperCase());
            if (op.op().value().toString().equals("0.0")) append("0000000");
        } else {
            append(op.op().value().toString());
        }
    }

    public void javaYield(YieldOpWrapper op) {
        append("exit");
    }

    public void funcCall(FuncCallOpWrapper op) {
        // TODO: fix????
        append(op.toString());
    }

    public void methodCall(InvokeOpWrapper op) {
        if (op.methodRef().toString().equals("hat.buffer.S32Array::array(long)int")) {
            PTXRegister temp = new PTXRegister(incrOrdinal(addrType()), addrType());
            append("add").s64().space().append(temp.name()).commaSpace().printVar(op.operandNAsValue(0)).commaSpace().printVar(op.operandNAsValue(1)).ptxNl();
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(temp.name(), 4);
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array::array(long, int)void")) {
            PTXRegister temp = new PTXRegister(incrOrdinal(addrType()), addrType());
            append("add").s64().space().append(temp.name()).commaSpace().printVar(op.operandNAsValue(0)).commaSpace().printVar(op.operandNAsValue(1)).ptxNl();
            st().global().u32().space().address(temp.name(), 4).commaSpace().printVar(op.operandNAsValue(2));
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array::length()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(getVar(op.operandNAsValue(0)).name());
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::array(long, int)void")) {
            PTXRegister temp = new PTXRegister(incrOrdinal(addrType()), addrType());
            append("add").s64().space().append(temp.name()).commaSpace().printVar(op.operandNAsValue(0)).commaSpace().printVar(op.operandNAsValue(1)).ptxNl();
            st().global().u32().space().address(temp.name(), 8).commaSpace().printVar(op.operandNAsValue(2));
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::width()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(getVar(op.operandNAsValue(0)).name());
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::height()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(getVar(op.operandNAsValue(0)).name(), 4);
        } else {
            for (int i = 0; i < op.operands().size(); i++) {
                param().space().printParamType(op.operandNAsValue(i).type()).space().append("param").append(String.valueOf(i)).ptxNl();
                st().param().printParamType(op.operandNAsValue(i).type()).space().osbrace().append("param").append(String.valueOf(i)).csbrace().commaSpace().printVar(op.operandNAsValue(i)).ptxNl();
            }
            param().space().printParamType(op.resultType()).space().append("retval").ptxNl();
            append("call.uni").space().oparen().append("retval").cparen().commaSpace().append(op.method().getName()).commaSpace();
            final int[] counter = {0};
            paren(_ -> commaSeparated(op.operands(), _ -> append("param").append(String.valueOf(counter[0]++)))).ptxNl();
            ld().param().printParamType(op.resultType()).space().printResult(op, resultType(op.resultType())).commaSpace().osbrace().append("retval").csbrace();
        }
    }

    public void varDeclaration(VarDeclarationOpWrapper op) {
        ld().param().printResultType(op.resultType(), false).space().printResult(op, addrType()).commaSpace().printVar(op.operandNAsValue(0));
    }

    public void varFuncDeclaration(VarFuncDeclarationOpWrapper op) {
        ld().param().printResultType(op.resultType(), false).space().printResult(op, addrType()).commaSpace().printVar(op.operandNAsValue(0));
    }

    public void ret(ReturnOpWrapper op) {
        if (op.hasOperands()) {
            st().param();
            if (returnReg.type().equals(PTXRegister.Type.U32)) {
                b32();
            } else if (returnReg.type().equals(PTXRegister.Type.U64)) {
                b64();
            } else {
                dot().append(returnReg.type().toString());
            }
            space().osbrace().append(returnReg.name()).csbrace().commaSpace().printVar(op.operandNAsValue(0)).ptxNl();
        }
        append("ret");
    }

    public void javaBreak(JavaBreakOpWrapper op) {
        append("brkpt");
    }

    public void branch(CoreOp.BranchOp op) {
        loadBlockParams(op.successors().getFirst());
        append("bra").space().printBlock(op.successors().getFirst().targetBlock());
    }

    public void condBranch(CoreOp.ConditionalBranchOp op) {
        loadBlockParams(op.successors().getFirst());
        loadBlockParams(op.successors().getLast());
        append("@").printVar(op.operands().getFirst()).space()
                .append("bra").space().printBlock(op.successors().getFirst().targetBlock()).ptxNl();
        append("bra").space().printBlock(op.successors().getLast().targetBlock());
    }

    /*
     * Helper functions for printing blocks and variables
     */

    public void loadBlockParams(Block.Reference block) {
        for (int i = 0; i < block.arguments().size(); i++) {
            Block.Parameter p = block.targetBlock().parameters().get(i);
            mov().printResultType(p.type(), false).space().printAndAddVar(p, resultType(p.type()))
                    .commaSpace().printVar(block.arguments().get(i)).ptxNl();
        }
    }

    public PTXCodeBuilder printBlock(Block block) {
        return append("block_").append(String.valueOf(block.index()));
    }

    public PTXCodeBuilder printField(String ref) {
        return printField(getFieldObj(ref));
    }

    public PTXCodeBuilder printField(Field ref) {
        if (fieldToRegMap.containsKey(ref)) {
            return append(fieldToRegMap.get(ref).name());
        }
        if (ref.isDestination()) {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(addrType()), addrType()));
        } else {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32));
        }
        return append(fieldToRegMap.get(ref).name());
    }

    public PTXCodeBuilder printFieldAndVal(Field ref, Value value) {
        if (fieldToRegMap.containsKey(ref)) {
            return append(fieldToRegMap.get(ref).name());
        }
        if (ref.isDestination()) {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(getOrdinal(addrType()), addrType()));
            return printAndAddVar(value, addrType());
        } else {
            fieldToRegMap.putIfAbsent(ref, new PTXRegister(getOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32));
            return printAndAddVar(value, PTXRegister.Type.U32);
        }
    }

    public Field getFieldObj(String fieldName) {
        for (Field f : fieldToRegMap.keySet()) {
            if (f.toString().equals(fieldName)) return f;
        }
        throw new IllegalStateException("no existing field");
    }

    public PTXCodeBuilder printResult(OpWrapper<?> opWrapper, PTXRegister.Type type) {
        return append(addVar(opWrapper.result(), type));
    }

    public PTXCodeBuilder printAndAddVar(Value val, PTXRegister.Type type) {
        if (varToRegMap.containsKey(val)) {
            return append(getVar(val).name());
        } else {
            return append(addVar(val, type));
        }
    }

    public PTXCodeBuilder printVar(Value val) {
        return append(getVar(val).name());
    }

    public PTXRegister getVar(Value val) {
        if (varToRegMap.get(val) == null && val instanceof Op.Result result && result.op() instanceof CoreOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
            return fieldToRegMap.get(getFieldObj(fieldLoadOp.fieldDescriptor().name()));
        }
        if (varToRegMap.containsKey(val)) {
            return varToRegMap.get(val);
        } else {
            throw new IllegalStateException("var to reg mapping doesn't exist");
        }
    }

    public String addVar(Value val, PTXRegister.Type type) {
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

    public PTXCodeBuilder addrSize() {
        return (addressSize == 32) ? u32() : u64();
    }

    public PTXRegister.Type addrType() {
        return (addressSize == 32) ? PTXRegister.Type.U32 : PTXRegister.Type.U64;
    }

    public PTXCodeBuilder printResultType(TypeElement type, boolean signedResult) {
        PTXRegister.Type res = resultType(type);
        if (signedResult && (res == PTXRegister.Type.U32)) return s32();
        return dot().append(resultType(type).toString());
    }

    public PTXCodeBuilder printParamType(TypeElement type) {
        PTXRegister.Type res = resultType(type);
        if (res == PTXRegister.Type.U32) return b32();
        if (res == PTXRegister.Type.U64) return b64();
        return dot().append(resultType(type).toString());
    }

    public PTXRegister.Type resultType(TypeElement type) {
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
        return append(String.valueOf(offset)).csbrace();
    }

    public void ptxNl() {
        semicolon().nl().ptxIndent();
    }

    public PTXCodeBuilder commaSpace() {
        return comma().space();
    }

    public PTXCodeBuilder param() {
        return dot().append("param");
    }

    public PTXCodeBuilder global() {
        return dot().append("global");
    }

    public PTXCodeBuilder mov() {
        return append("mov");
    }

    public PTXCodeBuilder ld() {
        return append("ld");
    }

    public PTXCodeBuilder st() {
        return append("st");
    }

    public PTXCodeBuilder ptxIndent() {
        return append("    ");
    }

    public PTXCodeBuilder u32() {
        return dot().append(PTXRegister.Type.U32.toString());
    }

    public PTXCodeBuilder s32() {
        return dot().append(PTXRegister.Type.S32.toString());
    }

    public PTXCodeBuilder f32() {
        return dot().append(PTXRegister.Type.F32.toString());
    }

    public PTXCodeBuilder b32() {
        return dot().append(PTXRegister.Type.B32.toString());
    }

    public PTXCodeBuilder u64() {
        return dot().append(PTXRegister.Type.U64.toString());
    }

    public PTXCodeBuilder s64() {
        return dot().append(PTXRegister.Type.S64.toString());
    }

    public PTXCodeBuilder f64() {
        return dot().append(PTXRegister.Type.F64.toString());
    }

    public PTXCodeBuilder b64() {
        return dot().append(PTXRegister.Type.B64.toString());
    }
}