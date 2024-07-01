package hat.backend;

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
    // can use paramMap to index into varToRegMap
    Map<String, Block.Parameter> paramMap;
    // field ref to reg map because we don't have values to map the field refs to :sob:
    Map<PTXCodeBuilder.FieldRef, PTXRegister> fieldRefToRegMap;

    HashMap<PTXRegister.Type, Integer> ordinalMap;

    private int addressSize;

    public enum FieldRef {
        NTID_X ("ntid.x", false),
        CTAID_X ("ctaid.x", false),
        TID_X ("tid.x", false),
        KC_X ("x", false),
        KC_X_ADDR ("[kc.x]", true),
        KC_MAXX ("maxX", false);

        private final String name;
        private final boolean destination;

        FieldRef(String name, boolean destination) {
            this.name = name;
            this.destination = destination;
        }
        public String toString() {
            return this.name;
        }
        public boolean isDestination() {return this.destination;}
    }

    public PTXCodeBuilder() {
        varToRegMap = new HashMap<>();
        params = new ArrayList<>();
        fieldRefToRegMap = new HashMap<>();
        paramMap = new HashMap<>();
        ordinalMap = new HashMap<>();
        addressSize = 32;
    }

    // check addr size
    public void ptxHeader(int major, int minor, String target, int addressSize) {
        append(".version ").append(String.valueOf(major)).dot().append(String.valueOf(minor)).nl();
        append(".target ").append(target).nl();
        append(".address_size ").append(String.valueOf(addressSize)).nl();
        nl();
        this.addressSize = addressSize;
    }

    public void functionHeader(String funcName) {
        append(".visible .entry ").append(funcName);
    }

    public PTXCodeBuilder parameters(List<FuncOpWrapper.ParamTable.Info> infoList) {
        paren(_ -> nl().commaNlSeparated(infoList, (info) -> {
            ptxIndent().param().space().dot().addrSize().space().append(info.varOp.varName());
            params.add(info.varOp.varName());
        }).nl()).nl();
        return this;
    }

    public void blockBody(Block block, Stream<OpWrapper<?>> ops) {
        if (block.index() == 0) {
            for (Block.Parameter p : block.parameters()) {
                ptxIndent().ld().param();
                dot().printResultType(p.type()).ptxIndent().space();
                printAndAddVar(p, resultType(p.type())).commaSpace().osbrace().append(params.get(p.index())).csbrace().semicolon().nl();
                paramMap.putIfAbsent(params.get(p.index()), p);
            }
        }
        printBlock(block);
//        if (!block.parameters().isEmpty() && block.index() > 0) {
//            space().oparen();
//            commaSeparated(block.parameters(), (p) -> printAndAddVar(p, addrType()));
//            cparen();
//        }
        colon().nl();
        ops.forEach(op -> ptxIndent().convert(op).semicolon().nl());
    }

    public void ptxRegisterDecl() {
        for (PTXRegister.Type t : ordinalMap.keySet()) {
            if (t.equals(PTXRegister.Type.U32)) {
                append(".reg").space().dot().append(PTXRegister.Type.B32.toString()).ptxIndent().append(t.getRegPrefix()).lt().append(String.valueOf(ordinalMap.get(t))).gt().semicolon().nl();
            } else if (t.equals(PTXRegister.Type.U64)) {
                append(".reg").space().dot().append(PTXRegister.Type.B64.toString()).ptxIndent().append(t.getRegPrefix()).lt().append(String.valueOf(ordinalMap.get(t))).gt().semicolon().nl();
            } else {
                append(".reg").space().dot().append(t.toString()).ptxIndent().append(t.getRegPrefix()).lt().append(String.valueOf(ordinalMap.get(t))).gt().semicolon().nl();
            }
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
        //TODO: check for param
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
            case LogicalOpWrapper op -> logical(op);
            case InvokeOpWrapper op -> methodCall(op);
            case TernaryOpWrapper op -> ternary(op);
            case VarDeclarationOpWrapper op -> varDeclaration(op);
            case VarFuncDeclarationOpWrapper op -> varFuncDeclaration(op);
            case TupleOpWrapper op -> tuple(op);
            case ReturnOpWrapper op -> ret(op);
            case JavaLabeledOpWrapper op -> javaLabeled(op);
            case JavaBreakOpWrapper op -> javaBreak(op);
            case JavaContinueOpWrapper op -> javaContinue(op);
            default -> {
                switch (wrappedOp.op()){
                    case CoreOp.BranchOp op -> branch(op);
                    case CoreOp.ConditionalBranchOp op -> condBranch(op);
                    default -> throw new IllegalStateException("oops");
                }
            }
        }
        return this;
    }

    public void varLoad(VarLoadOpWrapper op) {
        append(op.toString());
    }

    public void varStore(VarStoreOpWrapper op) {
        append(op.toString());
    }

    public void fieldLoad(FieldLoadOpWrapper op) {
        if (op.fieldName().equals(FieldRef.KC_X.toString())) {
            if (!fieldRefToRegMap.containsKey(FieldRef.KC_X)) {
                loadKcX(op.result());
            } else {
                mov().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().printFieldRef(FieldRef.KC_X);
            }
        } else if (op.fieldName().equals(FieldRef.KC_MAXX.toString())) {
            if (!fieldRefToRegMap.containsKey(FieldRef.KC_X)) {
                loadKcX(op.operandNAsValue(0));
            }
            ld().global().u32().space().printFieldRefAndVal(FieldRef.KC_MAXX, op.result()).commaSpace().address(fieldRefToRegMap.get(FieldRef.KC_X_ADDR).name(), 4);
        } else {
            append(op.fieldName()).append("hglksdhgklhdsklh");
            ld().global().u32().space().printResult(op, PTXRegister.Type.U64).commaSpace().printVar(op.operandNAsValue(0));
        }
    }

    public void loadKcX(Value value) {
        append("cvta.to").global().dot().addrSize().space().printFieldRef(FieldRef.KC_X_ADDR).commaSpace().printAndAddVar(paramMap.get("kc"), addrType()).semicolon().nl().ptxIndent();
        mov().u32().space().printFieldRef(FieldRef.NTID_X).commaSpace().append("%ntid.x").semicolon().nl().ptxIndent();
        mov().u32().space().printFieldRef(FieldRef.CTAID_X).commaSpace().append("%ctaid.x").semicolon().nl().ptxIndent();
        mov().u32().space().printFieldRef(FieldRef.TID_X).commaSpace().append("%tid.x").semicolon().nl().ptxIndent();
        append("mad.lo").s32().space().printFieldRefAndVal(FieldRef.KC_X, value).commaSpace()
                .printFieldRef(FieldRef.CTAID_X).commaSpace().printFieldRef(FieldRef.NTID_X).commaSpace().printFieldRef(FieldRef.TID_X).semicolon().nl().ptxIndent();
        st().global().u32().space().address(fieldRefToRegMap.get(FieldRef.KC_X_ADDR).name()).commaSpace().printFieldRef(FieldRef.KC_X);
    }

    public void fieldStore(FieldStoreOpWrapper op) {
        append(op.toString());
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
        symbol(op.op()).dot().printResultType(op.resultType()).space();
        printResult(op, resultType(op.resultType()));
        commaSpace();
        printVar(op.operandNAsValue(0));
        commaSpace();
        printVar(op.operandNAsValue(1));
    }

    public void binaryTest(BinaryTestOpWrapper op) {
        append("setp").dot();
        //TODO: fix type
        symbol(op.op()).dot().append(PTXRegister.Type.S32.toString()).space();
        printResult(op, PTXRegister.Type.PREDICATE);
        commaSpace();
        printVar(op.operandNAsValue(0));
        commaSpace();
        printVar(op.operandNAsValue(1));
    }

    //TODO: fix (i think this is multiplying the idx by 4)
    public void conv(ConvOpWrapper op) {
        if (op.resultJavaType().equals(JavaType.LONG)) {
            append("mul.wide").s32().space().printResult(op, PTXRegister.Type.U64).commaSpace();
            printVar(op.operandNAsValue(0)).commaSpace().append("4");
        } else if (op.resultJavaType().equals(JavaType.FLOAT)) {
            append("cvt.rn").f32().dot().append(getVar(op.operandNAsValue(0)).type().toString()).space().printResult(op, PTXRegister.Type.F32).commaSpace();
            printVar(op.operandNAsValue(0));
        } else {
            printResult(op, PTXRegister.Type.S32);
        }
    }

    public void constant(ConstantOpWrapper op) {
        append("mov").dot().printResultType(op.resultType()).space().printResult(op, resultType(op.resultType())).commaSpace().append(op.op().value().toString());
    }

    public void javaYield(YieldOpWrapper op) {
        append(op.toString());
    }

    public void funcCall(FuncCallOpWrapper op) {
        append(op.toString());
    }

    public void logical(LogicalOpWrapper op) {
        append(op.toString());
    }

    //TODO: fix later
    public void methodCall(InvokeOpWrapper op) {
        if (op.methodRef().toString().equals("hat.buffer.S32Array::array(long)int")) {
            PTXRegister temp = new PTXRegister(incrOrdinal(addrType()), addrType());
            append("add").s64().space().append(temp.name()).commaSpace().printVar(op.operandNAsValue(0)).commaSpace().printVar(op.operandNAsValue(1)).semicolon().nl().ptxIndent();
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(temp.name(), 4);
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array::array(long, int)void")) {
            PTXRegister temp = new PTXRegister(incrOrdinal(addrType()), addrType());
            append("add").s64().space().append(temp.name()).commaSpace().printVar(op.operandNAsValue(0)).commaSpace().printVar(op.operandNAsValue(1)).semicolon().nl().ptxIndent();
            st().global().u32().space().address(temp.name(), 4).commaSpace().printVar(op.operandNAsValue(2));
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array::length()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(getVar(op.operandNAsValue(0)).name());
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::array(long, int)void")) {
            PTXRegister temp = new PTXRegister(incrOrdinal(addrType()), addrType());
            append("add").s64().space().append(temp.name()).commaSpace().printVar(op.operandNAsValue(0)).commaSpace().printVar(op.operandNAsValue(1)).semicolon().nl().ptxIndent();
            st().global().u32().space().address(temp.name(), 8).commaSpace().printVar(op.operandNAsValue(2));
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::width()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(getVar(op.operandNAsValue(0)).name());
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::height()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(getVar(op.operandNAsValue(0)).name(), 4);
        } else if (op.methodRef().toString().equals("squares.Squares::squareit(int)int")) {
            append("mul.lo").s32().space().printResult(op, PTXRegister.Type.U32).commaSpace().printVar(op.operandNAsValue(0)).commaSpace().printVar(op.operandNAsValue(0));
        } else {
            append("call ").append(op.methodRef().toString());
            printResult(op, PTXRegister.Type.S32);
        }
    }

    public void ternary(TernaryOpWrapper op) {
        append(op.toString());
    }

    //TODO: fix
    public void varDeclaration(VarDeclarationOpWrapper op) {
        append("cvta.to").global().dot().space().printResult(op, PTXRegister.Type.U32).commaSpace();
    }

    // declaring function variables
    public void varFuncDeclaration(VarFuncDeclarationOpWrapper op) {
        ld().param().addrSize().space().printResult(op, addrType());
    }

    public void tuple(TupleOpWrapper op) {
        append(op.toString());
    }

    public void ret(ReturnOpWrapper op) {
        append("ret");
        if (op.hasOperands()) space().printVar(op.operandNAsResult(0));
    }

    public void javaLabeled(JavaLabeledOpWrapper op) {
        append(op.toString());
    }

    public void javaBreak(JavaBreakOpWrapper op) {
        append(op.toString());
    }

    public void javaContinue(JavaContinueOpWrapper op) {
        append(op.toString());
    }

    public void branch(CoreOp.BranchOp op) {
        Block.Reference blockRef = op.successors().getFirst();
        for (Value val : blockRef.arguments()) {
            Block.Parameter p = blockRef.targetBlock().parameters().get(blockRef.arguments().indexOf(val));
            append("mov").dot().printResultType(p.type()).space().printAndAddVar(p, resultType(p.type())).commaSpace().printVar(val).semicolon().nl().ptxIndent();
        }

        append("bra").space().printBlock(blockRef.targetBlock());

//        if (!blockRef.arguments().isEmpty()) {
//            space().oparen();
//            commaSeparated(blockRef.arguments(), this::printVar);
//            cparen();
//        }
    }

    public void condBranch(CoreOp.ConditionalBranchOp op) {
        Block.Reference blockRef = op.successors().getFirst();
        for (Value val : blockRef.arguments()) {
            Block.Parameter p = blockRef.targetBlock().parameters().get(blockRef.arguments().indexOf(val));
            append("mov").dot().printResultType(p.type()).space().printAndAddVar(p, resultType(p.type())).commaSpace().printVar(val).semicolon().nl().ptxIndent();
        }
        append("@").printVar(op.operands().getFirst()).space().append("bra").space().printBlock(blockRef.targetBlock()).semicolon().nl().ptxIndent();
        append("bra").space().printBlock(op.successors().getLast().targetBlock());
    }

    /*
     * Helper functions for printing blocks and variables
     */

    public PTXCodeBuilder printBlock(Block block) {
        return append("block_").append(String.valueOf(block.index()));
    }

    //TODO: FIX
    //prints result to PTXCodeBuilder (and adds if necessary)
    public PTXCodeBuilder printFieldRef(FieldRef ref) {
        if (fieldRefToRegMap.containsKey(ref)) {
            return append(fieldRefToRegMap.get(ref).name());
        }
        if (ref.isDestination()) {
            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(addrType()), addrType()));
        } else {
            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32));
        }
        return append(fieldRefToRegMap.get(ref).name());
    }

    public PTXCodeBuilder printFieldRefAndVal(FieldRef ref, Value value) {
        if (fieldRefToRegMap.containsKey(ref)) {
            return append(fieldRefToRegMap.get(ref).name());
        }
        if (ref.isDestination()) {
            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(getOrdinal(addrType()), addrType()));
            addVar(value, addrType());
        } else {
            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(getOrdinal(PTXRegister.Type.U32), PTXRegister.Type.U32));
            addVar(value, PTXRegister.Type.U32);
        }
        return append(fieldRefToRegMap.get(ref).name());
    }

    //prints result to PTXCodeBuilder (and adds if necessary)
    public PTXCodeBuilder printResult(OpWrapper<?> opWrapper, PTXRegister.Type type) {
        return append(addVar(opWrapper.result(), type));
    }

    public PTXCodeBuilder printAndAddVar(Value val, PTXRegister.Type type) {
        if (varToRegMap.containsKey(val)) {
            return append(varToRegMap.get(val).name());
        } else {
            return append(addVar(val, type));
        }
    }

    public PTXCodeBuilder printVar(Value val) {
        if (varToRegMap.containsKey(val)) {
            return append(varToRegMap.get(val).name());
        } else {
            throw new IllegalStateException("HUH");
        }
    }

    public PTXRegister getVar(Value val) {
        if (varToRegMap.containsKey(val)) {
            return varToRegMap.get(val);
        } else {
            throw new IllegalStateException("erm");
        }
    }

    public String addVar(Value val, PTXRegister.Type type) {
        if (varToRegMap.containsKey(val)) {
            return varToRegMap.get(val).name();
        }
        varToRegMap.putIfAbsent(val, new PTXRegister(incrOrdinal(type), type));
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
        return (addressSize == 32) ? append(PTXRegister.Type.U32.toString()) : append(PTXRegister.Type.U64.toString());
    }

    public PTXRegister.Type addrType() {
        return (addressSize == 32) ? PTXRegister.Type.U32 : PTXRegister.Type.U64;
    }

    public PTXCodeBuilder printResultType(TypeElement type) {
        return append(resultType(type).toString());
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

    public PTXCodeBuilder u64() {
        return dot().append(PTXRegister.Type.U64.toString());
    }

    public PTXCodeBuilder s64() {
        return dot().append(PTXRegister.Type.S64.toString());
    }
}
