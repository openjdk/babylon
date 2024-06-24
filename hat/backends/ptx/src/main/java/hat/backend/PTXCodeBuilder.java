package hat.backend;

import hat.optools.*;
import hat.text.CodeBuilder;
import hat.util.StreamCounter;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
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
    // field ref to reg map because we dont have values to map the field refs to :sob:
    Map<PTXCodeBuilder.FieldRef, PTXRegister> fieldRefToRegMap;

    private int rOrdinal;
    private int rdOrdinal;
    private int fOrdinal;
    private int fdOrdinal;
    private int predOrdinal;

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
        rOrdinal = 1;
        rdOrdinal = 1;
        fOrdinal = 1;
        fdOrdinal = 1;
        predOrdinal = 1;
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
            ptxIndent().param().space().addrSize().space().append(info.varOp.varName());
            params.add(info.varOp.varName());
        }).nl()).nl();
        return this;
    }

    public void blockBody(Block block, Stream<OpWrapper<?>> ops) {
        if (block.index() == 0) {
            for (Block.Parameter p : block.parameters()) {
                ptxIndent().ld().param().dot().addrSize().ptxIndent();
                printItem(p, addrType(), true).commaSpace().osbrace().append(params.get(p.index())).csbrace().nl();
                paramMap.putIfAbsent(params.get(p.index()), p);
            }
        }
        printBlock(block).colon().nl();
        if (!block.parameters().isEmpty() && block.index() > 0) {
            for (Block.Parameter p : block.parameters()) {
                ptxIndent().ld().param().dot().addrSize().ptxIndent();
                printItem(p, addrType(), true).commaSpace().osbrace().append(params.get(p.index())).csbrace().nl();
            }
        }
        ops.forEach(op -> ptxIndent().convert(op).semicolon().nl());
    }

    public void ptxRegisterDecl() {
        if (predOrdinal > 1) append(".reg").space().dot().append(PTXRegister.Type.PREDICATE.toString()).ptxIndent().percent().append("p").lt().append(String.valueOf(predOrdinal)).gt().semicolon().nl();
        if (rOrdinal > 1) append(".reg").space().dot().append(PTXRegister.Type.B32.toString()).ptxIndent().percent().append("r").lt().append(String.valueOf(rOrdinal)).gt().semicolon().nl();
        if (rdOrdinal > 1) append(".reg").space().dot().append(PTXRegister.Type.B64.toString()).ptxIndent().percent().append("rd").lt().append(String.valueOf(rdOrdinal)).gt().semicolon().nl();
        if (fOrdinal > 1) append(".reg").space().dot().append(PTXRegister.Type.F32.toString()).ptxIndent().percent().append("f").lt().append(String.valueOf(fOrdinal)).gt().semicolon().nl();
        if (fdOrdinal > 1) append(".reg").space().dot().append(PTXRegister.Type.F64.toString()).ptxIndent().percent().append("fd").lt().append(String.valueOf(fdOrdinal)).gt().semicolon().nl();
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

    }

    public void varStore(VarStoreOpWrapper op) {

    }

    // TODO: handle for S32Array
    public void fieldLoad(FieldLoadOpWrapper op) {
        if (op.fieldName().equals(FieldRef.KC_X.toString())) {
            if (!fieldRefToRegMap.containsKey(FieldRef.KC_X)) {
                loadKcX(op.result());
            } else {
                ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().printFieldRef(FieldRef.KC_X);
            }
        } else if (op.fieldName().equals(FieldRef.KC_MAXX.toString())) {
            if (!fieldRefToRegMap.containsKey(FieldRef.KC_X)) {
                loadKcX(op.operandNAsValue(0));
            }
            ld().global().u32().space().printFieldRefAndVal(FieldRef.KC_MAXX, op.result()).commaSpace()
                    .address(fieldRefToRegMap.get(FieldRef.KC_X_ADDR).name(), 4);
        } else {
            append("aa :: " + op.fieldName());
            ld().global().u32().space().printResult(op, PTXRegister.Type.U64).commaSpace().getItem(op.operandNAsValue(0));
        }
    }

    //TODO: FIX!!!
    public void loadKcX(Value value) {
        append("cvta.to").global().dot().addrSize().space().printFieldRef(FieldRef.KC_X_ADDR).commaSpace().printItem(paramMap.get("kc"), addrType(), true).nl().ptxIndent();
        mov().u32().space().printFieldRef(FieldRef.NTID_X).commaSpace().append("%ntid.x").nl().ptxIndent();
        mov().u32().space().printFieldRef(FieldRef.CTAID_X).commaSpace().append("%ctaid.x").nl().ptxIndent();
        mov().u32().space().printFieldRef(FieldRef.TID_X).commaSpace().append("%tid.x").nl().ptxIndent();
        append("mad.lo").s32().space().printFieldRefAndVal(FieldRef.KC_X, value).space()
                .printFieldRef(FieldRef.CTAID_X).space().printFieldRef(FieldRef.NTID_X).space().printFieldRef(FieldRef.TID_X).nl().ptxIndent();
        st().global().u32().space().address(fieldRefToRegMap.get(FieldRef.KC_X_ADDR).name()).commaSpace().printFieldRef(FieldRef.KC_X);
    }

    public void fieldStore(FieldStoreOpWrapper op) {

    }

    PTXCodeBuilder symbol(Op op) {
        return switch (op) {
            case CoreOp.ModOp _ -> append("rem").u32();
            case CoreOp.MulOp _ -> append("mul").s32();
            case CoreOp.DivOp _ -> append("div").s32();
            case CoreOp.AddOp _ -> append("add").s32();
            case CoreOp.SubOp _ -> append("sub").s32();
            case CoreOp.LtOp _ -> append("lt").s32();
            case CoreOp.GtOp _ -> append("gt").s32();
            case CoreOp.LeOp _ -> append("le").s32();
            case CoreOp.GeOp _ -> append("ge").s32();
            case CoreOp.NeqOp _ -> append("ne").u32();
            case CoreOp.EqOp _ -> append("eq").u32();
            case CoreOp.XorOp _ -> append("xor").u32();
            case ExtendedOp.JavaConditionalAndOp _ -> append("&&");
            case ExtendedOp.JavaConditionalOrOp _ -> append("||");
            default -> throw new IllegalStateException("Unexpected value: " + op);
        };
    }

    public PTXRegister.Type symbolReturnType(Op op) {
        return switch (op) {
            case CoreOp.MulOp _, CoreOp.DivOp _, CoreOp.AddOp _, CoreOp.SubOp _, CoreOp.LtOp _, CoreOp.GtOp _, CoreOp.LeOp _, CoreOp.GeOp _ -> PTXRegister.Type.S32;
            case CoreOp.ModOp _, CoreOp.NeqOp _, CoreOp.EqOp _ -> PTXRegister.Type.U32;
            default -> throw new IllegalStateException("Unexpected value: " + op);
        };
    }

    public void binaryOperation(BinaryArithmeticOrLogicOperation op) {
        System.out.println(op.operandNAsValue(0) + " " + op.operandNAsValue(1));
        symbol(op.op()).space();
        printResult(op, symbolReturnType(op.op()));
        commaSpace();
        getItem(op.operandNAsValue(0));
        commaSpace();
        getItem(op.operandNAsValue(1));
//        if (isParam(op.operandNAsValue(1))) {
//            printItem(op.operandNAsValue(1));
//        } else {
//            printResult(op.rhsAsOp().result());
//        }
    }

    public void binaryTest(BinaryTestOpWrapper op) {
        append("setp").dot();
        symbol(op.op()).space();
        printResult(op, PTXRegister.Type.PREDICATE);
        commaSpace();
        getItem(op.operandNAsValue(0));
        commaSpace();
        getItem(op.operandNAsValue(1));

//        getItem(op.lhsAsOp().result());
//        commaSpace();
//        getItem(op.rhsAsOp().result());
    }

    //TODO: fix (i think this is multiplying the idx by 4)
    public void conv(ConvOpWrapper op) {
        if (op.resultJavaType().equals(JavaType.LONG)) {
            append("mul.wide").s32().space().printResult(op, PTXRegister.Type.S32, true).commaSpace();
            getItem(op.operandNAsValue(0)).commaSpace().append("4");
        } else if (op.resultJavaType().equals(JavaType.FLOAT)) {
            append("cvt.rn").f32().s32().space().printResult(op, PTXRegister.Type.F32, false).commaSpace();
            getItem(op.operandNAsValue(0));
        } else {
            printResult(op, PTXRegister.Type.S32, true);
//            System.out.println("aaaaa " + getItem(op.result()));
        }
    }

    //TODO: fix
    public void constant(ConstantOpWrapper op) {
        printResult(op, PTXRegister.Type.S32, true);
    }

    public void javaYield(YieldOpWrapper op) {

    }

    public void funcCall(FuncCallOpWrapper op) {

    }

    public void logical(LogicalOpWrapper op) {

    }

    //TODO: fix later
    public void methodCall(InvokeOpWrapper op) {
//        if (op.methodRef().equals(MethodRef.)) {
        if (op.methodRef().toString().equals("hat.buffer.S32Array::array(long)int")) {
            PTXRegister temp = new PTXRegister(rdOrdinal++, addrType(), true);
            append("add").s64().space().append(temp.name()).commaSpace().getItem(op.operandNAsValue(0)).commaSpace().getItem(op.operandNAsValue(1)).nl().ptxIndent();
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().address(temp.name(), 4);
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array::array(long, int)void")) {
            PTXRegister temp = new PTXRegister(rdOrdinal++, addrType(), true);
            append("add").s64().space().append(temp.name()).commaSpace().getItem(op.operandNAsValue(0)).commaSpace().getItem(op.operandNAsValue(1)).nl().ptxIndent();
            st().global().u32().space().address(temp.name(), 4).commaSpace().getItem(op.operandNAsValue(2));
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::width()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().getItem(op.operandNAsValue(0));
        } else if (op.methodRef().toString().equals("hat.buffer.S32Array2D::length()int")) {
            ld().global().u32().space().printResult(op, PTXRegister.Type.U32).commaSpace().getItem(op.operandNAsValue(0));
        } else if (op.methodRef().toString().equals("squares.Squares::sq(int)int")) {
            append("mul.lo").s32().space().printResult(op, PTXRegister.Type.S32, false).commaSpace().getItem(op.operandNAsValue(0)).commaSpace().getItem(op.operandNAsValue(0));
        } else {
            append("call ").append(op.methodRef().toString());
            printResult(op, PTXRegister.Type.S32, true);
        }
    }

    public void ternary(TernaryOpWrapper op) {

    }

    //TODO: fix
    public void varDeclaration(VarDeclarationOpWrapper op) {
//        append("cvta.to").global().dot().varSize().space().printAndAddResult(op).commaSpace()
//                .printItem(op.operandNAsValue(0)).nl().ptxIndent();
//        mov().space().printResult(op, false).commaSpace().printItem(op.operandNAsValue(0));
    }

    // declaring function variables
    public void varFuncDeclaration(VarFuncDeclarationOpWrapper op) {
//        ld().param().addrSize().space().printResult(op, addrType());
//        commaSpace();
//        printItem(op.operandNAsValue(0));
    }

    public void tuple(TupleOpWrapper op) {

    }

    public void ret(ReturnOpWrapper op) {
        append("ret");
        if (op.hasOperands()) space().getItem(op.operandNAsResult(0));
    }

    public void javaLabeled(JavaLabeledOpWrapper op) {

    }

    public void javaBreak(JavaBreakOpWrapper op) {

    }

    public void javaContinue(JavaContinueOpWrapper op) {

    }

    public void branch(CoreOp.BranchOp op) {
        append("bra").space().printBlock(op.successors().getFirst().targetBlock());
    }

    public void condBranch(CoreOp.ConditionalBranchOp op) {
        append("@").getItem(op.operands().getFirst()).space().append("bra").space().printBlock(op.successors().getFirst().targetBlock()).nl().ptxIndent();
        append("bra").space().printBlock(op.successors().getLast().targetBlock());
    }

    /*
     * Helper functions for printing blocks and variables
     */

    public boolean isParam(Value val) {
        //TODO: FIXXXX
        return true;
    }

    public boolean isGlobal(Value val) {
        //TODO: FIXXXX
        return true;
    }

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
            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(addrType(), true), addrType(), true));
        } else {
            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(incrOrdinal(PTXRegister.Type.U32, false), PTXRegister.Type.U32, false));
        }
        return append(fieldRefToRegMap.get(ref).name());
    }

    // TODO: FIX
    public PTXCodeBuilder printFieldRefAndVal(FieldRef ref, Value value) {
        if (fieldRefToRegMap.containsKey(ref)) {
            return append(fieldRefToRegMap.get(ref).name());
        }
//        if (ref == FieldRef.KC_X_ADDR) {
            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(getOrdinal(addrType(), ref.isDestination()), addrType(), ref.isDestination()));
            addItem(value, addrType(), ref.isDestination());
//        } else {
//            fieldRefToRegMap.putIfAbsent(ref, new PTXRegister(rOrdinal, PTXRegister.Type.U32));
//            addItem(value, PTXRegister.Type.U32);
//        }
        return append(fieldRefToRegMap.get(ref).name());
    }

    //prints result to PTXCodeBuilder (and adds if necessary)
    public PTXCodeBuilder printResult(OpWrapper<?> opWrapper, PTXRegister.Type type, boolean destination) {
        return append(addItem(opWrapper.result(), type, destination));
    }

    //prints result to PTXCodeBuilder (and adds if necessary)
    public PTXCodeBuilder printResult(OpWrapper<?> opWrapper, PTXRegister.Type type) {
        return append(addItem(opWrapper.result(), type, false));
    }

    public PTXCodeBuilder printItem(Value val, PTXRegister.Type type, boolean destination) {
        return append(addItem(val, type, destination));
    }

    public PTXCodeBuilder getItem(Value val) {
        if (varToRegMap.containsKey(val)) {
            return append(varToRegMap.get(val).name());
        } else {
            throw new IllegalStateException("HUH " + val);
        }
    }

    public String addItem(Value val, PTXRegister.Type type, boolean destination) {
        if (varToRegMap.containsKey(val)) {
            System.out.println("aaaaaaa " + val + " " + type);
            return varToRegMap.get(val).name();
        }
        varToRegMap.putIfAbsent(val, new PTXRegister(incrOrdinal(type, destination), type, destination));
        return varToRegMap.get(val).name();
    }

    public Integer getOrdinal(PTXRegister.Type type, boolean destination) {
        if (destination) return rdOrdinal;
        if (type.getSize() == 1) return predOrdinal;
        switch (type) {
            case F32 -> {
                return fOrdinal;
            }
            case F64 -> {
                return fdOrdinal;
            }
            default -> {
                return rOrdinal;
            }
        }
    }

    public Integer incrOrdinal(PTXRegister.Type type, boolean destination) {
        if (destination) return rdOrdinal++;
        if (type.getSize() == 1) return predOrdinal++;
        switch (type) {
            case F32 -> {
                return fOrdinal++;
            }
            case F64 -> {
                return fdOrdinal++;
            }
            default -> {
                return rOrdinal++;
            }
        }
    }

    public PTXCodeBuilder addrSize() {
        return (addressSize == 32) ? append(PTXRegister.Type.U32.toString()) : append(PTXRegister.Type.U64.toString());
    }

    public PTXRegister.Type addrType() {
        return (addressSize == 32) ? PTXRegister.Type.U32 : PTXRegister.Type.U64;
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
