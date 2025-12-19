package hat.dialect;

import jdk.incubator.code.dialect.java.JavaOp;

public enum BinaryOpEnum {
    ADD("+"),
    SUB("-"),
    MUL("*"),
    DIV("/");

    String symbol;

    BinaryOpEnum(String symbol) {
        this.symbol = symbol;
    }
    public static BinaryOpEnum of(JavaOp.InvokeOp invokeOp) {
        return switch (invokeOp.invokeDescriptor().name()) {
            case "add" -> BinaryOpEnum.ADD;
            case "sub" -> BinaryOpEnum.SUB;
            case "mul" -> BinaryOpEnum.MUL;
            case "div" -> BinaryOpEnum.DIV;
            default -> throw new RuntimeException("Unknown binary op " + invokeOp.invokeDescriptor().name());
        };
    }
    public String symbol() {
        return symbol;
    }
}
