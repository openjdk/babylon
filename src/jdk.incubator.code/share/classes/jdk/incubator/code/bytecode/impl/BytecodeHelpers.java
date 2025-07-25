package jdk.incubator.code.bytecode.impl;

import java.lang.classfile.Opcode;

public final class BytecodeHelpers {

    // Copied from java.base/jdk.internal.classfile.impl.BytecodeHelpers
    // to avoid export of package from java.base to jdk.incubator.code

    public static boolean isUnconditionalBranch(Opcode opcode) {
        return switch (opcode) {
            case GOTO, ATHROW, GOTO_W, LOOKUPSWITCH, TABLESWITCH -> true;
            default -> opcode.kind() == Opcode.Kind.RETURN;
        };
    }
}
