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

package java.lang.reflect.code.bytecode;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.classfile.constantpool.ConstantPoolBuilder;
import java.lang.classfile.constantpool.DoubleEntry;
import java.lang.classfile.constantpool.FieldRefEntry;
import java.lang.classfile.constantpool.FloatEntry;
import java.lang.classfile.constantpool.IntegerEntry;
import java.lang.classfile.constantpool.LoadableConstantEntry;
import java.lang.classfile.constantpool.LongEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.ArrayStoreInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.IncrementInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.LookupSwitchInstruction;
import java.lang.classfile.instruction.NewMultiArrayInstruction;
import java.lang.classfile.instruction.NewObjectInstruction;
import java.lang.classfile.instruction.NewPrimitiveArrayInstruction;
import java.lang.classfile.instruction.NewReferenceArrayInstruction;
import java.lang.classfile.instruction.OperatorInstruction;
import java.lang.classfile.instruction.ReturnInstruction;
import java.lang.classfile.instruction.StackInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.classfile.instruction.SwitchCase;
import java.lang.classfile.instruction.TableSwitchInstruction;
import java.lang.classfile.instruction.ThrowInstruction;
import java.lang.classfile.instruction.TypeCheckInstruction;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDesc;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.FieldDesc;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.op.OpDeclaration;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.lang.reflect.code.TypeElement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

class BytecodeInstructionOps {

    interface MethodVisitorContext {
        Deque<ExceptionTableStart> exceptionRegionStack();

        Label getLabel(Object o);
    }

    public record InstructionDef<T extends Instruction>(T instruction, List<Block.Reference> successors) {

        InstructionDef(T instruction) {
            this(instruction, List.of());
        }

        Opcode opcode() {
            return instruction.opcode();
        }
    }

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.TYPE)
    public @interface Opcodes {
        Opcode[] value();
    }

    public static abstract class InstructionOp extends Op {
        InstructionOp(InstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        InstructionOp(String name) {
            super(name, List.of());
        }

        @Override
        public final Map<String, Object> attributes() {
            Map<String, Object> m = _attributes();
            return m.isEmpty() ? m : Collections.unmodifiableMap(m);
        }

        Map<String, Object> _attributes() {
            return Map.of();
        }

        // Produce an ASM bytecode instruction
        public abstract void apply(CodeBuilder b, MethodVisitorContext c);

        @Override
        public TypeElement resultType() {
            // I chose VOID, because bytecode instructions manipulate the stack
            // plus the type of what an operation will push/pop mayn not be known, e.g. pop instruction
            return JavaType.VOID;
        }
    }

    public static abstract class TerminatingInstructionOp extends InstructionOp implements Op.Terminating {
        final List<Block.Reference> successors;

        TerminatingInstructionOp(TerminatingInstructionOp that, CopyContext cc) {
            super(that, cc);

            // Copy successors
            this.successors = that.successors().stream()
                    .map(cc::getSuccessorOrCreate)
                    .toList();
        }

        TerminatingInstructionOp(String name) {
            super(name);

            this.successors = List.of();
        }

        TerminatingInstructionOp(String name, List<Block.Reference> s) {
            super(name);

            this.successors = List.copyOf(s);
        }

        @Override
        public List<Block.Reference> successors() {
            return successors;
        }
    }

    public static abstract class TypedInstructionOp extends InstructionOp {
        public static final String ATTRIBUTE_TYPE = "type";

        final TypeKind type;

        TypedInstructionOp(TypedInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.type = that.type;
        }

        TypedInstructionOp(String name, TypeKind type) {
            super(name);

            this.type = type;
        }

        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_TYPE, type);
            return m;
        }

        public TypeKind type() {
            return type;
        }

        public TypeElement typeDesc() {
            return switch (type) {
//                case BooleanType -> TypeDesc.BOOLEAN;
//                case ByteType -> TypeDesc.BYTE;
//                case ShortType -> TypeDesc.SHORT;
//                case CharType -> TypeDesc.CHAR;
                case IntType -> JavaType.INT;
                case FloatType -> JavaType.FLOAT;
                case LongType -> JavaType.LONG;
                case DoubleType -> JavaType.DOUBLE;
                case ReferenceType -> JavaType.J_L_OBJECT;
                default -> throw new IllegalArgumentException("Bad type kind: " + type);
            };
        }

    }

    public static abstract class TypedTerminatingInstructionOp extends TerminatingInstructionOp {
        public static final String ATTRIBUTE_TYPE = "type";

        final TypeKind type;

        TypedTerminatingInstructionOp(TypedTerminatingInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.type = that.type;
        }

        TypedTerminatingInstructionOp(String name, TypeKind type, List<Block.Reference> s) {
            super(name, s);

            this.type = type;
        }

        TypedTerminatingInstructionOp(String name, TypeKind type) {
            super(name);

            this.type = type;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_TYPE, type);
            return m;
        }

        public TypeKind type() {
            return type;
        }
    }

    public static abstract class VarInstructionOp extends TypedInstructionOp {
        public static final String ATTRIBUTE_INDEX = "index";

        final int slot;

        VarInstructionOp(VarInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.slot = that.slot;
        }

        VarInstructionOp(String name, TypeKind type, int slot) {
            super(name, type);

            this.slot = slot;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = super._attributes();
            m.put(ATTRIBUTE_INDEX, slot);
            return m;
        }

        public int slot() {
            return slot;
        }
    }


    @Opcodes({
            Opcode.ALOAD,
            Opcode.ALOAD_0,
            Opcode.ALOAD_1,
            Opcode.ALOAD_2,
            Opcode.ALOAD_3,
            Opcode.ILOAD,
            Opcode.ILOAD_0,
            Opcode.ILOAD_1,
            Opcode.ILOAD_2,
            Opcode.ILOAD_3,
            Opcode.LLOAD,
            Opcode.LLOAD_0,
            Opcode.LLOAD_1,
            Opcode.LLOAD_2,
            Opcode.LLOAD_3,
            Opcode.FLOAD,
            Opcode.FLOAD_0,
            Opcode.FLOAD_1,
            Opcode.FLOAD_2,
            Opcode.FLOAD_3,
            Opcode.DLOAD,
            Opcode.DLOAD_0,
            Opcode.DLOAD_1,
            Opcode.DLOAD_2,
            Opcode.DLOAD_3
    })
    @OpDeclaration(LoadInstructionOp.NAME)
    public static final class LoadInstructionOp extends VarInstructionOp {
        public static final String NAME = "Tload";

        LoadInstructionOp(InstructionDef<LoadInstruction> def) {
            this(def.instruction().typeKind(), def.instruction().slot());
        }

        LoadInstructionOp(LoadInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LoadInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new LoadInstructionOp(this, cc);
        }

        LoadInstructionOp(TypeKind type, int slot) {
            super(NAME, type, slot);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.loadInstruction(type(), slot());
        }
    }

    public static LoadInstructionOp load(TypeKind type, int index) {
        return new LoadInstructionOp(type, index);
    }

    @Opcodes({
            Opcode.ASTORE,
            Opcode.ASTORE_0,
            Opcode.ASTORE_1,
            Opcode.ASTORE_2,
            Opcode.ASTORE_3,
            Opcode.ISTORE,
            Opcode.ISTORE_0,
            Opcode.ISTORE_1,
            Opcode.ISTORE_2,
            Opcode.ISTORE_3,
            Opcode.LSTORE,
            Opcode.LSTORE_0,
            Opcode.LSTORE_1,
            Opcode.LSTORE_2,
            Opcode.LSTORE_3,
            Opcode.FSTORE,
            Opcode.FSTORE_0,
            Opcode.FSTORE_1,
            Opcode.FSTORE_2,
            Opcode.FSTORE_3,
            Opcode.DSTORE,
            Opcode.DSTORE_0,
            Opcode.DSTORE_1,
            Opcode.DSTORE_2,
            Opcode.DSTORE_3
    })
    @OpDeclaration(StoreInstructionOp.NAME)
    public static final class StoreInstructionOp extends VarInstructionOp {
        public static final String NAME = "Tstore";

        StoreInstructionOp(InstructionDef<StoreInstruction> def) {
            this(def.instruction().typeKind(), def.instruction().slot());
        }

        StoreInstructionOp(StoreInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public StoreInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new StoreInstructionOp(this, cc);
        }

        StoreInstructionOp(TypeKind type, int index) {
            super(NAME, type, index);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.storeInstruction(type(), slot());
        }
    }

    public static StoreInstructionOp store(TypeKind type, int index) {
        return new StoreInstructionOp(type, index);
    }


    @Opcodes({Opcode.LDC, Opcode.LDC_W, Opcode.LDC2_W})
    @OpDeclaration(LdcInstructionOp.NAME)
    public static final class LdcInstructionOp extends InstructionOp {
        public static final String NAME = "ldc";

        public static final String ATTRIBUTE_TYPE = "type";

        public static final String ATTRIBUTE_VALUE = "value";

        final TypeElement type;
        final Object value;

        LdcInstructionOp(InstructionDef<ConstantInstruction.LoadConstantInstruction> def) {
            this(toTypeDesc(def.instruction().constantEntry()), toValue(def.instruction().constantEntry()));
        }

        private static TypeElement toTypeDesc(LoadableConstantEntry entry) {
            if (entry instanceof IntegerEntry) {
                return JavaType.INT;
            } else if (entry instanceof LongEntry) {
                return JavaType.LONG;
            } else if (entry instanceof FloatEntry) {
                return JavaType.FLOAT;
            } else if (entry instanceof DoubleEntry) {
                return JavaType.DOUBLE;
            } else if (entry instanceof StringEntry) {
                return JavaType.J_L_STRING;
            } else if (entry instanceof ClassEntry) {
                return JavaType.J_L_CLASS;
            } else {
                // @@@ MethodType, MethodHandle, ConstantDynamic
                throw new IllegalArgumentException("Unsupported constant entry: " + entry);
            }
        }

        private static Object toValue(LoadableConstantEntry entry) {
            if (entry instanceof IntegerEntry e) {
                return e.intValue();
            } else if (entry instanceof LongEntry e) {
                return e.longValue();
            } else if (entry instanceof FloatEntry e) {
                return e.floatValue();
            } else if (entry instanceof DoubleEntry e) {
                return e.doubleValue();
            } else if (entry instanceof StringEntry e) {
                return e.stringValue();
            } else if (entry instanceof ClassEntry e) {
                return JavaType.ofNominalDescriptor(e.asSymbol());
            } else {
                // @@@ MethodType, MethodHandle, ConstantDynamic
                throw new IllegalArgumentException("Unsupported constant entry: " + entry);
            }
        }

        LdcInstructionOp(LdcInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.type = that.type;
            this.value = that.value;
        }

        @Override
        public LdcInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new LdcInstructionOp(this, cc);
        }

        LdcInstructionOp(TypeElement type, Object value) {
            super(NAME);

            // @@@ constant dynamic
            // @@@ check value

            this.type = type;
            this.value = value;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_TYPE, type);
            m.put(ATTRIBUTE_VALUE, value);
            return m;
        }

        public TypeElement type() {
            return type;
        }

        public Object value() {
            return value;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.ldc(fromValue(mv.constantPool(), value()));
        }

        static LoadableConstantEntry fromValue(ConstantPoolBuilder b, Object value) {
            if (value instanceof ConstantDesc cd) {
                return b.constantValueEntry(cd);
            } else if (value instanceof JavaType td) {
                return b.classEntry(td.toNominalDescriptor());
            } else {
                throw new IllegalArgumentException("Unsupported constant value: " + value);
            }
        }
    }

    public static LdcInstructionOp ldc(TypeElement type, Object value) {
        return new LdcInstructionOp(type, value);
    }

    @Opcodes({Opcode.ICONST_M1,
            Opcode.ICONST_0, Opcode.ICONST_1, Opcode.ICONST_2, Opcode.ICONST_3, Opcode.ICONST_4, Opcode.ICONST_5,
            Opcode.LCONST_0, Opcode.LCONST_1,
            Opcode.FCONST_0, Opcode.FCONST_1, Opcode.FCONST_2,
            Opcode.DCONST_0, Opcode.DCONST_1
    })
    @OpDeclaration(ConstInstructionOp.NAME)
    public static final class ConstInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tconst";

        public static final String ATTRIBUTE_VALUE = "value";

        final int value;

        ConstInstructionOp(InstructionDef<ConstantInstruction.IntrinsicConstantInstruction> def) {
            this(def.instruction().typeKind(), getValue(def.instruction().opcode().constantValue()));
        }

        ConstInstructionOp(ConstInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.value = that.value;
        }

        @Override
        public ConstInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new ConstInstructionOp(this, cc);
        }

        ConstInstructionOp(TypeKind type, int value) {
            super(NAME, type);

            switch (type) {
                case IntType -> {
                    if (value < -2 || value > 5) {
                        throw new IllegalArgumentException("Constant integer value out of range [-1, 5]: " + value);
                    }
                }
                case LongType -> {
                    if (value < 0 || value > 1) {
                        throw new IllegalArgumentException("Constant long value out of range [0, 1]: " + value);
                    }
                }
                case FloatType -> {
                    if (value < 0 || value > 2) {
                        throw new IllegalArgumentException("Constant float value not 0.0, 1.0, or 2.0: " + value);
                    }
                }
                case DoubleType -> {
                    if (value < 0 || value > 1) {
                        throw new IllegalArgumentException("Constant double value not 0.0, or 1.0: " + value);
                    }
                }
                default -> {
                    throw new IllegalArgumentException("Bad type for const instruction: " + type);
                }
            }

            this.value = value;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = super._attributes();
            m.put(ATTRIBUTE_VALUE, value);
            return m;
        }

        public int value() {
            return value;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.with(ConstantInstruction.ofIntrinsic(getOpcode(type, value)));
        }

        private static Opcode getOpcode(TypeKind t, int v) {
            return switch (t) {
                case IntType -> switch (v) {
                    case -1 -> Opcode.ICONST_M1;
                    case 0 -> Opcode.ICONST_0;
                    case 1 -> Opcode.ICONST_1;
                    case 2 -> Opcode.ICONST_2;
                    case 3 -> Opcode.ICONST_3;
                    case 4 -> Opcode.ICONST_4;
                    case 5 -> Opcode.ICONST_5;
                    default -> throw new InternalError("Should not reach here");
                };
                case LongType -> switch (v) {
                    case 0 -> Opcode.LCONST_0;
                    case 1 -> Opcode.LCONST_1;
                    default -> throw new InternalError("Should not reach here");
                };
                case FloatType -> switch (v) {
                    case 0 -> Opcode.FCONST_0;
                    case 1 -> Opcode.FCONST_1;
                    case 2 -> Opcode.FCONST_2;
                    default -> throw new InternalError("Should not reach here");
                };
                case DoubleType -> switch (v) {
                    case 0 -> Opcode.DCONST_0;
                    case 1 -> Opcode.DCONST_1;
                    default -> throw new InternalError("Should not reach here");
                };
                default -> throw new InternalError("Should not reach here");
            };
        }

        private static int getValue(ConstantDesc c) {
            if (c instanceof Number n) {
                return n.intValue();
            } else {
                throw new IllegalArgumentException("Unsupported constant value: " + c);
            }
        }
    }

    public static ConstInstructionOp _const(TypeKind type, int value) {
        return new ConstInstructionOp(type, value);
    }

    @Opcodes(Opcode.ARRAYLENGTH)
    @OpDeclaration(ArrayLengthInstructionOp.NAME)
    public static final class ArrayLengthInstructionOp extends InstructionOp {
        public static final String NAME = "arraylength";

        ArrayLengthInstructionOp(InstructionDef<OperatorInstruction> def) {
            this();
        }

        ArrayLengthInstructionOp(ArrayLengthInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ArrayLengthInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new ArrayLengthInstructionOp(this, cc);
        }

        ArrayLengthInstructionOp() {
            super(NAME);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.arraylength();
        }
    }

    public static ArrayLengthInstructionOp arraylength() {
        return new ArrayLengthInstructionOp();
    }

    @Opcodes({Opcode.AALOAD, Opcode.BALOAD, Opcode.CALOAD, Opcode.SALOAD,
            Opcode.IALOAD, Opcode.LALOAD, Opcode.FALOAD, Opcode.DALOAD})
    @OpDeclaration(ArrayLoadInstructionOp.NAME)
    public static final class ArrayLoadInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Taload";

        ArrayLoadInstructionOp(InstructionDef<ArrayLoadInstruction> def) {
            this(def.instruction().typeKind());
        }

        ArrayLoadInstructionOp(ArrayLoadInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ArrayLoadInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new ArrayLoadInstructionOp(this, cc);
        }

        ArrayLoadInstructionOp(TypeKind type) {
            super(NAME, type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.arrayLoadInstruction(type);
        }
    }

    public static ArrayLoadInstructionOp aload(TypeKind t) {
        return new ArrayLoadInstructionOp(t);
    }

    @Opcodes({Opcode.AASTORE, Opcode.BASTORE, Opcode.CASTORE, Opcode.SASTORE,
            Opcode.IASTORE, Opcode.LASTORE, Opcode.FASTORE, Opcode.DASTORE})
    @OpDeclaration(ArrayStoreInstructionOp.NAME)
    public static final class ArrayStoreInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tastore";

        ArrayStoreInstructionOp(InstructionDef<ArrayStoreInstruction> def) {
            this(def.instruction().typeKind());
        }

        ArrayStoreInstructionOp(ArrayStoreInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ArrayStoreInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new ArrayStoreInstructionOp(this, cc);
        }

        ArrayStoreInstructionOp(TypeKind type) {
            super(NAME, type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.arrayStoreInstruction(type);
        }
    }

    public static ArrayStoreInstructionOp astore(TypeKind t) {
        return new ArrayStoreInstructionOp(t);
    }


    @Opcodes({Opcode.INEG, Opcode.LNEG, Opcode.FNEG, Opcode.DNEG})
    @OpDeclaration(NegInstructionOp.NAME)
    public static final class NegInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tneg";

        NegInstructionOp(InstructionDef<OperatorInstruction> def) {
            this(def.instruction().typeKind());
        }

        NegInstructionOp(NegInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public NegInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new NegInstructionOp(this, cc);
        }

        NegInstructionOp(TypeKind type) {
            super(NAME, type);

            getOpcode(type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.operatorInstruction(getOpcode(type));
        }

        private static Opcode getOpcode(TypeKind t) {
            return switch (t) {
                case IntType -> Opcode.INEG;
                case LongType -> Opcode.LNEG;
                case FloatType -> Opcode.FNEG;
                case DoubleType -> Opcode.DNEG;
                default -> throw new IllegalArgumentException("Bad type: " + t);
            };
        }
    }

    public static NegInstructionOp neg(TypeKind type) {
        return new NegInstructionOp(type);
    }

    @Opcodes({Opcode.IADD, Opcode.LADD, Opcode.FADD, Opcode.DADD})
    @OpDeclaration(AddInstructionOp.NAME)
    public static final class AddInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tadd";

        AddInstructionOp(InstructionDef<OperatorInstruction> def) {
            this(def.instruction().typeKind());
        }

        AddInstructionOp(AddInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public AddInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new AddInstructionOp(this, cc);
        }

        AddInstructionOp(TypeKind type) {
            super(NAME, type);

            getOpcode(type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.operatorInstruction(getOpcode(type));
        }

        private static Opcode getOpcode(TypeKind t) {
            return switch (t) {
                case IntType -> Opcode.IADD;
                case LongType -> Opcode.LADD;
                case FloatType -> Opcode.FADD;
                case DoubleType -> Opcode.DADD;
                default -> throw new IllegalArgumentException("Bad type: " + t);
            };
        }
    }

    public static AddInstructionOp add(TypeKind type) {
        return new AddInstructionOp(type);
    }

    @Opcodes({Opcode.IMUL, Opcode.LMUL, Opcode.FMUL, Opcode.DMUL})
    @OpDeclaration(MulInstructionOp.NAME)
    public static final class MulInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tmul";

        MulInstructionOp(InstructionDef<OperatorInstruction> def) {
            this(def.instruction().typeKind());
        }

        MulInstructionOp(MulInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public MulInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new MulInstructionOp(this, cc);
        }

        MulInstructionOp(TypeKind type) {
            super(NAME, type);

            getOpcode(type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.operatorInstruction(getOpcode(type));
        }

        private static Opcode getOpcode(TypeKind t) {
            return switch (t) {
                case IntType -> Opcode.IMUL;
                case LongType -> Opcode.LMUL;
                case FloatType -> Opcode.FMUL;
                case DoubleType -> Opcode.DMUL;
                default -> throw new IllegalArgumentException("Bad type: " + t);
            };
        }
    }

    public static MulInstructionOp mul(TypeKind type) {
        return new MulInstructionOp(type);
    }


    @Opcodes({Opcode.IDIV, Opcode.LDIV, Opcode.FDIV, Opcode.DDIV})
    @OpDeclaration(DivInstructionOp.NAME)
    public static final class DivInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tdiv";

        DivInstructionOp(InstructionDef<OperatorInstruction> def) {
            this(def.instruction().typeKind());
        }

        DivInstructionOp(DivInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public DivInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new DivInstructionOp(this, cc);
        }

        DivInstructionOp(TypeKind type) {
            super(NAME, type);

            getOpcode(type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.operatorInstruction(getOpcode(type()));
        }

        private static Opcode getOpcode(TypeKind t) {
            return switch (t) {
                case IntType -> Opcode.IDIV;
                case LongType -> Opcode.LDIV;
                case FloatType -> Opcode.FDIV;
                case DoubleType -> Opcode.DDIV;
                default -> throw new IllegalArgumentException("Bad type: " + t);
            };
        }
    }

    public static DivInstructionOp div(TypeKind type) {
        return new DivInstructionOp(type);
    }

    @Opcodes({Opcode.ISUB, Opcode.LSUB, Opcode.FSUB, Opcode.DSUB})
    @OpDeclaration(SubInstructionOp.NAME)
    public static final class SubInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tsub";

        SubInstructionOp(InstructionDef<OperatorInstruction> def) {
            this(def.instruction().typeKind());
        }

        SubInstructionOp(SubInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SubInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new SubInstructionOp(this, cc);
        }

        SubInstructionOp(TypeKind type) {
            super(NAME, type);

            getOpcode(type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.operatorInstruction(getOpcode(type()));
        }

        private static Opcode getOpcode(TypeKind t) {
            return switch (t) {
                case IntType -> Opcode.ISUB;
                case LongType -> Opcode.LSUB;
                case FloatType -> Opcode.FSUB;
                case DoubleType -> Opcode.DSUB;
                default -> throw new IllegalArgumentException("Bad type: " + t);
            };
        }
    }

    public static SubInstructionOp sub(TypeKind type) {
        return new SubInstructionOp(type);
    }

    @Opcodes({Opcode.IREM, Opcode.LREM, Opcode.FREM, Opcode.DREM})
    @OpDeclaration(RemInstructionOp.NAME)
    public static final class RemInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Trem";

        RemInstructionOp(InstructionDef<OperatorInstruction> def) {
            this(def.instruction().typeKind());
        }

        RemInstructionOp(RemInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public RemInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new RemInstructionOp(this, cc);
        }

        RemInstructionOp(TypeKind type) {
            super(NAME, type);

            getOpcode(type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.operatorInstruction(getOpcode(type()));
        }

        private static Opcode getOpcode(TypeKind t) {
            return switch (t) {
                case IntType -> Opcode.IREM;
                case LongType -> Opcode.LREM;
                case FloatType -> Opcode.FREM;
                case DoubleType -> Opcode.DREM;
                default -> throw new IllegalArgumentException("Bad type: " + t);
            };
        }
    }

    public static RemInstructionOp rem(TypeKind type) {
        return new RemInstructionOp(type);
    }

    @Opcodes(Opcode.IINC)
    @OpDeclaration(IIncInstructionOp.NAME)
    public static final class IIncInstructionOp extends InstructionOp {
        public static final String NAME = "iinc";

        public static final String ATTRIBUTE_INDEX = "index";

        public static final String ATTRIBUTE_INCR = "incr";

        final int slot;
        final int incr;

        IIncInstructionOp(InstructionDef<IncrementInstruction> def) {
            this(def.instruction.slot(), def.instruction.constant());
        }

        IIncInstructionOp(IIncInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.slot = that.slot;
            this.incr = that.incr;
        }

        @Override
        public IIncInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new IIncInstructionOp(this, cc);
        }

        IIncInstructionOp(int slot, int incr) {
            super(NAME);

            this.slot = slot;
            this.incr = incr;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_INDEX, slot);
            m.put(ATTRIBUTE_INCR, incr);
            return m;
        }

        public int index() {
            return slot;
        }

        public int incr() {
            return incr;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.incrementInstruction(slot, incr);
        }
    }

    public static IIncInstructionOp iinc(int slot, int incr) {
        return new IIncInstructionOp(slot, incr);
    }


    @Opcodes({Opcode.LCMP, Opcode.FCMPG, Opcode.DCMPG})
    @OpDeclaration(CmpInstructionOp.NAME)
    public static final class CmpInstructionOp extends TypedInstructionOp {
        public static final String NAME = "Tcmp";

        CmpInstructionOp(InstructionDef<OperatorInstruction> def) {
            this(def.instruction.typeKind());
        }

        CmpInstructionOp(CmpInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public CmpInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new CmpInstructionOp(this, cc);
        }

        CmpInstructionOp(TypeKind type) {
            super(NAME, type);

            getOpcode(type);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.operatorInstruction(getOpcode(type()));
        }

        private static Opcode getOpcode(TypeKind t) {
            return switch (t) {
                case LongType -> Opcode.LCMP;
                case FloatType -> Opcode.FCMPG; // FCMPL?
                case DoubleType -> Opcode.DCMPG; // DCMPL?
                default -> throw new InternalError("Should not reach here");
            };
        }
    }

    public static CmpInstructionOp cmp(TypeKind type) {
        return new CmpInstructionOp(type);
    }


    // Stack instructions

    @Opcodes(Opcode.DUP)
    @OpDeclaration(DupInstructionOp.NAME)
    public static final class DupInstructionOp extends InstructionOp {
        public static final String NAME = "dup";

        DupInstructionOp(InstructionDef<StackInstruction> def) {
            this();
        }

        DupInstructionOp(DupInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public DupInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new DupInstructionOp(this, cc);
        }

        DupInstructionOp() {
            super(NAME);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.dup();
        }
    }

    public static DupInstructionOp dup() {
        return new DupInstructionOp();
    }

    @Opcodes(Opcode.POP)
    @OpDeclaration(PopInstructionOp.NAME)
    public static final class PopInstructionOp extends InstructionOp {
        public static final String NAME = "pop";

        PopInstructionOp(InstructionDef<StackInstruction> def) {
            this();
        }

        PopInstructionOp(PopInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public PopInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new PopInstructionOp(this, cc);
        }

        PopInstructionOp() {
            super(NAME);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.pop();
        }
    }

    public static PopInstructionOp pop() {
        return new PopInstructionOp();
    }

    @Opcodes(Opcode.BIPUSH)
    @OpDeclaration(BipushInstructionOp.NAME)
    public static final class BipushInstructionOp extends InstructionOp {
        public static final String NAME = "bipush";

        public static final String ATTRIBUTE_VALUE = "value";

        final int value;

        BipushInstructionOp(InstructionDef<ConstantInstruction.ArgumentConstantInstruction> def) {
            this(def.instruction.constantValue());
        }

        BipushInstructionOp(BipushInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.value = that.value;
        }

        @Override
        public BipushInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new BipushInstructionOp(this, cc);
        }

        BipushInstructionOp(int value) {
            super(NAME);

            this.value = value;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_VALUE, value);
            return m;
        }

        public int value() {
            return value;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.bipush(value);
        }
    }

    public static BipushInstructionOp bipush(byte value) {
        return new BipushInstructionOp(value);
    }

    @Opcodes(Opcode.SIPUSH)
    @OpDeclaration(SipushInstructionOp.NAME)
    public static final class SipushInstructionOp extends InstructionOp {
        public static final String NAME = "sipush";

        public static final String ATTRIBUTE_VALUE = "value";

        final int value;

        SipushInstructionOp(InstructionDef<ConstantInstruction.ArgumentConstantInstruction> def) {
            this(def.instruction.constantValue());
        }

        SipushInstructionOp(SipushInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.value = that.value;
        }

        @Override
        public SipushInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new SipushInstructionOp(this, cc);
        }

        SipushInstructionOp(int value) {
            super(NAME);

            this.value = value;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_VALUE, value);
            return m;
        }

        public int value() {
            return value;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.sipush(value);
        }
    }

    public static SipushInstructionOp sipush(short value) {
        return new SipushInstructionOp(value);
    }

    // Reflective instructions

    public static abstract class ClassTypeInstructionOp extends InstructionOp {
        public static final String ATTRIBUTE_DESC = "desc";

        final TypeElement desc;

        ClassTypeInstructionOp(ClassTypeInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.desc = that.desc;
        }

        ClassTypeInstructionOp(String name, TypeElement desc) {
            super(name);

            this.desc = desc;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_DESC, desc);
            return m;
        }

        public TypeElement desc() {
            return desc;
        }
    }

    @Opcodes(Opcode.NEW)
    @OpDeclaration(NewInstructionOp.NAME)
    public static final class NewInstructionOp extends ClassTypeInstructionOp {
        public static final String NAME = "new";

        NewInstructionOp(InstructionDef<NewObjectInstruction> def) {
            this(JavaType.ofNominalDescriptor(def.instruction.className().asSymbol()));
        }

        NewInstructionOp(NewInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public NewInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new NewInstructionOp(this, cc);
        }

        NewInstructionOp(TypeElement desc) {
            super(NAME, desc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.new_(((JavaType) desc).toNominalDescriptor());
        }
    }

    public static NewInstructionOp _new(TypeElement desc) {
        return new NewInstructionOp(desc);
    }

    @Opcodes({Opcode.ANEWARRAY, Opcode.NEWARRAY})
    @OpDeclaration(NewArrayInstructionOp.NAME)
    public static final class NewArrayInstructionOp extends ClassTypeInstructionOp {
        public static final String NAME = "Tnewarray";

        NewArrayInstructionOp(InstructionDef<Instruction> def) {
            this(getType(def.instruction));
        }

        static TypeElement getType(Instruction instruction) {
            if (instruction instanceof NewPrimitiveArrayInstruction a) {
                return switch (a.typeKind()) {
                    case BooleanType -> JavaType.BOOLEAN;
                    case ByteType -> JavaType.BYTE;
                    case ShortType -> JavaType.SHORT;
                    case CharType -> JavaType.CHAR;
                    case IntType -> JavaType.INT;
                    case FloatType -> JavaType.FLOAT;
                    case LongType -> JavaType.LONG;
                    case DoubleType -> JavaType.DOUBLE;
                    default -> throw new IllegalArgumentException("Bad array component type: " + a.typeKind());
                };
            } else if (instruction instanceof NewReferenceArrayInstruction ra) {
                return JavaType.ofNominalDescriptor(ra.componentType().asSymbol());
            } else {
                throw new InternalError();
            }
        }

        NewArrayInstructionOp(NewArrayInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public NewArrayInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new NewArrayInstructionOp(this, cc);
        }

        NewArrayInstructionOp(TypeElement desc) {
            super(NAME, desc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            ClassDesc cd = ((JavaType )desc()).toNominalDescriptor();
            if (cd.isPrimitive()) {
                mv.newPrimitiveArrayInstruction(TypeKind.fromDescriptor(cd.descriptorString()));
            } else {
                mv.newReferenceArrayInstruction(cd);
            }
        }
    }

    public static NewArrayInstructionOp newarray(TypeElement desc) {
        return new NewArrayInstructionOp(desc);
    }

    @Opcodes(Opcode.MULTIANEWARRAY)
    @OpDeclaration(MultiNewArrayInstructionOp.NAME)
    public static final class MultiNewArrayInstructionOp extends ClassTypeInstructionOp {
        public static final String NAME = "multinewarray";

        public static final String ATTRIBUTE_DIMS = "dims";

        final int dims;

        MultiNewArrayInstructionOp(InstructionDef<NewMultiArrayInstruction> def) {
            this(JavaType.ofNominalDescriptor(def.instruction().arrayType().asSymbol()), def.instruction().dimensions());
        }

        MultiNewArrayInstructionOp(MultiNewArrayInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.dims = that.dims;
        }

        @Override
        public MultiNewArrayInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new MultiNewArrayInstructionOp(this, cc);
        }

        MultiNewArrayInstructionOp(TypeElement desc, int dims) {
            super(NAME, desc);

            this.dims = dims;
        }

        public int dims() {
            return dims;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = super._attributes();
            m.put(ATTRIBUTE_DIMS, dims);
            return m;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.multianewarray(((JavaType )desc()).toNominalDescriptor(), dims);
        }
    }

    public static MultiNewArrayInstructionOp multinewarray(TypeElement desc, int dims) {
        return new MultiNewArrayInstructionOp(desc, dims);
    }

    @Opcodes(Opcode.INSTANCEOF)
    @OpDeclaration(InstanceOfInstructionOp.NAME)
    public static final class InstanceOfInstructionOp extends ClassTypeInstructionOp {
        public static final String NAME = "instanceof";

        InstanceOfInstructionOp(InstructionDef<TypeCheckInstruction> def) {
            this(JavaType.ofNominalDescriptor(def.instruction().type().asSymbol()));
        }

        InstanceOfInstructionOp(InstanceOfInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public InstanceOfInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new InstanceOfInstructionOp(this, cc);
        }

        InstanceOfInstructionOp(TypeElement desc) {
            super(NAME, desc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.instanceof_(((JavaType) desc()).toNominalDescriptor());
        }
    }

    public static InstanceOfInstructionOp instanceOf(TypeElement desc) {
        return new InstanceOfInstructionOp(desc);
    }

    @Opcodes(Opcode.CHECKCAST)
    @OpDeclaration(CheckCastInstructionOp.NAME)
    public static final class CheckCastInstructionOp extends ClassTypeInstructionOp {
        public static final String NAME = "checkcast";

        CheckCastInstructionOp(InstructionDef<TypeCheckInstruction> def) {
            this(JavaType.ofNominalDescriptor(def.instruction().type().asSymbol()));
        }

        CheckCastInstructionOp(CheckCastInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public CheckCastInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new CheckCastInstructionOp(this, cc);
        }

        CheckCastInstructionOp(TypeElement desc) {
            super(NAME, desc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.checkcast(((JavaType) desc()).toNominalDescriptor());
        }
    }

    public static CheckCastInstructionOp checkCast(TypeElement desc) {
        return new CheckCastInstructionOp(desc);
    }

    enum FieldKind {
        STATIC, INSTANCE,
    }

    public static abstract class FieldInstructionOp extends InstructionOp {
        public static final String ATTRIBUTE_KIND = "kind";
        public static final String ATTRIBUTE_DESC = "desc";

        final FieldKind kind;
        final FieldDesc desc;

        FieldInstructionOp(FieldInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.kind = that.kind;
            this.desc = that.desc;
        }

        FieldInstructionOp(String name, FieldKind kind, FieldDesc desc) {
            super(name);

            this.kind = kind;
            this.desc = desc;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_KIND, kind);
            m.put(ATTRIBUTE_DESC, desc);
            return m;
        }

        public FieldKind kind() {
            return kind;
        }

        public FieldDesc desc() {
            return desc;
        }

        static FieldDesc getFieldDesc(FieldRefEntry node) {
            return FieldDesc.field(
                    JavaType.ofNominalDescriptor(node.owner().asSymbol()),
                    node.name().stringValue(),
                    JavaType.ofNominalDescriptorString(node.type().stringValue()));
        }
    }

    @Opcodes({Opcode.GETFIELD, Opcode.GETSTATIC})
    @OpDeclaration(GetFieldInstructionOp.NAME)
    public static final class GetFieldInstructionOp extends FieldInstructionOp {
        public static final String NAME = "getfield";

        GetFieldInstructionOp(InstructionDef<FieldInstruction> def) {
            this(getFieldKind(def.opcode()), getFieldDesc(def.instruction().field()));
        }

        GetFieldInstructionOp(GetFieldInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public GetFieldInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new GetFieldInstructionOp(this, cc);
        }

        GetFieldInstructionOp(FieldKind kind, FieldDesc desc) {
            super(NAME, kind, desc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            FieldDesc desc = desc();
            mv.fieldInstruction(getOpcode(kind),
                    ((JavaType) desc.refType()).toNominalDescriptor(),
                    desc.name(),
                    ((JavaType) desc.type()).toNominalDescriptor());
        }

        private static Opcode getOpcode(FieldKind kind) {
            return switch (kind) {
                case STATIC -> Opcode.GETSTATIC;
                case INSTANCE -> Opcode.GETFIELD;
            };
        }

        private static FieldKind getFieldKind(Opcode opcode) {
            return switch (opcode) {
                case GETSTATIC -> FieldKind.STATIC;
                case GETFIELD -> FieldKind.INSTANCE;
                default -> throw new InternalError();
            };
        }
    }

    public static GetFieldInstructionOp getField(FieldKind kind, FieldDesc desc) {
        return new GetFieldInstructionOp(kind, desc);
    }

    @Opcodes({Opcode.PUTFIELD, Opcode.PUTSTATIC})
    @OpDeclaration(PutFieldInstructionOp.NAME)
    public static final class PutFieldInstructionOp extends FieldInstructionOp {
        public static final String NAME = "putfield";

        PutFieldInstructionOp(InstructionDef<FieldInstruction> def) {
            this(getFieldKind(def.opcode()), getFieldDesc(def.instruction().field()));
        }

        PutFieldInstructionOp(PutFieldInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public PutFieldInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new PutFieldInstructionOp(this, cc);
        }

        PutFieldInstructionOp(FieldKind kind, FieldDesc desc) {
            super(NAME, kind, desc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            FieldDesc desc = desc();
            mv.fieldInstruction(getOpcode(kind),
                    ((JavaType) desc.refType()).toNominalDescriptor(),
                    desc.name(),
                    ((JavaType) desc.type()).toNominalDescriptor());
        }

        private static Opcode getOpcode(FieldKind kind) {
            return switch (kind) {
                case STATIC -> Opcode.PUTSTATIC;
                case INSTANCE -> Opcode.PUTFIELD;
            };
        }

        private static FieldKind getFieldKind(Opcode opcode) {
            return switch (opcode) {
                case PUTSTATIC -> FieldKind.STATIC;
                case PUTFIELD -> FieldKind.INSTANCE;
                default -> throw new InternalError();
            };
        }
    }

    public static PutFieldInstructionOp putField(FieldKind kind, FieldDesc desc) {
        return new PutFieldInstructionOp(kind, desc);
    }

    enum InvokeKind {
        STATIC, VIRTUAL, INTERFACE, SPECIAL,
    }

    // @@@ static/virtual/special invocation on interfaces
    @Opcodes({Opcode.INVOKESTATIC, Opcode.INVOKEVIRTUAL, Opcode.INVOKEINTERFACE, Opcode.INVOKESPECIAL})
    @OpDeclaration(InvokeInstructionOp.NAME)
    public static final class InvokeInstructionOp extends InstructionOp {
        public static final String NAME = "invoke";

        public static final String ATTRIBUTE_KIND = "kind";
        public static final String ATTRIBUTE_DESC = "desc";
        public static final String ATTRIBUTE_IFACE = "iface";

        final InvokeKind kind;
        final MethodDesc desc;
        final boolean iface;

        InvokeInstructionOp(InstructionDef<InvokeInstruction> def) {
            this(getInvokeKind(def.opcode()), getMethodDesc(def.instruction().method()), def.instruction().isInterface());
        }

        InvokeInstructionOp(InvokeInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.kind = that.kind;
            this.desc = that.desc;
            this.iface = that.iface;
        }

        @Override
        public InvokeInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new InvokeInstructionOp(this, cc);
        }

        InvokeInstructionOp(InvokeKind kind, MethodDesc desc, boolean iface) {
            super(NAME);

            this.kind = kind;
            this.desc = desc;
            this.iface = iface;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_KIND, kind);
            m.put(ATTRIBUTE_DESC, desc);
            m.put(ATTRIBUTE_IFACE, iface);
            return m;
        }

        public InvokeKind kind() {
            return kind;
        }

        public MethodDesc desc() {
            return desc;
        }

        public boolean iface() {
            return iface;
        }

        public MethodTypeDesc callOpDescriptor() {
            return switch (kind) {
                case STATIC -> desc.type();
                case VIRTUAL, INTERFACE, SPECIAL -> {
                    List<TypeElement> params = new ArrayList<>();
                    params.add(desc.refType());
                    params.addAll(desc.type().parameters());
                    yield MethodTypeDesc.methodType(desc.type().returnType(), params);
                }
            };
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            MethodDesc desc = desc();

            // @@@ interfaces
            mv.invokeInstruction(
                    getOpcode(kind()),
                    ((JavaType) desc.refType()).toNominalDescriptor(),
                    desc.name(),
                    desc.type().toNominalDescriptor(),
                    iface()
            );
        }

        private static Opcode getOpcode(InvokeKind kind) {
            return switch (kind) {
                case STATIC -> Opcode.INVOKESTATIC;
                case VIRTUAL -> Opcode.INVOKEVIRTUAL;
                case INTERFACE -> Opcode.INVOKEINTERFACE;
                case SPECIAL -> Opcode.INVOKESPECIAL;
            };
        }

        private static InvokeKind getInvokeKind(Opcode opcode) {
            return switch (opcode) {
                case INVOKESTATIC -> InvokeKind.STATIC;
                case INVOKEVIRTUAL -> InvokeKind.VIRTUAL;
                case INVOKEINTERFACE -> InvokeKind.INTERFACE;
                case INVOKESPECIAL -> InvokeKind.SPECIAL;
                default -> throw new InternalError();
            };
        }

        private static MethodDesc getMethodDesc(MemberRefEntry node) {
            return MethodDesc.method(
                    JavaType.ofNominalDescriptor(node.owner().asSymbol()),
                    node.name().stringValue(),
                    MethodTypeDesc.ofNominalDescriptor(java.lang.constant.MethodTypeDesc.ofDescriptor(node.type().stringValue())));
        }
    }

    public static InvokeInstructionOp invoke(InvokeKind kind, MethodDesc desc) {
        return new InvokeInstructionOp(kind, desc, false);
    }

    public static InvokeInstructionOp invoke(InvokeKind kind, MethodDesc desc, boolean isInterface) {
        return new InvokeInstructionOp(kind, desc, isInterface);
    }

    // Terminating instructions

    @Opcodes(Opcode.GOTO)
    @OpDeclaration(GotoInstructionOp.NAME)
    public static final class GotoInstructionOp extends TerminatingInstructionOp implements Op.BlockTerminating {
        public static final String NAME = "goto";

        GotoInstructionOp(InstructionDef<BranchInstruction> def) {
            this(def.successors.get(0));
        }

        GotoInstructionOp(GotoInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public GotoInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new GotoInstructionOp(this, cc);
        }

        GotoInstructionOp(Block.Reference t) {
            super(NAME, List.of(t));

            if (!t.arguments().isEmpty()) {
                throw new IllegalArgumentException();
            }
        }

        public Block targetBranch() {
            return successors().get(0).targetBlock();
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.goto_(c.getLabel(targetBranch()));
        }
    }

    public static GotoInstructionOp _goto(Block.Reference t) {
        return new GotoInstructionOp(t);
    }

    enum Comparison {
        EQ("NE"),
        NE("EQ"),
        LT("GE"),
        GE("LT"),
        GT("LE"),
        LE("GT");

        private final String inverseName;

        Comparison(String inverseName) {
            this.inverseName = inverseName;
        }

        public Comparison inverse() {
            return Comparison.valueOf(inverseName);
        }
    }

    @Opcodes({Opcode.IF_ACMPEQ, Opcode.IF_ACMPNE,
            Opcode.IF_ICMPEQ, Opcode.IF_ICMPNE, Opcode.IF_ICMPLT, Opcode.IF_ICMPGE, Opcode.IF_ICMPGT, Opcode.IF_ICMPLE
    })
    @OpDeclaration(IfcmpInstructionOp.NAME)
    public static final class IfcmpInstructionOp extends TypedTerminatingInstructionOp implements Op.BlockTerminating {
        public static final String NAME = "if_TcmpC";

        public static final String ATTRIBUTE_COND = "cond";

        final Comparison cond;

        IfcmpInstructionOp(InstructionDef<BranchInstruction> def) {
            this(def.instruction().opcode().primaryTypeKind(), getComparison(def.opcode()),
                    def.successors.get(0), def.successors.get(1));
        }

        IfcmpInstructionOp(IfcmpInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.cond = that.cond;
        }

        @Override
        public IfcmpInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new IfcmpInstructionOp(this, cc);
        }

        IfcmpInstructionOp(TypeKind type, Comparison c, Block.Reference t, Block.Reference f) {
            // Ensure successor order is false branch, then true branch, for correct topological ordering
            super(NAME, type, List.of(f, t));

            if (type != TypeKind.IntType && type != TypeKind.ReferenceType) {
                throw new IllegalArgumentException("Unsupported type: " + type);
            }

            if (type == TypeKind.ReferenceType) {
                if (c != Comparison.EQ && c != Comparison.NE) {
                    throw new IllegalArgumentException("Unsupported condition for reference (A) type: " + c);
                }
            }

            if (!t.arguments().isEmpty()) {
                throw new IllegalArgumentException();
            }

            if (!f.arguments().isEmpty()) {
                throw new IllegalArgumentException();
            }

            this.cond = c;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = super._attributes();
            m.put(ATTRIBUTE_COND, cond);
            return m;
        }

        public Comparison cond() {
            return cond;
        }

        public Block trueBranch() {
            return successors().get(1).targetBlock();
        }

        public Block falseBranch() {
            return successors().get(0).targetBlock();
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            // False branch must be immediately after (in sequence) to this block
            mv.branchInstruction(getOpcode(type(), cond()), c.getLabel(trueBranch()));
        }

        private static Opcode getOpcode(TypeKind t, Comparison c) {
            return switch (t) {
                case ReferenceType -> switch (c) {
                    case EQ -> Opcode.IF_ACMPEQ;
                    case NE -> Opcode.IF_ACMPNE;
                    default -> throw new InternalError("Should not reach here");
                };
                case IntType -> switch (c) {
                    case EQ -> Opcode.IF_ICMPEQ;
                    case NE -> Opcode.IF_ICMPNE;
                    case LT -> Opcode.IF_ICMPLT;
                    case GE -> Opcode.IF_ICMPGE;
                    case GT -> Opcode.IF_ICMPGT;
                    case LE -> Opcode.IF_ICMPLE;
                };
                default -> throw new InternalError("Should not reach here");
            };
        }

        private static Comparison getComparison(Opcode opcode) {
            return switch (opcode) {
                case IF_ACMPEQ -> Comparison.EQ;
                case IF_ACMPNE -> Comparison.NE;
                case IF_ICMPEQ -> Comparison.EQ;
                case IF_ICMPNE -> Comparison.NE;
                case IF_ICMPLT -> Comparison.LT;
                case IF_ICMPGE -> Comparison.GE;
                case IF_ICMPGT -> Comparison.GT;
                case IF_ICMPLE -> Comparison.LE;
                default -> throw new InternalError("Should not reach here");
            };
        }
    }

    public static IfcmpInstructionOp if_cmp(TypeKind type, Comparison c, Block.Reference t, Block.Reference f) {
        return new IfcmpInstructionOp(type, c, t, f);
    }

    @Opcodes({Opcode.IFEQ, Opcode.IFNE,
            Opcode.IFLT, Opcode.IFGE, Opcode.IFGT, Opcode.IFLE
    })
    @OpDeclaration(IfInstructionOp.NAME)
    public static final class IfInstructionOp extends TerminatingInstructionOp implements Op.BlockTerminating {
        public static final String NAME = "ifC";

        public static final String ATTRIBUTE_COND = "cond";

        final Comparison cond;

        IfInstructionOp(InstructionDef<BranchInstruction> def) {
            this(getComparison(def.opcode()), def.successors.get(0), def.successors.get(1));
        }

        IfInstructionOp(IfInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.cond = that.cond;
        }

        IfInstructionOp(Comparison c, Block.Reference t, Block.Reference f) {
            // Ensure successor order is false branch, then true branch, for correct topological ordering
            super(NAME, List.of(f, t));

            if (!t.arguments().isEmpty()) {
                throw new IllegalArgumentException();
            }

            if (!f.arguments().isEmpty()) {
                throw new IllegalArgumentException();
            }

            this.cond = c;
        }

        @Override
        public IfInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new IfInstructionOp(this, cc);
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_COND, cond);
            return m;
        }

        public Comparison cond() {
            return cond;
        }

        public Block trueBranch() {
            return successors().get(1).targetBlock();
        }

        public Block falseBranch() {
            return successors().get(0).targetBlock();
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            // False branch must be immediately after (in sequence) to this block
            mv.branchInstruction(getOpcode(cond()), c.getLabel(trueBranch()));
        }

        private static Opcode getOpcode(Comparison c) {
            return switch (c) {
                case EQ -> Opcode.IFEQ;
                case NE -> Opcode.IFNE;
                case LT -> Opcode.IFLT;
                case GE -> Opcode.IFGE;
                case GT -> Opcode.IFGT;
                case LE -> Opcode.IFLE;
            };
        }

        private static Comparison getComparison(Opcode opcode) {
            return switch (opcode) {
                case IFEQ -> Comparison.EQ;
                case IFNE -> Comparison.NE;
                case IFLT -> Comparison.LT;
                case IFGE -> Comparison.GE;
                case IFGT -> Comparison.GT;
                case IFLE -> Comparison.LE;
                default -> throw new InternalError();
            };
        }
    }

    public static IfInstructionOp _if(Comparison c, Block.Reference t, Block.Reference f) {
        return new IfInstructionOp(c, t, f);
    }

    @Opcodes({Opcode.ARETURN, Opcode.IRETURN, Opcode.LRETURN, Opcode.FRETURN, Opcode.DRETURN})
    @OpDeclaration(ReturnInstructionOp.NAME)
    public static final class ReturnInstructionOp extends TypedTerminatingInstructionOp implements Op.BodyTerminating {
        public static final String NAME = "Treturn";

        ReturnInstructionOp(InstructionDef<ReturnInstruction> def) {
            this(def.instruction().typeKind());
        }

        ReturnInstructionOp(ReturnInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        ReturnInstructionOp(TypeKind type) {
            super(NAME, type);
        }

        @Override
        public ReturnInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new ReturnInstructionOp(this, cc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.returnInstruction(type());
        }
    }

    @Opcodes(Opcode.RETURN)
    @OpDeclaration(VoidReturnInstructionOp.NAME)
    public static final class VoidReturnInstructionOp extends TerminatingInstructionOp implements Op.BodyTerminating {
        public static final String NAME = "return";

        VoidReturnInstructionOp(InstructionDef<ReturnInstruction> def) {
            this();
        }

        VoidReturnInstructionOp(VoidReturnInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        VoidReturnInstructionOp() {
            super(NAME);
        }

        @Override
        public VoidReturnInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new VoidReturnInstructionOp(this, cc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.return_();
        }
    }

    public static ReturnInstructionOp _return(TypeKind type) {
        return new ReturnInstructionOp(type);
    }

    public static VoidReturnInstructionOp _return() {
        return new VoidReturnInstructionOp();
    }

    @Opcodes(Opcode.ATHROW)
    @OpDeclaration(AthrowInstructionOp.NAME)
    public static final class AthrowInstructionOp extends TerminatingInstructionOp implements Op.BodyTerminating {
        public static final String NAME = "athrow";

        AthrowInstructionOp(InstructionDef<ThrowInstruction> def) {
            this();
        }

        AthrowInstructionOp(AthrowInstructionOp that, CopyContext cc) {
            super(that, cc);
        }

        AthrowInstructionOp() {
            super(NAME);
        }

        @Override
        public AthrowInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new AthrowInstructionOp(this, cc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.athrow();
        }
    }

    static public AthrowInstructionOp athrow() {
        return new AthrowInstructionOp();
    }

    @Opcodes(Opcode.TABLESWITCH)
    @OpDeclaration(TableswitchInstructionOp.NAME)
    public static final class TableswitchInstructionOp extends TerminatingInstructionOp implements Op.BlockTerminating {
        public static final String NAME = "tableswitch";

        public static final String ATTRIBUTE_LOW = "low";
        public static final String ATTRIBUTE_HIGH = "high";

        final int low;
        final int high;

        TableswitchInstructionOp(InstructionDef<TableSwitchInstruction> def) {
            this(def.instruction().lowValue(), def.instruction().highValue(), def.successors);
        }

        TableswitchInstructionOp(TableswitchInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.low = that.low;
            this.high = that.high;
        }

        TableswitchInstructionOp(int low, int high, List<Block.Reference> successors) {
            super(NAME, successors);

            if (low > high) {
                throw new IllegalArgumentException();
            }

            if (high - low + 1 != successors.size()) {
                throw new IllegalArgumentException();
            }

            this.low = low;
            this.high = high;
        }

        @Override
        public TableswitchInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new TableswitchInstructionOp(this, cc);
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_LOW, low);
            m.put(ATTRIBUTE_HIGH, high);
            return m;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.tableswitch(low, high, c.getLabel(successors().get(0).targetBlock()), getSwitchCases(c));
        }

        private List<SwitchCase> getSwitchCases(MethodVisitorContext c) {
            List<SwitchCase> cases = new ArrayList<>();
            int caseValue = low;
            for (int i = 1; i < successors.size(); i++) {
                cases.add(SwitchCase.of(caseValue++, c.getLabel(successors.get(i))));
            }
            return cases;
        }
    }

    static public TableswitchInstructionOp tableswitch(int min, int max, List<Block.Reference> successors) {
        return new TableswitchInstructionOp(min, max, successors);
    }

    @Opcodes(Opcode.LOOKUPSWITCH)
    @OpDeclaration(LookupswitchInstructionOp.NAME)
    public static final class LookupswitchInstructionOp extends TerminatingInstructionOp implements Op.BlockTerminating {
        public static final String NAME = "lookupswitch";

        public static final String ATTRIBUTE_KEYS = "keys";

        final List<Integer> keys;

        LookupswitchInstructionOp(InstructionDef<LookupSwitchInstruction> def) {
            this(getKeys(def.instruction().cases()), def.successors);
        }

        LookupswitchInstructionOp(LookupswitchInstructionOp that, CopyContext cc) {
            super(that, cc);

            this.keys = that.keys;
        }

        LookupswitchInstructionOp(List<Integer> keys, List<Block.Reference> successors) {
            super(NAME, successors);

            if (keys.size() != successors.size() - 1) {
                throw new IllegalArgumentException("Number of keys must be one less than number of successors");
            }

            this.keys = List.copyOf(keys);
        }

        @Override
        public LookupswitchInstructionOp transform(CopyContext cc, OpTransformer ot) {
            return new LookupswitchInstructionOp(this, cc);
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            m.put(ATTRIBUTE_KEYS, keys.toString());
            return m;
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            mv.lookupswitch(c.getLabel(successors().get(0).targetBlock()), getSwitchCases(c));
        }

        private static List<Integer> getKeys(List<SwitchCase> cases) {
            return cases.stream().map(SwitchCase::caseValue).toList();
        }

        private List<SwitchCase> getSwitchCases(MethodVisitorContext c) {
            List<SwitchCase> cases = new ArrayList<>();
            for (int i = 1; i < successors.size(); i++) {
                cases.add(SwitchCase.of(keys.get(i - 1), c.getLabel(successors.get(i))));
            }
            return cases;
        }
    }

    static public LookupswitchInstructionOp lookupswitch(List<Integer> keys, List<Block.Reference> successors) {
        return new LookupswitchInstructionOp(keys, successors);
    }

    // Internal control operations

    public static abstract class ControlInstructionOp extends Op {
        private final TypeElement resultType;

        ControlInstructionOp(ControlInstructionOp that, CopyContext cc) {
            super(that, cc);
            this.resultType = that.resultType;
        }

        ControlInstructionOp(String name, TypeElement resultType, List<Value> operands) {
            super(name, operands);
            this.resultType = resultType;
        }

        @Override
        public final Map<String, Object> attributes() {
            Map<String, Object> m = _attributes();
            return m.isEmpty() ? m : Collections.unmodifiableMap(m);
        }

        Map<String, Object> _attributes() {
            return Map.of();
        }

        public abstract void apply(CodeBuilder mv, MethodVisitorContext c);

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    enum PrimitiveFrameType {
        TOP, INTEGER, FLOAT, DOUBLE, LONG, NULL, UNINITIALIZED_THIS
    }

    public static class Frame extends ControlInstructionOp {
        public static final String NAME = "frame";

        public static final String ATTRIBUTE_TYPE = "type";
        public static final String ATTRIBUTE_LOCAL = "local";
        public static final String ATTRIBUTE_STACK = "stack";

        final StackMapFrameInfo node;

        Frame(Frame that, CopyContext cc) {
            super(that, cc);

            this.node = that.node;
        }

        @Override
        public Frame transform(CopyContext cc, OpTransformer ot) {
            return new Frame(this, cc);
        }

        Frame(StackMapFrameInfo node) {
            super(NAME, JavaType.VOID, List.of());

            this.node = node;
        }

        @Override
        Map<String, Object> _attributes() {
            Map<String, Object> m = new HashMap<>();
            // @@@ Convert local/stack elements to types
            m.put(ATTRIBUTE_TYPE, node.frameType());
            m.put(ATTRIBUTE_LOCAL, node.locals().toString());
            m.put(ATTRIBUTE_STACK, node.stack().toString());
            return m;
        }

        public void apply(CodeBuilder mv, MethodVisitorContext c) {
        }

        public boolean hasOperandStackElements() {
            return !node.stack().isEmpty();
        }

        public List<TypeElement> operandStackTypes() {
            List<TypeElement> stackTypes = new ArrayList<>();
            for (StackMapFrameInfo.VerificationTypeInfo ost : node.stack()) {
                if (ost instanceof StackMapFrameInfo.SimpleVerificationTypeInfo i) {
                    switch (i) {
                        case ITEM_TOP -> {
                            // @@@
                            stackTypes.add(JavaType.J_L_OBJECT);
                        }
                        case ITEM_INTEGER -> {
                            stackTypes.add(JavaType.INT);
                        }
                        case ITEM_FLOAT -> {
                            stackTypes.add(JavaType.FLOAT);
                        }
                        case ITEM_DOUBLE -> {
                            stackTypes.add(JavaType.DOUBLE);
                        }
                        case ITEM_LONG -> {
                            stackTypes.add(JavaType.LONG);
                        }
                        case ITEM_NULL -> {
                            // @@@
                            stackTypes.add(JavaType.J_L_OBJECT);
                        }
                        case ITEM_UNINITIALIZED_THIS -> {
                            // @@@
                            stackTypes.add(JavaType.J_L_OBJECT);
                        }
                    }
                } else if (ost instanceof StackMapFrameInfo.ObjectVerificationTypeInfo i) {
                    stackTypes.add(JavaType.ofNominalDescriptor(i.classSymbol()));
                } else if (ost instanceof StackMapFrameInfo.UninitializedVerificationTypeInfo i) {
                    // @@@
                    // label designates the NEW instruction that created the uninitialized value
                }
            }
            return stackTypes;
        }
    }

    public static Frame frame(StackMapFrameInfo node) {
        return new Frame(node);
    }


    public static final class ExceptionTableStart extends ControlInstructionOp implements Op.BlockTerminating {
        public static final String NAME = "exceptionTableStart";

        // First successor is the non-exceptional successor whose target indicates
        // the first block in the exception region.
        // One or more subsequent successors target the exception catching blocks
        // each of which have one block argument whose type is an exception type,
        // or no block argument for the finally block (that occurs last)
        final List<Block.Reference> s;

        ExceptionTableStart(ExceptionTableStart that, CopyContext cc) {
            super(that, cc);

            this.s = that.s.stream().map(cc::getSuccessorOrCreate).toList();
        }

        @Override
        public ExceptionTableStart transform(CopyContext cc, OpTransformer ot) {
            return new ExceptionTableStart(this, cc);
        }

        ExceptionTableStart(List<Block.Reference> s) {
            super(NAME, JavaType.VOID, List.of());

            if (s.size() < 2) {
                throw new IllegalArgumentException("Operation must have two or more successors" + opName());
            }

            this.s = List.copyOf(s);
        }

        @Override
        public List<Block.Reference> successors() {
            return s;
        }

        public Block.Reference start() {
            return s.get(0);
        }

        public List<Block.Reference> catchBlocks() {
            return s.subList(1, s.size());
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            Label start = c.getLabel(result());
            c.exceptionRegionStack().push(this);
            mv.labelBinding(start);
        }
    }

    public static ExceptionTableStart exceptionTableStart(Block.Reference start, Block.Reference... catchers) {
        return exceptionTableStart(start, List.of(catchers));
    }

    public static ExceptionTableStart exceptionTableStart(Block.Reference start, List<Block.Reference> catchers) {
        List<Block.Reference> s = new ArrayList<>();
        s.add(start);
        s.addAll(catchers);
        return new ExceptionTableStart(s);
    }

    public static final class ExceptionTableEnd extends ControlInstructionOp {
        public static final String NAME = "exceptionTableEnd";

        ExceptionTableEnd(ExceptionTableEnd that, CopyContext cc) {
            super(that, cc);
        }

        ExceptionTableEnd() {
            super(NAME, JavaType.VOID, List.of());
        }

        @Override
        public ExceptionTableEnd transform(CopyContext cc, OpTransformer ot) {
            return new ExceptionTableEnd(this, cc);
        }

        @Override
        public void apply(CodeBuilder mv, MethodVisitorContext c) {
            ExceptionTableStart er = c.exceptionRegionStack().pop();
            Label start = c.getLabel(er.result());
            Label end = c.getLabel(er);
            mv.labelBinding(end);
            for (Block.Reference catchBlockSuccessor : er.catchBlocks()) {
                Block catchBlock = catchBlockSuccessor.targetBlock();
                Label handle = c.getLabel(catchBlock);

                if (!catchBlock.parameters().isEmpty()) {
                    ClassDesc type = ((JavaType) catchBlock.parameters().get(0).type()).toNominalDescriptor();
                    mv.exceptionCatch(start, end, handle, type);
                } else {
                    mv.exceptionCatchAll(start, end, handle);
                }
            }
        }
    }

    public static ExceptionTableEnd exceptionTableEnd() {
        return new ExceptionTableEnd();
    }


    // Opcode factory creation

    public static InstructionOp create(InstructionDef<? extends Instruction> def) {
        MethodHandle mh = INSTRUCTION_FACTORY[def.opcode().bytecode()];
        if (mh == null) {
            throw new UnsupportedOperationException("Instruction unsupported, opcode = '" + def.opcode());
        }
        try {
            return (InstructionOp) mh.invoke(def);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    static final MethodHandle[] INSTRUCTION_FACTORY = createInstructionMapping();

    static MethodHandle[] createInstructionMapping() {
        MethodHandle[] instructionFactory = new MethodHandle[ClassFile.GOTO_W + 1];

        for (Class<?> opClass : BytecodeInstructionOps.class.getNestMembers()) {
            if (opClass.isAnnotationPresent(Opcodes.class)) {
                if (!Modifier.isPublic(opClass.getModifiers())) {
                    throw new InternalError("Operation class not public: " + opClass.getName());
                }

                if (!InstructionOp.class.isAssignableFrom(opClass)) {
                    throw new InternalError("Operation class is not assignable to Instruction: " + opClass);
                }

                MethodHandle handle = getOpcodeConstructorMethodHandle(opClass);
                if (handle == null) {
                    throw new InternalError("Operation constructor for operation class not found: " + opClass.getName());
                }

                if (!InstructionOp.class.isAssignableFrom(handle.type().returnType())) {
                    throw new InternalError("Operation constructor does not return an Op: " + handle);
                }

                Opcode[] opcodes = opClass.getAnnotation(Opcodes.class).value();
                for (Opcode opcode : opcodes) {
                    if (instructionFactory[opcode.bytecode()] != null) {
                        throw new InternalError("Opcode already assigned to " + instructionFactory[opcode.bytecode()]);
                    }
                    instructionFactory[opcode.bytecode()] = handle;
                }
            }
        }

        return instructionFactory;
    }

    static MethodHandle getOpcodeConstructorMethodHandle(Class<?> opClass) {
        Optional<Constructor<?>> oc = Stream.of(opClass.getDeclaredConstructors())
                .filter(c -> c.getParameterCount() == 1)
                .filter(c -> InstructionDef.class.isAssignableFrom(c.getParameterTypes()[0]))
                .findFirst();
        Constructor<?> constructor = oc.orElse(null);
        if (constructor == null) {
            return null;
        }

        try {
            return MethodHandles.lookup().unreflectConstructor(constructor);
        } catch (IllegalAccessException e) {
            throw new InternalError("Inaccessible operation constructor for operation: " +
                    constructor);
        }
    }
}
