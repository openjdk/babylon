import java.lang.classfile.*;
import java.lang.classfile.components.ClassPrinter;
import java.lang.classfile.constantpool.FieldRefEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.instruction.*;
import java.lang.constant.ConstantDescs;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.parser.OpParser;
import java.lang.reflect.code.type.*;
import java.lang.reflect.code.writer.OpBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static java.lang.reflect.code.op.CoreOp.FuncOp;

public class OpFieldToMethodBuilder {

    public static void main(String[] args) {
        for (var arg : args) {
            var path = Path.of(arg);
            byte[] originalBytes;
            byte[] newBytes;
            try {
                originalBytes = Files.readAllBytes(path);
                newBytes = OpFieldToMethodBuilder.replaceOpFieldWithBuilderMethod(originalBytes);
            } catch (Throwable e) {
                continue; // ignore errors for now
            }
            System.out.printf("%s %d %d%n", arg, originalBytes.length, newBytes.length);
            // TODO output useful info like avg size increase
            // TODO remove duplicate
            // TODO reduce size if possible (by reducing the code of the builder method)
        }
    }

    static byte[] replaceOpFieldWithBuilderMethod(byte[] classData) {
        return replaceOpFieldWithBuilderMethod(ClassFile.of().parse(classData));
    }

    record OpFieldAndIR(FieldRefEntry opField, String ir) {
    }

    static byte[] replaceOpFieldWithBuilderMethod(ClassModel classModel) {
        var opFieldsAndIRs = new ArrayList<OpFieldAndIR>();
        var classTransform = ClassTransform.dropping(e -> e instanceof FieldModel fm && fm.fieldName().stringValue().endsWith("$op")).andThen(
                ClassTransform.transformingMethods(mm -> mm.methodName().equalsString(ConstantDescs.CLASS_INIT_NAME), (mb, me) -> {
                    if (!(me instanceof CodeModel codeModel)) {
                        mb.with(me);
                        return;
                    }
                    mb.withCode(cob -> {
                        ConstantInstruction.LoadConstantInstruction ldc = null;
                        for (CodeElement e : codeModel) {
                            if (ldc != null && e instanceof FieldInstruction fi && fi.opcode() == Opcode.PUTSTATIC && fi.owner().equals(classModel.thisClass()) && fi.name().stringValue().endsWith("$op")) {
                                opFieldsAndIRs.add(new OpFieldAndIR(fi.field(), ((StringEntry) ldc.constantEntry()).stringValue()));
                                ldc = null;
                            } else {
                                if (ldc != null) {
                                    cob.with(ldc);
                                    ldc = null;
                                }
                                switch (e) {
                                    case ConstantInstruction.LoadConstantInstruction lci when lci.constantEntry() instanceof StringEntry ->
                                            ldc = lci;
                                    case LineNumber _, CharacterRange _, LocalVariable _, LocalVariableType _ -> {
                                    }
                                    default -> cob.with(e);
                                }
                            }
                        }
                    });
                })).andThen(ClassTransform.endHandler(clb -> {
            for (var opFieldAndIR : opFieldsAndIRs) {
                var funcOp = ((FuncOp) OpParser.fromStringOfFuncOp(opFieldAndIR.ir()));
                var builderOp = OpBuilder.createBuilderFunction(funcOp);
                testBuilderOp(builderOp, opFieldAndIR.ir());
                var opFieldName = opFieldAndIR.opField().name().stringValue();
                var methodName = builderMethodName(opFieldName);
                byte[] bytes = BytecodeGenerator.generateClassData(MethodHandles.lookup(), methodName, builderOp);
                var builderMethod = ClassFile.of().parse(bytes).methods().stream()
                        .filter(mm -> mm.methodName().equalsString(methodName)).findFirst().orElseThrow();
                clb.with(builderMethod);
            }
        }));
        var cf = ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL);
        var newBytes = cf.transformClass(classModel, classTransform);
        testBuilderMethods(newBytes, opFieldsAndIRs);
        return newBytes;
    }

    static void testBuilderOp(FuncOp builderOp, String expectedIR) {
        var op = (Op) Interpreter.invoke(builderOp, ExtendedOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY);
        assert expectedIR.equals(op.toText());
    }

    static void testBuilderMethods(byte[] classData, List<OpFieldAndIR> opFieldsAndIRs) {
        MethodHandles.Lookup lookup = null;
        try {
            lookup = MethodHandles.lookup().defineHiddenClass(classData, true);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
        for (var opFieldAndIR : opFieldsAndIRs) {
            var opFieldName = opFieldAndIR.opField().name().stringValue();
            var methodName = builderMethodName(opFieldName);
            var functionType = FunctionType.functionType(JavaType.type(Op.class), JavaType.type(OpFactory.class),
                    JavaType.type(TypeElementFactory.class));
            MethodHandle mh = null;
            try {
                mh = lookup.findStatic(lookup.lookupClass(),
                        methodName,
                        MethodRef.toNominalDescriptor(functionType).resolveConstantDesc(lookup));
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
            Op builtOp = null;
            try {
                builtOp = ((Op) mh.invoke(ExtendedOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY));
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
            assert builtOp.toText().equals(opFieldAndIR.ir());
        }
    }

    static String builderMethodName(String opFieldName) {
        // e.g. A::add(int, int)int$op ---> add(int, int)int$op
        return opFieldName.substring(opFieldName.indexOf(':') + 2);
    }

    static void print(byte[] bytes) {
        print(ClassFile.of().parse(bytes));
    }

    static void print(ClassModel cm) {
        ClassPrinter.toYaml(cm, ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
    }
}
