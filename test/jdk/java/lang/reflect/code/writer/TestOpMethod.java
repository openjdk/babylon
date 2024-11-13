import java.io.IOException;
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

public class TestOpMethod {

//    We discussed replacing the mechanism to store models in class files.
//    Currently, we serialize to the textual form.
//    We want to explore replacing that with methods in class files that build and return models.
//    See the test TestCodeBuilder and familiarize yourself with that.
//    It's likely that transformation code is buggy because it has not been tested on a wide range of source.
//    It would also be interesting to get some size comparison between the two approaches.
//    To properly do this we will need to avoid the copying of code from java.base into jdk.compiler,
//    otherwise we need to copy more code and there are certain restrictions on what features the code can use.
//    Maurizio is looking into that.
//    There may be an interim solution before moving the code to an incubating model. We should discuss more in our meetings.
//    One way to explore sooner for experimentation purposes to write a class file transformer using the Classfile API
//    and rewrite the class file replacing code models in textual form with the corresponding methods.
//    That's more challenging when reflecting over lambda bodies, but should more feasible when reflecting over method bodies.

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        for (var arg : args) {
            var path = Path.of(arg);
            var originalBytes = Files.readAllBytes(path);
            var newBytes = TestOpMethod.replaceOpFieldWithBuilderMethod(originalBytes);

            System.out.printf("%s %d %d%n", arg, originalBytes.length, newBytes.length);
            // TODO add assertion
            // TODO pass path to classfile, no load
            // TODO name, before, after

            // TODO a script that runs the tool for many classes
            // TODO reduce size if possible
        }
    }

    static byte[] replaceOpFieldWithBuilderMethod(byte[] classData) {
        return replaceOpFieldWithBuilderMethod(ClassFile.of().parse(classData));
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
        var newBytes = ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL).transformClass(classModel, classTransform);
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

    record OpFieldAndIR(FieldRefEntry opField, String ir) {
    }
}
