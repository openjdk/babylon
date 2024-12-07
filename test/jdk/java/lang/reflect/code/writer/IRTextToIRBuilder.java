import java.io.IOException;
import java.io.PrintStream;
import java.lang.classfile.*;
import java.lang.classfile.components.ClassPrinter;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.instruction.*;
import java.lang.constant.ConstantDescs;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.parser.OpParser;
import java.lang.reflect.code.writer.OpBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

import static java.lang.reflect.code.op.CoreOp.FuncOp;

/*
* @test
* */

public class IRTextToIRBuilder {
    // TODO original cf size of text, new cf size of code builder
    // TODO rename + doc (do it after separation)
    // TODO zip cf
    // TODO can we reduce code builder ?

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.err.println("Usage: <program> <path_to_cf>");
            System.exit(-1);
        }
        var cf_path = Path.of(args[0]);
        var bytes = Files.readAllBytes(cf_path);
        var new_bytes = replaceOpFieldWithBuilderMethod(bytes);
        new PrintStream(System.out).write(new_bytes);
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
                var opFieldName = opFieldAndIR.opField().name().stringValue();
                var methodName = Utils.irBuilderName(opFieldName);
                byte[] bytes = BytecodeGenerator.generateClassData(MethodHandles.lookup(), methodName, builderOp);
                var builderMethod = ClassFile.of().parse(bytes).methods().stream()
                        .filter(mm -> mm.methodName().equalsString(methodName)).findFirst().orElseThrow();
                clb.with(builderMethod);
            }
        }));
        var cf = ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL);
        var newBytes = cf.transformClass(classModel, classTransform);
        return newBytes;
    }

    static void print(byte[] bytes) {
        print(ClassFile.of().parse(bytes));
    }

    static void print(ClassModel cm) {
        ClassPrinter.toYaml(cm, ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
    }
}
