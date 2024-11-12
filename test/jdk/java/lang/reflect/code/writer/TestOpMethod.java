import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.IOException;
import java.lang.classfile.*;
import java.lang.classfile.components.ClassPrinter;
import java.lang.classfile.constantpool.FieldRefEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.instruction.*;
import java.lang.constant.ConstantDescs;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.parser.OpParser;
import java.lang.reflect.code.type.*;
import java.lang.reflect.code.writer.OpBuilder;
import java.util.ArrayList;
import java.util.List;

import static java.lang.reflect.code.op.CoreOp.FuncOp;

/*
 * @test
 * @enablePreview
 * @run testng TestOpMethod
 *
 */
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

    static ClassModel CLASS_MODEL;

    @BeforeClass
    static void setup() throws IOException {
        CLASS_MODEL = ClassFile.of().parse(IR.class.getResourceAsStream("IR.class").readAllBytes());
    }

    @DataProvider
    byte[][] classes() {
        try {
            return new byte[][]{
                    IR.class.getResourceAsStream("IR.class").readAllBytes()
            };
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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
                var methodName = opFieldAndIR.opField().name().stringValue().substring(opFieldName.indexOf(':') + 2);
                byte[] bytes = BytecodeGenerator.generateClassData(MethodHandles.lookup(), methodName, builderOp);
                var builderMethod = ClassFile.of().parse(bytes).methods().stream()
                        .filter(mm -> mm.methodName().equalsString(methodName)).findFirst().orElseThrow();
                clb.with(builderMethod);
            }
        }));
        return ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL).transformClass(classModel, classTransform);
    }

    @Test(dataProvider = "classes")
    void test(byte[] classData) throws Throwable {
        var bytes = replaceOpFieldWithBuilderMethod(ClassFile.of().parse(classData));
        print(bytes);
        var opFieldsAndIRs = getOpFieldsAndIRs(ClassFile.of().parse(classData));
        MethodHandles.Lookup l = MethodHandles.lookup().defineHiddenClass(bytes, true);
        for (var opFieldAndIR : opFieldsAndIRs) {
            var opFieldName = opFieldAndIR.opField().name().stringValue();
            var methodName = opFieldAndIR.opField().name().stringValue().substring(opFieldName.indexOf(':') + 2);
            var functionType = FunctionType.functionType(JavaType.type(Op.class), JavaType.type(OpFactory.class),
                    JavaType.type(TypeElementFactory.class));
            var mh = l.findStatic(l.lookupClass(),
                    methodName,
                    MethodRef.toNominalDescriptor(functionType).resolveConstantDesc(l));
            Assert.assertEquals(
                    ((Op) mh.invoke(ExtendedOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY)).toText(),
                    opFieldAndIR.ir());
        }
    }

    static void print(byte[] bytes) {
        print(ClassFile.of().parse(bytes));
    }

    static void print(ClassModel cm) {
        ClassPrinter.toYaml(cm, ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
    }

    record OpFieldAndIR(FieldRefEntry opField, String ir) {
    }

    List<OpFieldAndIR> getOpFieldsAndIRs(ClassModel cm) {
        var res = new ArrayList<OpFieldAndIR>();
        var cinit = cm.methods().stream().filter(mm -> mm.methodName().equalsString(ConstantDescs.CLASS_INIT_NAME))
                .findFirst().orElseThrow();
        CodeElement prev = null;
        for (CodeElement curr : cinit.code().get().elementList()) {
            if (curr instanceof FieldInstruction fi && fi.opcode() == Opcode.PUTSTATIC
                    && fi.owner().equals(cm.thisClass()) && fi.name().stringValue().endsWith("$op")) {
                if (prev instanceof ConstantInstruction.LoadConstantInstruction lci && lci.constantEntry() instanceof StringEntry se) {
                    res.add(new OpFieldAndIR(fi.field(), se.stringValue()));
                }
            }
            prev = curr;
        }
        return res;
    }
}
