import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.IOException;
import java.lang.classfile.*;
import java.lang.classfile.components.ClassPrinter;
import java.lang.classfile.constantpool.PoolEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.instruction.*;
import java.lang.constant.ConstantDescs;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.parser.OpParser;
import java.lang.reflect.code.type.CoreTypeFactory;
import java.lang.reflect.code.type.MethodRef;
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
            return new byte[][] {
                    IR.class.getResourceAsStream("IR.class").readAllBytes()
            };
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test(dataProvider = "classes")
    void test(byte[] cd) throws Throwable {
        var cf = ClassFile.of();
        var cm = cf.parse(cd);
        var opFieldsAndIRs = getOpFieldsAndIRs(cm);
        cd = removeOpFields(cm);
        for (var opFieldAndIR : opFieldsAndIRs) {
            var funcOp = ((FuncOp) OpParser.fromStringOfFuncOp(opFieldAndIR.ir()));
            var builderOp = OpBuilder.createBuilderFunction(funcOp);
            var builtOp = (Op) Interpreter.invoke(builderOp, ExtendedOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY);
            Assert.assertEquals(builtOp.toText(), opFieldAndIR.ir());

            cd = BytecodeGenerator.addOpByteCodeToClassFile(MethodHandles.lookup(), cm, opFieldAndIR.fieldName(), builderOp);
            cm = cf.parse(cd);
            Assert.assertEquals(cm.methods().stream().filter(mm -> mm.methodName().equalsString(opFieldAndIR.fieldName())).count(),
                    opFieldsAndIRs.size());

            var l = MethodHandles.lookup().defineHiddenClass(cd, true);
            var mh = l.findStatic(l.lookupClass(),
                    opFieldAndIR.fieldName(),
                    MethodRef.toNominalDescriptor(builderOp.invokableType()).resolveConstantDesc(l));
            Assert.assertEquals(
                    ((Op) mh.invoke(ExtendedOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY)).toText(),
                    opFieldAndIR.ir());
        }
        print(cm);
    }

    static void print(byte[] bytes) {
        print(ClassFile.of().parse(bytes));
    }

    static void print(ClassModel cm) {
        ClassPrinter.toYaml(cm, ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
    }

    record OpFieldAndIR(String fieldName, String ir) {}

    List<OpFieldAndIR> getOpFieldsAndIRs(ClassModel cm) {
        List<OpFieldAndIR> res = new ArrayList<>();
        var cinit = cm.methods().stream().filter(mm -> mm.methodName().equalsString(ConstantDescs.CLASS_INIT_NAME))
                .findFirst().orElseThrow();
        CodeElement prev = null;
        for (CodeElement curr : cinit.code().get().elementList()) {
            if (curr instanceof FieldInstruction fi && fi.opcode() == Opcode.PUTSTATIC
                    && fi.owner().equals(cm.thisClass()) && fi.name().stringValue().endsWith("$op")) {
                var lci = (ConstantInstruction.LoadConstantInstruction) prev;
                var pe = (StringEntry) lci.constantEntry();
                res.add(new OpFieldAndIR(fi.name().stringValue().substring(fi.name().stringValue().indexOf(':') + 2), pe.stringValue()));
            }
            prev = curr;
        }
        return res;
    }

    @Test
    void testRemovingOpField() {
        var bytes = removeOpFields(CLASS_MODEL);
        var cm = ClassFile.of().parse(bytes);
        Assert.assertFalse(cm.fields().stream().anyMatch(fm -> fm.fieldName().stringValue().endsWith("$op")
                && fm.fieldType().equalsString("String")));
        for (PoolEntry poolEntry : cm.constantPool()) {
            Assert.assertFalse(poolEntry instanceof StringEntry se && se.stringValue().startsWith("func @"));
        }
    }

    private byte[] removeOpFields(ClassModel cm) {
        var bytes = ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL).transformClass(cm,
                ClassTransform.dropping(e -> e instanceof FieldModel fm && fm.fieldName().stringValue().endsWith("$op")).andThen(
                        ClassTransform.transformingMethods(mm -> mm.methodName().equalsString(ConstantDescs.CLASS_INIT_NAME), (mb, me) -> {
                            if (me instanceof CodeModel com) {
                                mb.withCode(cob -> {
                                    ConstantInstruction.LoadConstantInstruction ldc = null;
                                    for (CodeElement e : com) {
                                        if (ldc != null && e instanceof FieldInstruction fi && fi.opcode() == Opcode.PUTSTATIC && fi.owner().equals(cm.thisClass()) && fi.name().stringValue().endsWith("$op")) {
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
                            } else {
                                mb.with(me);
                            }
                        })));
        return bytes;
    }
}
