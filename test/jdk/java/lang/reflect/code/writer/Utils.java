import java.lang.classfile.ClassModel;
import java.lang.classfile.CodeElement;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.constant.ConstantDescs;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.TypeElementFactory;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    static List<OpFieldAndIR> getOpFieldsAndIRs(ClassModel cm) {
        var cinit = cm.methods().stream()
                .filter(mm -> mm.methodName().equalsString(ConstantDescs.CLASS_INIT_NAME)).findFirst().orElseThrow();
        var opFieldsAndIRs = new ArrayList<OpFieldAndIR>();
        CodeElement prev = null;
        for (CodeElement ce : cinit.code().orElseThrow()) {
            if (ce instanceof FieldInstruction fi && fi.opcode() == Opcode.PUTSTATIC &&
                    fi.owner().equals(cm.thisClass()) && fi.name().stringValue().endsWith("$op") &&
                    prev instanceof ConstantInstruction.LoadConstantInstruction lci) {
                opFieldsAndIRs.add(new OpFieldAndIR(fi.field(), ((StringEntry) lci.constantEntry()).stringValue()));
            }
            prev = ce;
        }
        return opFieldsAndIRs;
    }

    static String irBuilderName(String opFieldName) {
        // e.g. A::add(int, int)int$op ---> add(int, int)int$op
        return opFieldName.substring(opFieldName.indexOf(':') + 2);
    }

    static MethodType irBuilderMethodType() throws ReflectiveOperationException {
        var functionType = FunctionType.functionType(JavaType.type(Op.class), JavaType.type(OpFactory.class),
                JavaType.type(TypeElementFactory.class));
        var mt = MethodRef.toNominalDescriptor(functionType).resolveConstantDesc(MethodHandles.lookup());
        return mt;
    }
}
