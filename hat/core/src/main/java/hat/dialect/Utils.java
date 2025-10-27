package hat.dialect;

import hat.annotations.HATVectorType;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.annotation.Annotation;

public class Utils {

    public static TypeElement getVectorElementType(String primitive) {
        return switch (primitive) {
            case "float" -> JavaType.FLOAT;
            case "double" -> JavaType.DOUBLE;
            case "int" -> JavaType.INT;
            case "long" -> JavaType.LONG;
            case "short" -> JavaType.SHORT;
            case "byte" -> JavaType.BYTE;
            case "char" -> JavaType.CHAR;
            case "boolean" -> JavaType.BOOLEAN;
            default -> null;
        };
    }

    public record VectorMetaData(TypeElement vectorTypeElement, int lanes) {}

    public static VectorMetaData getVectorTypeInfo(JavaOp.InvokeOp invokeOp, int param) {
        Value varValue = invokeOp.operands().get(param);
        if (varValue instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getVectorMetaData(varLoadOp.resultType());
        }
        return null;
    }

    public static VectorMetaData getVectorTypeInfo(JavaOp.InvokeOp invokeOp) {
        return getVectorMetaData(invokeOp.resultType());
    }

    public static VectorMetaData getVectorMetaData(TypeElement typeElement) {
        try {
            Class<?> aClass = Class.forName(typeElement.toString());
            if (!aClass.isPrimitive()) {
                Annotation[] annotations = aClass.getAnnotations();
                for (Annotation annotation : annotations) {
                    if (annotation instanceof HATVectorType hatVectorType) {
                        return new VectorMetaData(getVectorElementType(hatVectorType.primitiveType()), hatVectorType.lanes());
                    }
                }
            }
        } catch (ClassNotFoundException _) {
        }
        return null;
    }


    public static int getWitdh(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return getWitdh(varLoadOp.operands().getFirst());
    }

    public static int getWitdh(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getWitdh(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorN();
            }
            return -1;
        }
    }

    public static TypeElement findVectorTypeElement(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorTypeElement(varLoadOp.operands().getFirst());
    }

    public static TypeElement findVectorTypeElement(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVectorTypeElement(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorElementType;
            }
            return null;
        }
    }



    public static String findNameVector(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameVector(varLoadOp.operands().getFirst());
    }

    public static String findNameVector(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.varName();
            }
            return null;
        }
    }

}
