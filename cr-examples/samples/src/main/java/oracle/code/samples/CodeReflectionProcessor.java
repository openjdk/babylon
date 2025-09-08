package oracle.code.samples;

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp;
import jdk.incubator.code.dialect.java.JavaOp.ThrowOp;
import jdk.incubator.code.dialect.java.JavaOp.TryOp;
import jdk.incubator.code.dialect.java.JavaType;

import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.util.ElementScannerPreview;
import javax.tools.Diagnostic.Kind;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

@SupportedAnnotationTypes("jdk.incubator.code.CodeReflection")
@SupportedSourceVersion(SourceVersion.RELEASE_26)
public class CodeReflectionProcessor extends AbstractProcessor {

    @Override
    public synchronized void init(ProcessingEnvironment processingEnv) {
        super.init(processingEnv);
    }

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
        for (Element e : roundEnv.getElementsAnnotatedWith(CodeReflection.class)) {
            ReflectableMethodScanner scanner = new ReflectableMethodScanner();
            scanner.scan(e);
        }
        return true;
    }

    class ReflectableMethodScanner extends ElementScannerPreview<Void, Void> {

        final Messager logger = processingEnv.getMessager();

        @Override
        public Void scan(Element e, Void unused) {
            return super.scan(e, unused);
        }

        @Override
        public Void visitExecutable(ExecutableElement e, Void unused) {
            if (e.getAnnotationsByType(CodeReflection.class) != null) {
                Optional<FuncOp> funcOp = Op.ofElement(processingEnv, e);
                funcOp.ifPresent(f -> processMethodModel(e, f));
            }
            return null;
        }

        void processMethodModel(ExecutableElement element, FuncOp funcOp) {
            funcOp.traverse(null, (acc, ce) -> processOp(element, ce));
        }

        Void processOp(Element element, CodeElement<?, ?> codeElement) {
            switch (codeElement) {
                case InvokeOp invokeOp -> {
                    var desc = invokeOp.invokeDescriptor();
                    var receiverType = (JavaType) desc.refType();
                    String methodName = desc.name();
                    List<String> unsupportedMethods = UNSUPPORTED_METHODS.getOrDefault(receiverType, List.of());
                    for (String unsupportedMethod : unsupportedMethods) {
                        if (unsupportedMethod.equals(methodName)) {
                            String methErrString = receiverType.toNominalDescriptor().displayName() + "." + methodName;
                            logger.printMessage(Kind.ERROR,  methErrString + " not supported in reflectable methods", element);
                            break;
                        }
                    }
                }
                case Op op -> {
                    String unsupportedOp = UNSUPPORTED_OPS.get(op.getClass());
                    if (unsupportedOp != null) {
                        logger.printMessage(Kind.ERROR, unsupportedOp + " not supported in reflectable methods", element);
                    }
                }
                default -> {
                    // do nothing
                }
            }
            return null;
        }
    }

    static final Map<JavaType, List<String>> UNSUPPORTED_METHODS = Map.of(
            JavaType.type(System.class), List.of("exit", "gc", "load", "loadLibrary"),
            JavaType.type(Runtime.class), List.of("load", "loadLibrary"));

    static final Map<Class<?>, String> UNSUPPORTED_OPS = Map.of(
            ThrowOp.class, "throw statement",
            TryOp.class, "try/catch statement"
    );
}
