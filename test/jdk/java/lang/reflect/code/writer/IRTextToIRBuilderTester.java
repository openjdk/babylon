import java.lang.classfile.ClassFile;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.type.CoreTypeFactory;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;

public class IRTextToIRBuilderTester {

    public static void main(String[] args) throws Throwable {
        if (args.length != 2) {
            System.err.println("Usage: <program> <path_to_original_cf> <path_to_new_cf>");
            System.exit(-1);
        }
        var path_to_original_cf = Path.of(args[0]);
        var path_to_new_cf = Path.of(args[1]);

        var original_bytes = Files.readAllBytes(path_to_original_cf);
        var original_cm = ClassFile.of().parse(original_bytes);

        var url1 = path_to_new_cf.getParent().toUri().toURL();
        var url2 = path_to_original_cf.getParent().toUri().toURL();
        var ucl = new URLClassLoader(new URL[]{url1, url2});
        var fn = path_to_new_cf.getFileName().toString();
        var cn = fn.substring(0, fn.lastIndexOf('.'));
        var new_class = ucl.loadClass(cn);

        var opFieldsAndIRs = Utils.getOpFieldsAndIRs(original_cm);
        for (OpFieldAndIR opFieldAndIR : opFieldsAndIRs) {
            String mn = Utils.irBuilderName(opFieldAndIR.opField().name().stringValue());
            var mh = MethodHandles.lookup().findStatic(new_class, mn, Utils.irBuilderMethodType());
            var builtOp = ((Op) mh.invoke(ExtendedOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY));
            assert builtOp.toText().equals(opFieldAndIR.ir());
        }
    }
}
