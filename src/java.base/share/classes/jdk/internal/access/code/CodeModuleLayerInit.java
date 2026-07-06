package jdk.internal.access.code;

/**
 * Intializes a module layer containing the incubating module jdk.incubator.code so that
 * it can be used by the jdk.compiler module for compiling reflectable code.
 * See use in com.sun.tools.javac.main.JavaCompiler of the jdk.compiler module.
 */
public class CodeModuleLayerInit {
    public static void initCodeModuleLayer(ModuleLayer layer) {
        Module codeReflectionModule = layer.findModule("jdk.incubator.code").get();
        Module jdkCompilerModule = CodeModuleLayerInit.class.getModule();
        // We need to add exports all java.base packages so that the jdk.incubator.code module can use them
        for (String packageName : jdkCompilerModule.getPackages()) {
            jdkCompilerModule.addExports(packageName, codeReflectionModule);
        }
    }
}
