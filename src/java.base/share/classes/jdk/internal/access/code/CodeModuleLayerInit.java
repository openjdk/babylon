package jdk.internal.access.code;

public class CodeModuleLayerInit {
    public static void initCodeModuleLayer(ModuleLayer layer) {
        Module codeReflectionModule = layer.findModule("jdk.incubator.code").get();
        Module jdkCompilerModule = CodeModuleLayerInit.class.getModule();
        // We need to add exports all java.base packages so that the plugin can use them
        for (String packageName : jdkCompilerModule.getPackages()) {
            jdkCompilerModule.addExports(packageName, codeReflectionModule);
        }
    }
}
