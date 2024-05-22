# Interface Mapping
We have a copy of Per's segment mapping code from
https://github.com/minborg/panama-foreign/blob/segment-mapper/src/java.base/share/classes

Specifically we hava a copy of the following dirs

```
src/main/java.base
     java.lang.foreign.mapper
     jdk.internal.foreign.mapper
```

To allow this code to be used with 'babylon jdk' we need to add:-

`--patch-module=java.base=src/main/java.base`

To javac opts (to include this src in the compilation 

`--patch-module=java.base=babylon/out/production/babylon`

To the java VM options at runtime, so we can access these classes as if in java.base 

In intellij we need to make sure that the following is added to .idea/compiler.xml
```xml
<!-- babylon/.idea/compiler.xml -->
<component name="JavacSettings">
    <option name="ADDITIONAL_OPTIONS_OVERRIDE">
        <component name="JavacSettings">
            <option name="ADDITIONAL_OPTIONS_OVERRIDE">
                <module name="babylon" options="
                    --enable-preview
                    --patch-module=java.base=$PROJECT_DIR$/../src/main/java.base
                    --add-exports=java.base/java.lang.reflect.code.descriptor.impl=ALL-UNNAMED
                    --add-exports=java.base/java.lang.reflect.code.impl=ALL-UNNAMED
                    --add-exports=java.base/jdk.internal=ALL-UNNAMED
                    --add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED
                    --add-exports=java.base/jdk.internal.foreign.layout=ALL-UNNAMED
                    --add-exports=java.base/jdk.internal.util=ALL-UNNAMED
                    --add-exports=java.base/sun.security.action=ALL-UNNAMED
                    --add-exports=java.base/jdk.internal.misc=ALL-UNNAMED
                " />
                <module name="segment.mapper" options="
                    --add-exports java.base/jdk.internal.vm.annotation=ALL-UNNAMED
                    --add-exports java.base/sun.security.action=ALL-UNNAMED" />
            </option>
        </component>
    </option>
</component>
```
Note specifically the `options` attribute above must include the `--patch-module` option above

To run we need to ensure that the following Vm options are added to the run configuration

```--enable-preview
--add-opens=java.base/java.lang.reflect.code.descriptor.impl=ALL-UNNAMED
--patch-module=java.base=out/production/babylon
```
