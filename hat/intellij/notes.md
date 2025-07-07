This is how we exclude in the .iml files
For example in example-nbody.iml
```xml
<module>
  <component name="NewModuleRootManager" inherit-compiler-output="true">
    <exclude-output />
    <content url="file://$MODULE_DIR$/../examples/nbody">
      <sourceFolder url="file://$MODULE_DIR$/../examples/nbody/src/main/java" isTestSource="false" />
      <sourceFolder url="file://$MODULE_DIR$/../examples/nbody/src/main/resources" type="java-resource" />
      <!--<excludeFolder url="file://$MODULE_DIR$/../examples/nbody/src/main/java/nbody" />-->
  </component>
</module>
```
Sadly to exclude a single file we have to go to
```
    IntelliJ>Settings>Build, Execution, Deployment> Compiler>Excludes

    And add to the UI
    FULL_PATH_TO_HAT/hat/wraps/opengl/src/main/java/wrap/opengl/GLCallbackEventHandler.java
```


We need these to be able to run in a run configuration
```
RunConfiguration
   VMOptions
     -XstartOnFirstThread  (on mac only for nbody/opengl)
     --enable-native-access=ALL-UNNAMED
      -Djava.library.path=FULL_PATH_TO_HAT/build

  WorkingDirectory
     FULL_PATH_TO_HAT/wraps/opengl/src/main/java/wrap/opengl/GLCallbackEventHandler.java
     /Users/grfrost/github/babylon-grfrost-fork/hat/intellij

  Assuming Working dir is
     FULL_PATH_TO_HAT/intellij

```

Note that run configs are held in `intellij/.idea/workspace.xml`

I tend to add them manually.
```xml
<project version="4">
  // other components
<component name="RunManager">
  <configuration name="Main" type="Application" factoryName="Application" temporary="true" nameIsGenerated="true">
      <classpathModifications>
        <entry path="$PROJECT_DIR$/../build/hat-backend-ffi-opencl-1.0.jar" />
        <entry path="$PROJECT_DIR$/backend_ffi_shared.iml" />
      </classpathModifications>
      <option name="MAIN_CLASS_NAME" value="nbody.Main" />
      <module name="example_nbody" />
      <option name="VM_PARAMETERS" value="-XstartOnFirstThread --enable-native-access=ALL-UNNAMED -Djava.library.path=../build" />
      <extension name="coverage">
        <pattern>
          <option name="PATTERN" value="nbody.*" />
          <option name="ENABLED" value="true" />
        </pattern>
      </extension>
      <method v="2">
        <option name="Make" enabled="true" />
      </method>
  </configuration>
  <recent_temporary>
      <list>
        <item itemvalue="Application.Main" />
      </list>
  </recent_temporary>
</component>
</project>
```


