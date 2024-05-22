# HAT Project Primer

This is a fairly large project with Java and Native artifacts.

We also rely on the `babylon` (JDK23+Babylon) project, and the project will initially be made available as a subproject
called `hat` under [github.com/openjdk/babylon](https://github.com/openjdk/babylon)

## Intellij and Clion

Whilst I do use JetBrains' `IntelliJ` and `Clion` for dev work and plan to leave project artifacts in the tree.
Care must be taken as these tools do not play well together, specifically we cannot have `Clion` and `Intellij`
project artifacts rooted under each other or in the same dir.

I have `intellij` and `clion` siblings which will act as roots
for various tools and then use relative paths (or even symbolic links) in the various `.iml` files to locate the various source roots

So far this has worked ok.

### cmake

I have never been a fan of maven especially for projects with native code. So I suggest using `cmake`

```bash
cd hat
mkdir build
cd build
cmake ..
cd ..
cmake --build build --target <yourtarget>
```

Who knows, one day Maven may prevail.

### Initial Project Layout


```
${BABYLON_JDK}
   └── hat
        │
        ├── CMakeFile
        ├── build
        │
        ├── intellij
        │    ├── .idea
        │    │    ├── compiler.xml
        │    │    ├── misc.xml
        │    │    ├── modules.xml
        │    │    ├── uiDesigner.xml
        │    │    ├── vcs.xml
        │    │    └── workspace.xml
        │    │
        │    ├── hat.iml
        │    ├── backend_(spirv|mock|cuda|ptx|opencl).iml
        │    └── (mandel|violajones|experiments).iml
        │
        ├── hat
        │    └── src
        │         └── java
        │
        ├── backends
        │    └── (opencl|cuda|ptx|mock|shared)
        │          └── src
        │              ├── cpp
        │              ├── include
        │              ├── java
        │              └── services
        └── examples
             ├── mandel
             │    └── src
             │         └── java
             └── violajones
                  └── src
                       ├── java
                       └── resources
```
As you will note the `intellij` dir is somewhat self contained.  the various `*.iml`
files refer to the source dirs using relative paths.

I tend to add `Intelli` modules by hand.  There are gotchas ;)

As with every intellij project, `.idea/modules.xml` 'points' to the iml files for each module (intellij's notion of module ;) )
```xml
<!--
   └──hat
       └── intellij
            └── .idea
                 └── modules.xml
-->
 <modules>
      <module fileurl="file://$PROJECT_DIR$/hat.iml"   />
      <module fileurl="file://$PROJECT_DIR$/backend_opencl.iml"  />
      <!-- yada yada -->
 </modules>

```

The various `.iml` files then  have relative paths to their source/resource dirs roots.

```xml
<module type="JAVA_MODULE" version="4">
  <component name="NewModuleRootManager" inherit-compiler-output="true">
    <exclude-output />
    <content url="file://$MODULE_DIR$/../../../hat/src/java">
      <sourceFolder url="file://$MODULE_DIR$/../../../hat/src/java" isTestSource="false" />
    </content>
    <orderEntry type="inheritedJdk" />
    <orderEntry type="sourceFolder" forTests="false" />
    <orderEntry type="module" module-name="hat" />
  </component>
</module>

```
### How intellij stores run configurations

I also tend to hand hack run configurations so will leave this here for reference

```xml
<component name="RunManager" selected="Application.MandelTest">
    <configuration name="Mandel" type="Application"
                   factoryName="Application" temporary="true"
                   nameIsGenerated="true">
      <option name="MAIN_CLASS_NAME" value="mandel.Mandel" />
      <module name="mandel" />
      <option name="VM_PARAMETERS" value="
          --enable-preview
          --add-exports=java.base/java.lang.reflect.code.descriptor.impl=ALL-UNNAMED
          --add-exports=java.base/java.lang.foreign.mapper=ALL-UNNAMED
          --patch-module=java.base=$PROJECT_DIR$/out/production/java_base_patch
          -Djava.lang.foreign.mapper.debug=true" />
      <extension name="coverage">
        <pattern>
          <option name="PATTERN" value="mandel.*" />
          <option name="ENABLED" value="true" />
        </pattern>
      </extension>
      <method v="2">
        <option name="Make" enabled="true" />
      </method>
    </configuration>
    <!-- more configs -->
</component>
```





