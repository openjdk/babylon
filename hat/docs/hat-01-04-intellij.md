
#  Intellij and Clion
----

* [Contents](hat-00.md)
* House Keeping
    * [Project Layout](hat-01-01-project-layout.md)
    * [Building Babylon](hat-01-02-building-babylon.md)
    * [Maven and CMake](hat-01-03-maven-cmake.md)
* Programming Model
    * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)

---

## Intellij and Clion

We can use JetBrains' `IntelliJ` and `Clion` for dev work and 
decided to leave some project artifacts in the repo.

Care must be taken with Intellij and Clion 
as these tools do not play well together,
specifically we cannot have `Clion` and `Intellij`
project artifacts rooted under each other or in the same dir.

### Intellij
The `intellij` subdir under the root HAT directory
contains the `.idea` project dir and the various `*.iml` files
for each of the various `modules`
(note the use of `Intellji`'s meaning of the word of module here)

As you will note the `intellij` dir is somewhat self contained.  the various `*.iml`
files refer to the source dirs using relative paths.

I tend to add `Intellij` modules by hand.  There are gotchas ;)

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

Making run configurations available to other developers is probably `a bridge too far`

But with some careful XML tooling we can make it easier to add 'run configurations'.

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

### Clion

Thankfully Clion uses cmake, so we get to re-use the CMakeLists.txt in the various backends to build.

The intent is that these cmake artifacts can be run standalone (using cmake in the appropriate dir),
from within Clion and can be used by maven.  So the CMakeLists.txt files have some extra variables to
help us use them in these three modes.  




