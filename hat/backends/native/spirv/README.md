# SPIR Backend

This backend depends on :

> A built version of on TornadoVm's SPIRV library being available
>   [https://github.com/beehive-lab/beehive-spirv-toolkit.git](https://github.com/beehive-lab/beehive-spirv-toolkit.git)

> The source for the spirv example in the babylon project tree
> [https://github.com/openjdk/babylon/tree/code-reflection/cr-examples/spirv](https://github.com/openjdk/babylon/tree/code-reflection/cr-examples/spirv)

So by default it is commented out in the parent `pom.xml`

```
   <modules>
     <module>opencl</module>
     <module>cuda</module>
     <module>mock</module>
     <module>ptx</module>
     <!--<module>spirv</module>-->
   </modules>
```

To include SPIRV our maven build assumes you have a project layout matching that described in the projiect roots README.md

```
/${HOME}/github
   ├── babylon
   ├── babylon-my-fork
   └── ...
```

And you are either working in '~/github/babylon/hat' or your fork '~/github/babylon-my-fork/hat'

### Getting and building TornadoVM's SPIRV library
```
export GITHUB=${HOME}/github
mkdir -p ${GITHUB}
cd ${GITHUB}
git clone https://github.com/beehive-lab/beehive-spirv-toolkit.git
```

So now you have

```
/${HOME}/github
   ├── babylon
   ├── beehive-spirv-toolkit
   ├── babylon-my-fork
   └── ...
```

Assuming you have your babylon (or forked babylon) JDK built and you have already set `${JAVA_HOME}` to point to your built jdk and have `${JAVA_HOME}/bin` in your PATH

Then you should be able to build `beehive-spirv-toolkit` using maven.

```
cd ${GITHUB}/
GITHUB=${HOME}/github/beehive-spirv-toolkit
mvn clean install
```

The hat maven build process will assume that it will find
> ../../beehive-spirv-toolkit

and
> ../../babylon/cr-examples/spirv

relative to the `hat` dir

If this is correct you should be able to uncomment out the 'spirv' module in the parent's `pom.xml` and the spirv backend will build as part of a normal hat build.




