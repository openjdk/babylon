## Code Reflection Examples

List of examples to learn and play with the Code Reflection API.

- GitHub repo: [https://github.com/openjdk/babylon](https://github.com/openjdk/babylon)

### Learning Code Reflection?

Here's an ordered list of code example to start learning code reflection and some of its features.
Each example is self-contained, and it can be used within the IDE to explore and understand how code reflection
transforms and executes code.

1. [`HelloCodeReflection`](https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/HelloCodeReflection.java): Just start using code reflection and lowering code models.
2. [`MathOptimizer`](https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/MathOptimizer.java): First code transformations to optimize a math function.
3. [`InlineExample`](https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/InliningExample.java): Simple example to illustrate the inlining.
4. [`MathOptimizerWithInlining`](https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/MathOptimizerWithInlining.java): Follow up of the [`MathOptimizer`](https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/MathOptimizer.java) to inline optimize calls into the code model.
6. [`DialectWithInvoke`][https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/DialectWithInvoke.java]:
Example of creating a dialect that replaces `Invoke` `Op` with a specific signature with a new `Op`. The dialect is handled as an intrinsic replacement.
6. [`DialectFMAOp`][https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/DialectFMAOp.java]: Example of how to extend the code reflection `Op` to create a new dialect. It analysis the code for substitution of Add(Mult) to create a new `FMA` Op.
7. [`DynamicFunctionBuild`][https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/DynamicFunctionBuild.java]: Example of how to create a new function dynamically to compute the inverse of a square root. The code model is built dynamically for a new method and it is evaluated in the `Interpreter`.
8. [`CodeReflectionProcessor`](https://github.com/openjdk/babylon/blob/code-reflection/cr-examples/samples/src/main/java/oracle/code/samples/CodeReflectionProcessor.java): A simple code model-based annotation processor

### Resources

1. [Article] [Code Models](https://openjdk.org/projects/babylon/articles/code-models)
2. [Article] [Emulating C# LINQ in Java using Code Reflection](https://openjdk.org/projects/babylon/articles/linq)
3. [Video] [Project Babylon - Code Reflection @JVMLS 2024](https://www.youtube.com/watch?v=6c0DB2kwF_Q)
4. [Video] [Java and GPUs using Code Reflection @JVMLS 2023](https://www.youtube.com/watch?v=lbKBu3lTftc)

### How to build with project?

#### 1. Build Babylon JDK

We need to use the JDK build that enables the code reflection API (Babylon).

```bash
git clone https://github.com/openjdk/babylon
cd babylon
bash configure --with-boot-jdk=${JAVA_HOME}
```

Then, we use the built JDK as `JAVA_HOME`

```bash
export JAVA_HOME=/$HOME/repos/babylon/build/macosx-aarch64-server-release/jdk/
export PATH=$JAVA_HOME/bin:$PATH
```

#### 2. Build examples

```bash
mvn clean package
```

#### 3. Run the examples


##### Run HelloCodeReflection

```bash
java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.HelloCodeReflection
```

##### Run MathOptimizer

```bash
java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.MathOptimizer
```

##### Run InlineExample

```bash
java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.InlineExample
```

##### Run MathOptimizerWithInlining

```bash
java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.MathOptimizerWithInlining
```

##### Run DialectWithInvoke

```bash
java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DialectWithInvoke
```

##### Run DialectFMAOp

```bash
java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DialectFMAOp
```

##### Run DynamicFunctionBuild

```bash
java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DynamicFunctionBuild
```

##### Compile with CodeReflectionProcessor

```bash
javac --add-modules jdk.incubator.code --processor-path target/crsamples-1.0-SNAPSHOT.jar -processor oracle.code.samples.CodeReflectionProcessor <.java files to compile>
```
