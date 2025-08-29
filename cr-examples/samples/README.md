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

### Resources

1. [Article] [Code Models](https://openjdk.org/projects/babylon/articles/code-models)
2. [Article] [Emulating C# LINQ in Java using Code Reflection
   ](https://openjdk.org/projects/babylon/articles/linq)
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
java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.HelloCodeReflection
```

##### Run MathOptimizer

```bash
java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.MathOptimizer
```

##### Run InlineExample

```bash
java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.InlineExample
```

##### Run MathOptimizerWithInlining

```bash
java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.MathOptimizerWithInlining
```
