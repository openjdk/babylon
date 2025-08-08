## Code Reflection Examples

List of examples to learn and play with the Code Reflection API.

https://github.com/openjdk/babylon

### How to build?

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
