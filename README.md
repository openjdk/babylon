# Welcome to the JDK!

For build instructions please see the
[online documentation](https://openjdk.org/groups/build/doc/building.html),
or either of these files:

- [doc/building.html](doc/building.html) (html version)
- [doc/building.md](doc/building.md) (markdown version)

See <https://openjdk.org/> for more information about the OpenJDK
Community and the JDK and see <https://bugs.openjdk.org> for JDK issue
tracking.

## Particulars related to Babylon

The Babylon JDK builds like any other JDK, see the build instructions above.

### Code shared between `java.base` and `jdk.compiler` modules

A subset of code in `java.base` is copied with package renaming into
the `jdk.compiler` module. This is the set of code required to build
and serialize code models. Due to bootstrapping constraints, compiling
the compiler it cannot depend on all code in `java.base`. In the future
we may come up with a better solution. For now the build has been modified
to copy the code, which leverages the script `cr-util/copy-to-compiler.sh`.
If there are issues where code in `java.base` has been modified but is
not being copied then doing `make clean-gensrc` should resolve the issue.

### Testing

Specific compiler tests can be executed using `jtreg`, for example:

```
jtreg -jdk:./build/macosx-x86_64-server-release/jdk/ -ea -esa -avm -va test/langtools/tools/javac/reflect/
```

Specific runtime tests can be executed using `jtreg`, for example:

```
jtreg -jdk:./build/macosx-x86_64-server-release/jdk/ -ea -esa -avm -va test/jdk/java/lang/reflect/code/
```

In addition, the runtime tests can be executed using make with the test group 
`jdk_lang_reflect_code` as follows:

```
make test TEST=jdk_lang_reflect_code
```