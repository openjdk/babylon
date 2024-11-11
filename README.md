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

The Babylon API and implementation resides in the incubating model
`jdk.incubator.code`. Compilation and execution of dependent code requires
that this module be made visible by explicitly adding to the list of modules
e.g., such as with the command line option `--add-modules jdk.incubator.code`.

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