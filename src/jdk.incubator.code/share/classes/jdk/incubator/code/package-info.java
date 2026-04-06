/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

/// Defines an enhancement to the [core reflection][java.lang.reflect] API called _code reflection_.
///
/// Code reflection supports access to a model of code in a method or lambda expression, a
/// [_code model_](#code-models-heading), that is suited for analysis and [transformation](#transforming-heading).
///
/// ## Core reflection API
///
/// The core reflection API is a powerful feature that enables inspection of Java code at run time. For example,
/// consider the following Java code that we want to inspect, a class containing one field and one method, and another
/// class also containing one field and one method.
///
/// {@snippet lang = "java":
/// static class Example {
///     static Runnable R = () -> IO.println("Example:field:R");
///     static int add(int a, int b) {
///         IO.println("Example:method:add");
///         return a + b;
///     }
///
///     static class Nested {
///         static Runnable R = () -> IO.println("Example.Nested:field:R");
///         void m() { IO.println("Example.Nested:method:m"); }
///     }
/// }
///}
///
/// We can write a simple stream that uses core reflection and traverses program structure, a tree of annotated
/// elements, starting from a given class and reporting elements in a topological order.
///
/// {@snippet lang = "java":
/// static Stream<AnnotatedElement> elements(Class<?> c) {
///     return Stream.of(c).mapMulti((e, mapper) -> traverse(e, mapper));
/// }
/// private static void traverse(AnnotatedElement e,
///                              Consumer<? super AnnotatedElement> mapper) {
///     mapper.accept(e);
///     if (e instanceof Class<?>c) {
///         for (Field df : c.getDeclaredFields()) { traverse(df, mapper); }
///         for (Method dm : c.getDeclaredMethods()) { traverse(dm, mapper); }
///         for (Class<?> dc : c.getDeclaredClasses()) { traverse(dc, mapper); }
///     }
/// }
///}
///
/// ([AnnotatedElement][java.lang.reflect.AnnotatedElement] is the common super type of [Class][java.lang.Class],
/// [Field][java.lang.reflect.Field], and [Method][java.lang.reflect.Method].)
///
/// The `traverse` method recursively traverses a class's declared fields, methods, and enclosed class. Starting from
/// `Example`, using a class literal expression, we can print out the fields, methods, and classes we encounter.
///
/// {@snippet lang = "java":
/// elements(Example.class)
///     .forEach(IO::println);
///}
///
/// More interestingly we can perform some simple analysis, such as counting the number of static fields whose type is
/// [Runnable][java.lang.Runnable].
///
/// {@snippet lang = "java":
/// static boolean isStaticRunnableField(Field f) {
///     return f.accessFlags().contains(AccessFlag.STATIC)
///         && Runnable.class.isAssignableFrom(f.getType());
/// }
/// assert 2 == elements(Example.class)
///     .filter(e -> e instanceof Field f && isStaticRunnableField(f))
///     .count();
///}
///
/// However, it is not possible to perform some analysis of the code in the lambda expressions and methods. The core
/// reflection API can only inspect the classes, fields, and methods – it provides no facility to go deeper and inspect
/// code.
///
/// ## Code reflection
///
/// Using code reflection we can go deeper. We can update `Example` so that the code of the lambda expressions and
/// methods is accessible just like the fields and method.
///
/// {@snippet lang = "java":
/// import jdk.incubator.code.*;
/// import jdk.incubator.code.bytecode.*;
/// import jdk.incubator.code.dialect.core.*;
/// import jdk.incubator.code.dialect.java.*;
/// import static jdk.incubator.code.dialect.core.CoreOp.*;
/// import static jdk.incubator.code.dialect.java.JavaOp.*;
///
/// static class Example {
///
/// @Reflect     static Runnable R = () -> IO.println("Example:field:R");
/// @Reflect     static int add(int a, int b) {
///         IO.println("Example:method:add");
///         return a + b;
///     }
///
///     static class Nested {
/// @Reflect         static Runnable R = () -> IO.println("Example.Nested:field:R");
/// @Reflect         void m() { IO.println("Example.Nested:method:m"); }
///     }
/// }
///}
///
/// We declare the lambda expressions and methods are reflectable by annotating their declarations with
/// [Reflect][jdk.incubator.code.Reflect]. By doing so we grant access to their code. When the source of the `Example`
/// class is compiled by javac it translates its internal model of method `add`’s code to a standard model, called a
/// [_code model_](#code-models-heading), and stores the code model in a class file related to the `Example` class file
/// where `add`’s code is compiled to bytecode. (The same occurs for the other method and lambda expressions.)
///
/// A code model is an immutable tree of [_code elements_][jdk.incubator.code.CodeElement], where each element models
/// some Java statement or expression (for further details see the [Code models](#code-models-heading) section).
///
/// We can use code reflection to access the code model of an annotated element, which loads the corresponding code
/// model that was stored in the related class file.
///
/// {@snippet lang = "java":
/// static Object getStaticFieldValue(Field f) {
///     try { return f.get(null); }
///     catch (IllegalAccessException e) { throw new RuntimeException(e); }
/// }
/// static Optional<? extends CodeElement<?, ?>> getCodeModel(AnnotatedElement ae) { // @link substring="CodeElement" target="CodeElement"
///     return switch (ae) {
///         case Method m -> Op.ofMethod(m); // @link substring="Op.ofMethod" target="Op#ofMethod"
///         case Field f when isStaticRunnableField(f) ->
///                 Op.ofLambda(getStaticFieldValue(f)).map(Quoted::op); // @link substring="Op.ofLambda" target="Op#ofLambda"
///         default -> Optional.empty();
///    };
/// }
///}
///
/// The method `getCodeModel` returns the code model for a reflectable method or lambda expression, a code element that
/// is the root of the code model tree. By default, methods and lambda expressions are not reflectable, so we return an
/// optional value. If the annotated element is a method we retrieve the code model from the method. If the annotated
/// element is a static field whose type is `Runnable` we access its value, an instance of `Runnable` whose result is
/// produced from a lambda expression, and from that instance we retrieve the lambda expression’s code model. The
/// retrieval is slightly different for lambda expressions since they can capture values (for more details see the
/// [Declaring and accessing reflectable code](#declaring-and-accessing-reflectable-code-heading) section).
///
/// We can use `getCodeModel` to map from `Example`’s annotated elements to their code models.
///
/// {@snippet lang = "java":
/// elements(Example.class)
///         // AnnotatedElement -> CodeModel?
///         .flatMap(ae -> getCodeModel(ae).stream())
///         .forEach(IO::println);
///}
///
/// More interestingly we can now perform some simple analysis of code, such as extracting the values of the string
/// literal expressions that are printed.
///
/// {@snippet lang = "java":
/// static final MethodRef PRINTLN = MethodRef.method(IO.class, "println", // @link substring="MethodRef" target="jdk.incubator.code.dialect.java.MethodRef"
///         void.class, Object.class);
/// static Optional<String> isPrintConstantString(CodeElement<?, ?> e) {
///     if (e instanceof InvokeOp i && // @link substring="InvokeOp" target="jdk.incubator.code.dialect.java.JavaOp.InvokeOp"
///             i.invokeDescriptor().equals(PRINTLN) &&
///             i.operands().get(0).declaringElement() instanceof ConstantOp cop && // @link substring="ConstantOp" target="jdk.incubator.code.dialect.core.CoreOp.ConstantOp"
///             cop.value() instanceof String s) {
///         return Optional.of(s);
///     } else {
///         return Optional.empty();
///     }
/// }
/// static List<String> analyzeCodeModel(CodeElement<?, ?> codeModel) {
///     return codeModel.elements()
///             // CodeElement -> String?
///             .flatMap(e -> isPrintConstantString(e).stream())
///             .toList();
/// }
///}
///
/// The method `analyzeCodeModel` uses a stream to [traverse](#traversing-heading) over all elements of a code model and
/// returns the list of string literal values passed to invocations of `IO.println`. We can then use `analyzeCodeModel`
/// to further refine our steam expression to print out all such string literal values.
///
/// {@snippet lang = "java":
/// elements(Example.class)
///         // AnnotatedElement -> CodeModel?
///         .flatMap(ae -> getCodeModel(ae).stream())
///         // CodeModel -> List<String>
///         .map(codeModel -> analyzeCodeModel(codeModel))
///         .forEach(IO::println);
///}
///
/// ## Declaring and accessing reflectable code
///
/// In total there are four syntactic locations where the [jdk.incubator.code.Reflect] annotation can appear that
/// govern what is declared reflectable. The locations are describe in detail in the [jdk.incubator.code.Reflect]
/// documentation.
///
/// The code model of a reflectable method is accessed by invoking [jdk.incubator.code.Op#ofMethod] with an argument
/// that is a `Method` instance (retrieved using the core reflection API) representing the reflectable method. The
/// result is an optional value that contains the code model modeling the method. For example, we can access the
/// code model for the `Example.add` method as follows:
///
/// {@snippet lang = "java":
/// Method addMethod = Example.class.getDeclaredMethod("add", int.class, int.class);
/// CoreOp.FuncOp codeModel = Op.ofMethod(addMethod).orElseThrow(); // @link substring="CoreOp.FuncOp" target="jdk.incubator.code.dialect.core.CoreOp.FuncOp"
/// assert codeModel == Op.ofMethod(addMethod).orElseThrow();
///}
///
/// We assert that if we obtain the code model for a second time the _same_ instance is returned. The identity
/// of code elements (and more generally items) in the code model are stable, and therefore they can be used as
/// stable keys for associating code elements with other information.
///
/// The code model of a reflectable lambda expression (or method reference) is accessed by invoking
/// [jdk.incubator.code.Op#ofLambda] with an argument that is an instance of a functional interface associated with the
/// reflectable lambda expression. The result is an optional value that contains a [jdk.incubator.code.Quoted] instance,
/// from which may be retrieved the code model modelling the lambda expression. In addition, it is possible to retrieve
/// a mapping of run time values to items in the code model that model final, or effectively final, variables used but
/// not declared in the lambda expression. For example, we can access the code model for the lambda expression used to
/// initialize the `Example.R` field as follows:
///
/// {@snippet lang = "java":
/// Field rField = Example.class.getDeclaredField("R", Runnable.class);
/// Object rFieldInstance = rField.get(null);
/// Quoted<JavaOp.LambdaOp> quotedCodeModel = Op.ofLambda(rFieldInstance).orElseThrow(); // @link substring="Quoted" target="jdk.incubator.code.Quoted"
/// JavaOp.LambdaOp codeModel = quotedCodeModel.op(); // @link substring="JavaOp.LambdaOp" target="jdk.incubator.code.dialect.java.JavaOp.LambdaOp"
///}
///
/// If a lambda expression captures values we can additionally access those values. For example,
///
/// {@snippet lang = "java":
/// int capture = 42;
/// @Reflect Runnable r = () -> IO.println(capture);
/// Quoted<JavaOp.LambdaOp> quotedCodeModel = Op.ofLambda(r).orElseThrow();
///
/// SequencedMap<Value, Object> capturedValues = quotedCodeModel.capturedValues(); // @link substring="capturedValues()" target="jdk.incubator.code.Quoted#capturedValues"
/// assert capturedValues.size() == 1;
/// assert capturedValues.values().contains(42);
///}
///
/// ## Code models
///
/// A code model is an _immutable_ instance of data structures that can, in general, model many kinds of code, be it
/// Java code or foreign code. It has some properties like an Abstract Syntax Tree ([AST][AST]) used by a source
/// compiler, such as modeling code as a tree of arbitrary depth, and some properties like an
/// [intermediate representation][IR] used by an optimizing compiler, such as modeling control flow and data flow as
/// graphs. These properties ensure code models can preserve many important details of code they model and ensure code
/// models are suited for analysis and transformation.
///
/// [AST]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
///
/// [IR]: https://en.wikipedia.org/wiki/Intermediate_representation
///
/// The primary data structure of a code model is a tree of [_code elements_][jdk.incubator.code.CodeElement]. There
/// are three kinds of code elements, [operation][jdk.incubator.code.Op], [body][jdk.incubator.code.Body], and
/// [block][jdk.incubator.code.Block]. The root of a code model is an operation, and descendant operations form a tree
/// of arbitrary depth.
///
/// Code reflection supports representing the data structures of a code model, code elements for modeling Java language
/// constructs and behavior, traversing code models, building code models, and transforming code models.
///
/// ## Traversing
///
/// A code model, a tree of code elements, can be traversed. One approach to write a recursive method that iterates
/// over code elements and their children. That way we can get a sense of what a code model contains.
///
/// {@snippet lang = "java":
/// Method addMethod = Example.class.getDeclaredMethod("add", int.class, int.class);
/// CoreOp.FuncOp codeModel = Op.ofMethod(addMethod).orElseThrow();
///
/// static void traverse(int depth, CodeElement<?, ?> e) {
///     IO.println("  ".repeat(depth) + e.getClass());
///
///     for (CodeElement<?, ?> c : e.children()) { // @link substring="children()" target="jdk.incubator.code.CodeElement#children"
///         traverse(depth + 1, c);
///     }
/// }
/// traverse(0, codeModel);
///}
///
/// The `traverse` method prints out the class of the code element it encounters and prefixes that with white space
/// proportionate to the depth of the element in the code model tree. Invoking the `traverse` method with the code model
/// of the `Example.add` method prints out the following:
///
/// {@snippet lang = "text":
/// class jdk.incubator.code.dialect.core.CoreOp$FuncOp
///   class jdk.incubator.code.Body
///     class jdk.incubator.code.Block
///       class jdk.incubator.code.dialect.core.CoreOp$VarOp
///       class jdk.incubator.code.dialect.core.CoreOp$VarOp
///       class jdk.incubator.code.dialect.core.CoreOp$ConstantOp
///       class jdk.incubator.code.dialect.java.JavaOp$InvokeOp
///       class jdk.incubator.code.dialect.core.CoreOp$VarAccessOp$VarLoadOp
///       class jdk.incubator.code.dialect.core.CoreOp$VarAccessOp$VarLoadOp
///       class jdk.incubator.code.dialect.java.JavaOp$AddOp
///       class jdk.incubator.code.dialect.core.CoreOp$ReturnOp
///}
///
/// We can observe that the top of the tree is the [FuncOp][jdk.incubator.code.dialect.core.CoreOp.FuncOp] which
/// contains one child, a [Body][jdk.incubator.code.Body], which in turn contains one child, a
/// [Block][jdk.incubator.code.Block], which in turn contains a sequence of eight operations. Bodies and blocks provide
/// additional structure for modeling code. Each operation models some part of the `add` methods code, for example
/// variable declaration operations (instances of [VarOp][jdk.incubator.code.dialect.core.CoreOp.VarOp]) model Java
/// variable declarations, in this case the method parameters, and the add operation (instance of
/// [AddOp][jdk.incubator.code.dialect.java.JavaOp.AddOp]) models the Java + operator.
///
/// Alternatively, we can stream over elements of the code model (as we did previously when analyzing the code for
/// string literals) in the same topologically sorted order using the
/// [CodeElement.elements][jdk.incubator.code.CodeElement#elements] method.
///
/// {@snippet lang = "java":
/// codeModel.elements().forEach((CodeElement<?, ?> e) -> {
///     int depth = 0;
///     var parent = e;
///     while ((parent = parent.parent()) != null) depth++; // @link substring="parent()" target="jdk.incubator.code.CodeElement#parent"
///     IO.println("  ".repeat(depth) + e.getClass());
/// });
///}
///
/// We compute the depth for each code element by traversing back up the code model tree until the root element is
/// reached. So, it is possible to traverse up and down the code model tree.
///
/// A superior way to view the contents of a code model is to convert the root of the code model, an operation, to code
/// model text and print it out.
///
/// {@snippet lang = "java":
/// IO.println(codeModel.toText()); // @link substring="toText()" target="jdk.incubator.code.Op#toText"
///}
///
/// The `toText` method will traverse the code elements in a similar manner as before but print out more detail.
///
/// {@snippet lang = "text":
/// func @loc="22:5:file:///...Example.java" @"add" (
///         %0 : java.type:"int", %1 : java.type:"int")java.type:"int" -> {
///     %2 : Var<java.type:"int"> = var %0 @loc="22:5" @"a";
///     %3 : Var<java.type:"int"> = var %1 @loc="22:5" @"b";
///     %4 : java.type:"java.lang.String" = constant @loc="24:20" @"Example:method:add";
///     invoke %4 @loc="24:9" @java.ref:"java.lang.IO::println(java.lang.Object):void";
///     %5 : java.type:"int" = var.load %2 @loc="25:16";
///     %6 : java.type:"int" = var.load %3 @loc="25:20";
///     %7 : java.type:"int" = add %5 %6 @loc="25:16";
///     return %7 @loc="25:9";
/// };
///}
///
/// The format of code model’s text is unspsecified. It is designed to be human-readable, and intended for debugging and
/// testing. It is also invaluable for explaining code models. To aid debugging each operation has line number
/// information, and the root operation also has source information from where the code model originated.
///
/// The code model text shows the code model’s root element is a function declaration (`func`) operation. The
/// lambda-like expression represents the fusion of the function declaration operation’s single body and the body’s
/// first and only block, called the _entry_ block. Then there is a sequence of operations in the entry block. For each
/// operation there is an instance of a corresponding Java class, all of which extend from the abstract class
/// [Op][jdk.incubator.code.Op] and which have already seen when we printed out the operation classes. The printed
/// operations and printed operation classes occur in the same order since the `toText` method traverses the model in
/// the same order as we explicitly traversed.
///
/// The entry block declares two values called [_block parameters_][jdk.incubator.code.Block.Parameter], `%0` and `%1`,
/// which model the method’s initial values for parameters `a` and `b`. The method parameter declarations are modeled as
/// embedded `var` operations, each initialized with a corresponding block parameter _used_ as the `var` operation’s
/// single _operand_. The `var` operations produce values called [_operation results_][jdk.incubator.code.Op.Result],
/// variable values `%2` and `%3`, which model the variables `a` and `b`. A variable value can be loaded from or stored
/// to using variable access operations, respectively modeling an expression that denotes a variable and assignment to a
/// variable. The expressions denoting parameters `a` and `b` are modeled as `var.load` operations that _use_ the
/// variable values `%2` and `%3` respectively as _operands_. The operation results of these operations are _used_ as
/// _operands_ of subsequent operations and so on, e.g., `%7` the result of the `add` operation modeling the `+`
/// operator is used as an operand of the `return` operation modeling the `return` statement.
///
/// The source code of the `add` method might contain all sorts of syntactic details that `javac` rightly needs to know
/// about but are extraneous for modeling purposes. This complexity is not present in the code model. For example, the
/// same code model would be produced if the return statement’s expression was `((a) + (b))` instead of `a + b`.
///
/// In addition to the code model containing code elements forming a tree it also contains other items called
/// [_code items_][jdk.incubator.code.CodeItem], [values][jdk.incubator.code.Value] (block parameters or operation
/// results) we previously introduced, that form bidirectional dependency graphs between their declaration and their
/// use. A value has a [_type element_][jdk.incubator.code.TypeElement], another code item, modeling the set of all
/// possible values. In our example many of the type elements model Java types, and some model the type of variable
/// values (the type element of the operation result of a var operation). In summary a code model contains five kinds of
/// code item, operation, body, block, value, and type element.
///
/// Code models are in Static Single-Assignment ([SSA][SSA]) form, and there is no explicit distinction, as there is in
/// the source code, between Java [statements][java-statements] and [expressions][java-expressions]. Block parameters
/// and operation results are declared before they are used and cannot be reassigned (and we therefore require special
/// operations and type elements to model variables as previously shown).
///
/// [SSA]: https://en.wikipedia.org/wiki/Static_single-assignment_form
///
/// [java-statements]: https://docs.oracle.com/javase/specs/jls/se25/html/jls-14.html
///
/// [java-expressions]: https://docs.oracle.com/javase/specs/jls/se25/html/jls-15.html
///
/// Finally, we can execute the code model by transforming it to byte code, wrapping it in a method handle, and invoking
/// the handle.
///
/// {@snippet lang = "java":
/// var handle = BytecodeGenerator.generate(MethodHandles.lookup(), addModel); // @link substring="generate(" target="jdk.incubator.code.bytecode.BytecodeGenerator#generate"
/// assert ExampleAdd.add(1, 1) == (int) handle.invokeExact(1, 1);
///}
///
/// ## Building
///
/// Code reflection provides functionality to build code models. We can use the API to build an equivalent model of the
/// `Example.add` method we previously accessed and traversed.
///
/// {@snippet lang = "java":
/// var builtCodeModel = func( // @link substring="func(" target="jdk.incubator.code.dialect.core.CoreOp#func"
///     "add",
///     CoreType.functionType(JavaType.INT, JavaType.INT, JavaType.INT))
///     .body((Block.Builder builder) -> { // @link substring="builder" target="jdk.incubator.code.Block.Builder"
///         // Check the entry block parameters
///         assert builder.parameters().size() == 2;
///         assert builder.parameters().stream().allMatch(
///                 (Block.Parameter param) -> param.type().equals(JavaType.INT));
///
///         // int a
///         VarOp varOpA = var("a", builder.parameters().get(0)); // @link substring="var(" target="jdk.incubator.code.dialect.core.CoreOp#var"
///         Op.Result varA = builder.op(varOpA); // @link substring="builder.op(" target="jdk.incubator.code.Block.Builder#op"
///
///         // int b
///         VarOp varOpB = var("b", builder.parameters().get(1));
///         Op.Result varB = builder.op(varOpB);
///
///         // IO.println("A:method:m")
///         builder.op(invoke(PRINTLN, // // @link substring="inovoke(" target="jdk.incubator.code.dialect.core.JavaOp#invoke"
///                 builder.op(constant(JavaType.J_L_STRING, "A:method:m"))));
///
///         // return a + b;
///         builder.op(return_(
///                 builder.op(add( // @link substring="add(" target="jdk.incubator.code.dialect.java.JavaOp#add"
///                         builder.op(varLoad(varA)),
///                         builder.op(varLoad(varB))))));
///     });
/// IO.println(builtCodeModel.toText());
///}
///
/// The consuming lambda expression passed to the `body` method operates on a
/// [block builder][jdk.incubator.code.Block.Builder], representing the
/// [entry block][jdk.incubator.code.Block#isEntryBlock()] being built. We use that to append operations to the entry
/// block. When an operation is appended it produces an operation result that can be _used_ as an _operand_ of a further
/// operation and so on. When the `body` method returns a [body][jdk.incubator.code.Body] element and the
/// entry [block][jdk.incubator.code.Block] element it contains will be fully built.
///
/// Building, like the text output, mirrors the source code structure. Building is carefully designed so that
/// structurally invalid models cannot be built, either because it is correct by construction or because an exception is
/// produced when given invalid input.
///
/// We can approximately test equivalence with our previously accessed model as follows.
///
/// {@snippet lang = "java":
/// var builtCodeModelElements = builtCodeModel.elements()
///         .map(CodeElement::getClass).toList();
/// var codeModelElements = addModel.elements()
///         .map(CodeElement::getClass).toList();
/// assert builtCodeModelElements.equals(codeModelElements);
///}
///
/// We don’t anticipate most users will commonly build complete models of Java code, since it’s a rather verbose and
/// tedious process, although potentially less so than other approaches e.g., building byte code, or method handle
/// combinators. `Javac` already knows how to build models. In fact, `javac` uses the same API to build models, and the
/// run time uses it to produce models that are accessed. Instead, we anticipate many users will build parts of models
/// when they transform them.
///
/// ## Transforming
///
/// Code reflection supports the transformation of code models by combining traversing and building. A code model
/// transformation is represented by a function that takes an operation, encountered in the (input) model being
/// transformed, and a code model builder for the resulting transformed (output) model, and mediates how, if at all,
/// that operation is transformed into other code elements that are built. We were inspired by the functional
/// [transformation][cf-transformation] approach devised by the Class-File API and adapted that design to work on the
/// nested structure of code models that are immutable trees of code elements.
///
/// [cf-transformation]: https://openjdk.org/jeps/484#Transforming-class-files
///
/// We can write a simple code model transformer that transforms our method’s code model, replacing the operation
/// modeling the `+` operator with an invocation operation modeling an invocation expression to the method
/// `Integer.sum`.
///
/// {@snippet lang = "java":
/// final MethodRef SUM = MethodRef.method(Integer.class, "sum", int.class,
///         int.class, int.class);
/// CodeTransformer addToMethodTransformer = CodeTransformer.opTransformer(( // @link substring="opTransformer(" target="jdk.incubator.code.CodeTransformer#opTransformer"
///         Function<Op, Op.Result> builder,
///         Op inputOp,
///         List<Value> outputOperands) -> {
///     switch (inputOp) {
///         // Replace a + b; with Integer.sum(a, b);
///         case AddOp _ -> builder.apply(invoke(SUM, outputOperands));
///         // Copy operation
///         default -> builder.apply(inputOp);
///     }
/// });
///}
///
/// The code transformation function, passed as lambda expression to
/// [CodeTransformer.opTransformer][jdk.incubator.code.CodeTransformer#opTransformer], accepts as parameters a block
/// builder function, `builder`, an operation encountered when traversing the input code model, `inputOp`, and a list of
/// values in the output model being built that are associated with input operation’s operands, `outputOperands`. We
/// must have previously encountered and transformed the input operations whose results are associated with those
/// values, since values can only be used after they have been declared.
///
/// In the code transformer we switch over the input operation, and in this case we just match on `add` operation and
/// by default any other operation. In the latter case we apply the input operation to the builder function, which
/// creates a new output operation that is a copy of the input operation, appends the new operation to the block being
/// built, and associates the new operation’s result with the input operation’s result. When we match on an `add`
/// operation we replace it by building part of a code model, a method `invoke` operation to the `Integer.sum` method
/// constructed with the given output operands. The result of the output `invoke` operation is automatically associated
/// with the result of the input `add` operation.
///
/// We can then transform the method’s code model by invoking the
/// [FuncOp.transform][jdk.incubator.code.dialect.core.CoreOp.FuncOp#transform] method and passing the code transformer
/// as an argument.
///
/// {@snippet lang = "java":
/// FuncOp transformedCodeModel = codeModel.transform(addToMethodTransformer);
/// IO.println(transformedCodeModel.toText());
///}
///
/// The transformed code model is naturally very similar to the input code model.
///
/// {@snippet lang = "text":
/// func @loc="22:5:file:///...Example.java" @"add" (
///         %0 : java.type:"int", %1 : java.type:"int")java.type:"int" -> {
///     %2 : Var<java.type:"int"> = var %0 @loc="22:5" @"a";
///     %3 : Var<java.type:"int"> = var %1 @loc="22:5" @"b";
///     %4 : java.type:"java.lang.String" = constant @loc="24:20" @"Example:method:add";
///     invoke %4 @loc="24:9" @java.ref:"java.lang.IO::println(java.lang.Object):void";
///     %5 : java.type:"int" = var.load %2 @loc="25:16";
///     %6 : java.type:"int" = var.load %3 @loc="25:20";
///     %7 : java.type:"int" = invoke %5 %6 @java.ref:"java.lang.Integer::sum(int, int):int";
///     return %7 @loc="25:9";
/// };
///}
///
/// We can observe the `add` operation has been replaced with the `invoke` operation. Also, by default, each operation
/// that was copied preserves line number information. The code transformation function can also be applied unmodified
/// to more complex code containing many `+` operators in arbitrarily nested positions.
///
/// ### Transforming primitive
///
/// The code transformation function previously shown is not a direct implementation of functional interface
/// [CodeTransformer][jdk.incubator.code.CodeTransformer]. Instead, we adapted from another functional interface, which
/// is easier to implement for simpler transformations on operations. Direct implementations of `CodeTransformer` are
/// more complex but are also capable of more complex transformations, such as building new blocks and retaining more
/// control over associating values in the input and output models.
///
/// The simple code transformer previously shown can implemented more directly as follows.
///
/// {@snippet lang = "java":
/// CodeTransformer lowLevelAddToMethodTransformer = (
///         Block.Builder builder,
///         Op inputOp) -> {
///     switch (inputOp) {
///         // Replace a + b; with Integer.sum(a, b);
///         case AddOp _ -> {
///             // Get output operands mapped to input op's operands
///             List<Value> outputOperands = builder.context().getValues(inputOp.operands());
///
///             Op.Result r = builder.op(invoke(SUM, outputOperands));
///
///             // Map intput op's result to output result of invocation operation
///             builder.context().mapValue(inputOp.result(), r);
///         }
///         // Copy operation
///         default -> builder.op(inputOp);
///     }
///     // Return the block builder to continue building from for next operation
///     return builder;
/// };
///}
///
/// Here we directly use a [block builder][jdk.incubator.code.Block.Builder]. In the prior example the block builder
/// was hidden behind an implementation of the functional interface `Function<Op, Op.Result>` that manages the mapping
/// of the the input operation's result to the output operation's result.
///
/// Code reflection provides complex code transformers, such as those for
/// - progressively [lowering][jdk.incubator.code.CodeTransformer#LOWERING_TRANSFORMER] code models;
/// - [transforming][jdk.incubator.code.dialect.core.SSA] models into pure SSA-form (where variable related operations
///   are removed);
/// - [inlining][jdk.incubator.code.dialect.core.Inliner] models into other models;
/// - [normalizing][jdk.incubator.code.dialect.core.NormalizeBlocksTransformer] to remove redundant blocks; and
/// - constant folding operations modeling Java expressions that are constant expressions.
///
/// Crucially, all of the above code transformers preserve the program behaviour of the input code model. However, in
/// general, code transformers are not required to preserve program behaviour and some will intentionally not do so as
/// they may transform into a different ouput programming domain that partially maps from the input programming domain.
///
/// ## Code model structure
///
/// The primary data structure of a code model is a tree of [code elements][jdk.incubator.code.CodeElement]. There are
/// three kinds of code elements, [operation][jdk.incubator.code.Op], [body][jdk.incubator.code.Body], and
/// [block][jdk.incubator.code.Block]. An operation contains a [sequence][jdk.incubator.code.Op#bodies()] of zero or
/// more bodies. A body contains a [sequence][jdk.incubator.code.Body#blocks()] of one or more blocks. A block contains
/// a [sequence][jdk.incubator.code.Block#ops()] of one or more operations. The root of a code model is an operation,
/// and descendant operations form a tree of arbitrary depth.
///
/// ### Bodies and blocks
///
/// The first block in a body is called the [entry block][jdk.incubator.code.Block#isEntryBlock()]. The last operation
/// in a block is called a [terminating operation][jdk.incubator.code.Op.Terminating]. A terminating operation describes
/// how control is passed from the operation’s [parent][jdk.incubator.code.Op#parent()] block to an ancestor operation,
/// or how control is passed from the operation’s parent block to one or more non-entry sibling blocks. In the latter
/// case the terminating operation declares a [sequence][jdk.incubator.code.Op#successors()] of one or more
/// [references][jdk.incubator.code.Block.Reference] to blocks called successors.
///
/// Block references form a data structure that is a control flow graph, where blocks are nodes in the graph and
/// references are directed edges in the graph. Blocks in a body occur in the same order as that produced by
/// topological sorting the control flow graph in reverse post-order, where the entry block always occurs first (since
/// it cannot be referenced) and any unreferenced blocks occur last (in any order). This ensures blocks are by default
/// traversed in a program order.
///
/// ### Block parameters, operation results, and values
///
/// A block declares a [sequence][jdk.incubator.code.Block#parameters()] of zero or more
/// [block parameters][jdk.incubator.code.Block.Parameter], values. An operation declares an
/// [operation result][jdk.incubator.code.Op.Result], also a value. [Values][jdk.incubator.code.Value] are variables
/// assigned exactly once, so code models are in static single-assignment (SSA) form.
///
/// A value declares a [type element][jdk.incubator.code.TypeElement], describing the set of values the value is a
/// member of.
///
/// An operation uses zero or more values in a [sequence][jdk.incubator.code.Op#operands()] of operands and in a
/// [sequence][jdk.incubator.code.Block.Reference#arguments()] of block arguments of any block references (the
/// same value may be used more than once in both cases). A value can be used by an operation only after it has been
/// declared and only if the operation's result is
/// [dominated by][jdk.incubator.code.Value#isDominatedBy(jdk.incubator.code.Value)] the value.
///
/// The [declaring block][jdk.incubator.code.Value#declaringBlock()] of a value that is a block parameter is the block
/// that declares the block parameter. The declaring block of a value that is an operation result is the parent block
/// of the operation result’s operation.
///
/// Values form two data structures that are dependency graphs and use graphs. Given a value we can ask what values this
/// value [depends on][jdk.incubator.code.Value#dependsOn()], and so on, to form a dependency graph. Or inversely we can
/// ask what operation results [use][jdk.incubator.code.Value#uses()] this value in their operations' operands and
/// successors (or what values depend on this value), and so on, to form a use graph.
///
/// ## Code model behavior
///
/// A code model's program behaviour is described by the arrangement of operations, bodies, and blocks. This
/// arrangement has generic program behaviour common to all code models and specific program behaviour of a code model,
/// giving rise to program meaning, as specified each operation's modeling of program behavior and the arrangement of
/// the operations within in a code model.
///
/// ### Environment and effects
///
/// We can describe the generic program behaviour in terms of an environment where code elements are executed. Execution
/// of a code element produces an effect that is used to update the environment and pass control to another code
/// element.
///
/// There are three kinds of effect:
///
/// 1. An operation result effect, containing a runtime value for the operation result
/// 2. A terminating operation effect, containing a terminating operation and runtime values for the operation's
///    operands
/// 3. A successor block effect, containing the successor block and runtime values for the block's arguments.
///
/// Each effect also contains the environment that the effect takes place in. The environment can be updated to a new
/// environment by binding runtime values to (symbolic) values. The environment can be used to access runtime values
/// given (symbolic) values.
///
/// ### Execution of operations
///
/// Execution of a non-terminating operation either completes _normally_ or _abruptly_, according to its specification.
/// If an operation completes normally it produces an operation result effect. If an operation completes abruptly it
/// produces a terminating operation effect.
///
/// Execution of a terminating operation may produce a successor effect or a terminating operation effect, according to
/// its specification, An operation that is an of instance of [jdk.incubator.code.Op.BlockTerminating] can produce a
/// successor effect. An operation that is an of instance of [jdk.incubator.code.Op.BodyTerminating] can produce a
/// terminating operation effect.
///
/// If an operation has one or more bodies it may execute them according to its specification. The effect produced by
/// executing a body may be used to determine whether to execute further bodies and so on until execution of the
/// operation completes (normally or abruptly) and produces its own effect.
///
/// ### Execution of bodies
///
/// Execution of a body produces a terminating operation effect.
///
/// Execution first proceeds by selecting the body's entry block for execution with given runtime values as arguments
/// for the entry block's parameters.
///
/// The current environment is updated by binding the selected block's parameters to the given runtime values, then the
/// selected block is executed:
///
/// - If execution produces a successor effect then the effect's successor block becomes the selected block, the
/// effect's runtime values become the given runtime values, and the effect's environment becomes the current
/// environment. Execution of the selected block then proceeds in the same manner as previously described.
///
/// - If execution of the selected block produces a terminating operation effect then execution of the body completes
/// and it produces that effect (passing the effect to execution of the parent operation, which may result in execution
/// completing abruptly).
///
/// ### Execution of blocks
///
/// Execution of a block produces a successor effect or a terminating operation effect.
///
/// Execution first proceeds by selecting the first operation in the block.
///
/// If the selected operation is a non-terminating operation then the non-terminating operation is executed:
///
/// - If execution of the operation produces an operation result effect then the next operation becomes the selected
/// operation, and the current environment is updated by binding the operation's result to the effect's runtime value.
/// Execution of the selected operation then proceeds as previously described.
///
/// - If execution of the operation produces a terminating operation effect then execution of the block completes and
/// it produces that effect (passing the effect to execution of the parent body, which in turn passes the effect to the
/// execution of the parent operation, and which may result in execution completing abruptly).
///
/// If the selected operation is a terminating operation then the terminating operation is executed, producing an
/// effect, and execution of the block completes with that effect (passing the effect to execution of the parent body).
///
/// ## Dialects
///
/// ## Java code models
///
/// ## Modeling Java code
///
package jdk.incubator.code;
