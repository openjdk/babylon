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
/// {@snippet lang="java" :
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
/// }
///
/// We can write a simple stream that uses core reflection and traverses program structure, a tree of annotated
/// elements, starting from a given class and reporting elements in a topological order.
///
/// {@snippet lang="java" :
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
/// }
///
/// ([AnnotatedElement][java.lang.reflect.AnnotatedElement] is the common super type of [Class][java.lang.Class],
/// [Field][java.lang.reflect.Field], and [Method][java.lang.reflect.Method].)
///
/// The `traverse` method recursively traverses a class's declared fields, methods, and enclosed class. Starting from
/// `Example`, using a class literal expression, we can print out the fields, methods, and classes we encounter.
///
/// {@snippet lang="java" :
/// elements(Example.class)
///     .forEach(IO::println);
/// }
///
/// More interestingly we can perform some simple analysis, such as counting the number of static fields whose type is
/// [Runnable][java.lang.Runnable].
///
/// {@snippet lang="java" :
/// static boolean isStaticRunnableField(Field f) {
///     return f.accessFlags().contains(AccessFlag.STATIC)
///         && Runnable.class.isAssignableFrom(f.getType());
/// }
/// assert 2 == elements(Example.class)
///     .filter(e -> e instanceof Field f && isStaticRunnableField(f))
///     .count();
/// }
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
/// {@snippet lang="java" :
/// import jdk.incubator.code.*;
/// import jdk.incubator.code.bytecode.*;
/// import jdk.incubator.code.dialect.core.*;
/// import jdk.incubator.code.dialect.java.*;
/// import static jdk.incubator.code.dialect.core.CoreOp.*;
/// import static jdk.incubator.code.dialect.java.JavaOp.*;
///
/// static class Example {
///     @Reflect
///     static Runnable R = () -> IO.println("Example:field:R");
///     @Reflect
///     static int add(int a, int b) {
///         IO.println("Example:method:add");
///         return a + b;
///     }
///
///     static class Nested {
///         @Reflect
///         static Runnable R = () -> IO.println("Example.Nested:field:R");
///         @Reflect
///         void m() { IO.println("Example.Nested:method:m"); }
///     }
/// }
/// }
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
/// {@snippet lang="java" :
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
/// }
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
/// {@snippet lang="java" :
/// elements(Example.class)
///         // AnnotatedElement -> CodeModel?
///         .flatMap(ae -> getCodeModel(ae).stream())
///         .forEach(IO::println);
/// }
///
/// More interestingly we can now perform some simple analysis of code, such as extracting the values of the string
/// literal expressions that are printed.
///
/// {@snippet lang="java" :
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
/// }
///
/// The method `analyzeCodeModel` uses a stream to [traverse](#traversing-heading) over all elements of a code model and
/// returns the list of string literal values passed to invocations of `IO.println`. We can then use `analyzeCodeModel`
/// to further refine our steam expression to print out all such string literal values.
///
/// {@snippet lang="java" :
/// elements(Example.class)
///         // AnnotatedElement -> CodeModel?
///         .flatMap(ae -> getCodeModel(ae).stream())
///         // CodeModel -> List<String>
///         .map(codeModel -> analyzeCodeModel(codeModel))
///         .forEach(IO::println);
/// }
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
/// {@snippet lang="java" :
/// Method addMethod = Example.class.getDeclaredMethod("add", int.class, int.class);
/// CoreOp.FuncOp codeModel = Op.ofMethod(addMethod).orElseThrow(); // @link substring="CoreOp.FuncOp" target="jdk.incubator.code.dialect.core.CoreOp.FuncOp"
/// assert codeModel == Op.ofMethod(addMethod).orElseThrow();
/// }
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
/// {@snippet lang="java" :
/// Field rField = Example.class.getDeclaredField("R", Runnable.class);
/// Object rFieldInstance = rField.get(null);
/// Quoted<JavaOp.LambdaOp> quotedCodeModel = Op.ofLambda(rFieldInstance).orElseThrow(); // @link substring="Quoted" target="jdk.incubator.code.Quoted"
/// JavaOp.LambdaOp codeModel = quotedCodeModel.op(); // @link substring="JavaOp.LambdaOp" target="jdk.incubator.code.dialect.java.JavaOp.LambdaOp"
/// }
///
/// If a lambda expression captures values we can additionally access those values. For example,
///
/// {@snippet lang="java" :
/// int capture = 42;
/// @Reflect
/// Runnable r = () -> IO.println(capture);
/// Quoted<JavaOp.LambdaOp> quotedCodeModel = Op.ofLambda(r).orElseThrow();
///
/// SequencedMap<Value, Object> capturedValues = quotedCodeModel.capturedValues(); // @link substring="capturedValues()" target="jdk.incubator.code.Quoted#capturedValues"
/// assert capturedValues.size() == 1;
/// assert capturedValues.values().contains(42);
/// }
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
/// {@snippet lang="java" :
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
/// }
///
/// The `traverse` method prints out the class of the code element it encounters and prefixes that with white space
/// proportionate to the depth of the element in the code model tree. Invoking the `traverse` method with the code model
/// of the `Example.add` method prints out the following:
///
/// {@snippet lang="text" :
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
/// }
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
/// {@snippet lang="java" :
/// codeModel.elements().forEach((CodeElement<?, ?> e) -> {
///     int depth = 0;
///     var parent = e;
///     while ((parent = parent.parent()) != null) depth++; // @link substring="parent()" target="jdk.incubator.code.CodeElement#parent"
///     IO.println("  ".repeat(depth) + e.getClass());
/// });
/// }
///
/// We compute the depth for each code element by traversing back up the code model tree until the root element is
/// reached. So, it is possible to traverse up and down the code model tree.
///
/// A superior way to view the contents of a code model is to convert the root of the code model, an operation, to code
/// model text and print it out.
///
/// {@snippet lang="java" :
/// IO.println(codeModel.toText()); // @link substring="toText()" target="jdk.incubator.code.Op#toText"
/// }
///
/// The `toText` method will traverse the code elements in a similar manner as before but print out more detail.
///
/// {@snippet lang="text" :
/// func @loc="22:5:string:///...Example.java" @"add" (
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
/// }
///
/// A code model’s text is designed to be human-readable, primarily intended for debugging and testing. It is also
/// invaluable for explaining code models. To aid debugging each operation has line number information, and the root
/// operation also has source information from where the code model originated.
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
/// {@snippet lang="java" :
/// var handle = BytecodeGenerator.generate(MethodHandles.lookup(), addModel); // @link substring="generate(" target="jdk.incubator.code.bytecode.BytecodeGenerator#generate"
/// assert ExampleAdd.add(1, 1) == (int) handle.invokeExact(1, 1);
/// }
///
/// ## Building
///
/// ## Transforming
///
/// ## Code model structure
///
/// ## Code model behavior
///
/// ## Dialects
///
/// ## Java code models
///
/// ## Modeling Java code
///
package jdk.incubator.code;
