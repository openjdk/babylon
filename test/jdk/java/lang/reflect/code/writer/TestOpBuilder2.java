import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.DialectFactory;
import jdk.incubator.code.internal.OpBuilder2;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @modules jdk.incubator.code/jdk.incubator.code.internal
 * @run junit TestOpBuilder2
 */
public class TestOpBuilder2 { // tests copied from TestCodeBuilder

    @CodeReflection
    static void constants() {
        boolean bool = false;
        byte b = 1;
        char c = 'a';
        short s = 1;
        int i = 1;
        long l = 1L;
        float f = 1.0f;
        double d = 1.0;
        String str = "1";
        Object obj = null;
        Class<?> klass = Object.class;
    }

    @Test
    public void testConstants() {
        testWithTransforms(getFuncOp("constants"));
    }

    static record X(int f) {
        void m() {}
    }

    @CodeReflection
    static void reflect() {
        X x = new X(1);
        int i = x.f;
        x.m();
        X[] ax = new X[1];
        int l = ax.length;
        x = ax[0];

        Object o = x;
        x = (X) o;
        if (o instanceof X) {
            return;
        }
        if (o instanceof X(var a)) {
            return;
        }
    }

    @Test
    public void testReflect() {
        testWithTransforms(getFuncOp("reflect"));
    }

    @CodeReflection
    static int bodies(int m, int n) {
        int sum = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum += i + j;
            }
        }
        return m > 10 ? sum : 0;
    }

    @Test
    public void testBodies() {
        testWithTransforms(getFuncOp("bodies"));
    }

    public void testWithTransforms(CoreOp.FuncOp f) {
        test(f);

        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        test(f);

        f = SSA.transform(f);
        test(f);
    }

    static void test(CoreOp.FuncOp f) {
        CoreOp.ModuleOp moduleOp = OpBuilder2.createBuilderFunction("fb", f,
                b -> b.parameter(JavaType.type(DialectFactory.class)));
        moduleOp = moduleOp.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(moduleOp.functionTable().get("$exterType").toText());

        CoreOp.FuncOp fb = moduleOp.functionTable().get("fb");
        var actual = (CoreOp.FuncOp) Interpreter.invoke(MethodHandles.lookup(), fb, JavaOp.JAVA_DIALECT_FACTORY);

        Assertions.assertEquals(f.toText(), actual.toText());
    }

    CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(this.getClass().getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }

    // {long=[0], java.type.primitive=[0, 1], TestOpBuilder2=[2], java.type.class=[2, 1, 3], Var=[1, 4]}

    // primitive has been used as identifier (so far)

    // ExternalizedTypeElement -> id, args_indexes_in_same_collection, can be empty

    /* simulation of the map content
    // input is a type
    // before we append op to call the helper method, we should know what index to pass
    // we can check if the collection has the type / externalized type
    // what about the args the type needs
    // if the arg is in the collection use its index
    // if it's not process it the same as the parent type

    // we just need to make sure the collection is built right
    registerType(Type) : int // returns type index in the map, the index can be passed to the helper method
        exType = type.externalize()
        if exType not in map
            deps = []
            for arg in extType.args
                deps.add(typeIndex(arg))
            map.add(exType -> deps)
        return index(exType)
    [0], ex_type_0 -> "long", [], as it has no args
    [1], ex_type_1 -> "primitive", [0]
    [2], ex_type_2 -> "TestOpBuilder2", []

    * */

    /*  (primitive, long)
    %4 : java.type:"java.lang.String" = constant @"long";
    %5 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %4 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String):jdk.incubator.code.extern.ExternalizedTypeElement";
    %6 : java.type:"java.lang.String" = constant @"java.type.primitive";
    %7 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %6 %5 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String, jdk.incubator.code.extern.ExternalizedTypeElement):jdk.incubator.code.extern.ExternalizedTypeElement";
    %8 : java.type:"jdk.incubator.code.TypeElement" = invoke %2 %7 @java.ref:"jdk.incubator.code.extern.TypeElementFactory::constructType(jdk.incubator.code.extern.ExternalizedTypeElement):jdk.incubator.code.TypeElement";
    * */

    /* (class, TestOpBuilder2, (primitive, void))
    %12 : java.type:"java.lang.String" = constant @"TestOpBuilder2";
    %13 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %12 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String):jdk.incubator.code.extern.ExternalizedTypeElement";
    %14 : java.type:"java.lang.String" = constant @"void";
    %15 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %14 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String):jdk.incubator.code.extern.ExternalizedTypeElement";
    %16 : java.type:"java.lang.String" = constant @"java.type.primitive";
    %17 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %16 %15 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String, jdk.incubator.code.extern.ExternalizedTypeElement):jdk.incubator.code.extern.ExternalizedTypeElement";
    %18 : java.type:"java.lang.String" = constant @"java.type.class";
    %19 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %18 %13 %17 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String, jdk.incubator.code.extern.ExternalizedTypeElement, jdk.incubator.code.extern.ExternalizedTypeElement):jdk.incubator.code.extern.ExternalizedTypeElement";
    %20 : java.type:"jdk.incubator.code.TypeElement" = invoke %2 %19 @java.ref:"jdk.incubator.code.extern.TypeElementFactory::constructType(jdk.incubator.code.extern.ExternalizedTypeElement):jdk.incubator.code.TypeElement";
    * */

    /*
    %22 : java.type:"java.lang.String" = constant @"int";
    %23 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %22 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String):jdk.incubator.code.extern.ExternalizedTypeElement";
    %24 : java.type:"java.lang.String" = constant @"java.type.primitive";
    %25 : java.type:"jdk.incubator.code.extern.ExternalizedTypeElement" = invoke %24 %23 @java.ref:"jdk.incubator.code.extern.ExternalizedTypeElement::of(java.lang.String, jdk.incubator.code.extern.ExternalizedTypeElement):jdk.incubator.code.extern.ExternalizedTypeElement";
    * */

    /*
    func @"gb" (%0 : java.type:"jdk.incubator.code.extern.DialectFactory")java.type:"jdk.incubator.code.Op" -> {
    %1 : java.type:"jdk.incubator.code.extern.OpFactory" = invoke %0 @java.ref:"jdk.incubator.code.extern.DialectFactory::opFactory():jdk.incubator.code.extern.OpFactory";
    %2 : java.type:"jdk.incubator.code.extern.TypeElementFactory" = invoke %0 @java.ref:"jdk.incubator.code.extern.DialectFactory::typeElementFactory():jdk.incubator.code.extern.TypeElementFactory";
    %3 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %4 : java.type:"int" = constant @1;
    %5 : java.type:"jdk.incubator.code.TypeElement" = invoke %4 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %6 : java.type:"jdk.incubator.code.dialect.core.FunctionType" = invoke %5 @java.ref:"jdk.incubator.code.dialect.core.CoreType::functionType(jdk.incubator.code.TypeElement, jdk.incubator.code.TypeElement[]):jdk.incubator.code.dialect.core.FunctionType" @invoke.kind="STATIC" @invoke.varargs=true;
    %7 : java.type:"jdk.incubator.code.Body::Builder" = invoke %3 %6 @java.ref:"jdk.incubator.code.Body::Builder::of(jdk.incubator.code.Body::Builder, jdk.incubator.code.dialect.core.FunctionType):jdk.incubator.code.Body::Builder";
    %8 : java.type:"jdk.incubator.code.Block::Builder" = invoke %7 @java.ref:"jdk.incubator.code.Body::Builder::entryBlock():jdk.incubator.code.Block::Builder";
    %9 : java.type:"int" = constant @5;
    %10 : java.type:"jdk.incubator.code.TypeElement" = invoke %9 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %11 : java.type:"jdk.incubator.code.Block$Parameter" = invoke %8 %10 @java.ref:"jdk.incubator.code.Block::Builder::parameter(jdk.incubator.code.TypeElement):jdk.incubator.code.Block$Parameter";
    %12 : java.type:"int" = constant @7;
    %13 : java.type:"jdk.incubator.code.TypeElement" = invoke %12 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %14 : java.type:"jdk.incubator.code.Block$Parameter" = invoke %8 %13 @java.ref:"jdk.incubator.code.Block::Builder::parameter(jdk.incubator.code.TypeElement):jdk.incubator.code.Block$Parameter";
    %15 : java.type:"int" = constant @7;
    %16 : java.type:"jdk.incubator.code.TypeElement" = invoke %15 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %17 : java.type:"jdk.incubator.code.Block$Parameter" = invoke %8 %16 @java.ref:"jdk.incubator.code.Block::Builder::parameter(jdk.incubator.code.TypeElement):jdk.incubator.code.Block$Parameter";
    %18 : java.type:"java.lang.String" = constant @"var";
    %19 : java.type:"int" = constant @18;
    %20 : java.type:"int" = constant @5;
    %21 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %22 : java.type:"int" = constant @8;
    %23 : java.type:"jdk.incubator.code.TypeElement" = invoke %22 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %24 : java.type:"java.lang.String" = constant @"a";
    %25 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %26 : java.type:"jdk.incubator.code.Op$Result" = invoke %8 %18 %19 %20 %14 %21 %23 %24 %25 @java.ref:"TestOpBuilder2::::op(jdk.incubator.code.Block::Builder, java.lang.String, int, int, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op$Result";
    %27 : java.type:"java.lang.String" = constant @"var";
    %28 : java.type:"int" = constant @18;
    %29 : java.type:"int" = constant @5;
    %30 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %31 : java.type:"int" = constant @8;
    %32 : java.type:"jdk.incubator.code.TypeElement" = invoke %31 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %33 : java.type:"java.lang.String" = constant @"b";
    %34 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %35 : java.type:"jdk.incubator.code.Op$Result" = invoke %8 %27 %28 %29 %17 %30 %32 %33 %34 @java.ref:"TestOpBuilder2::::op(jdk.incubator.code.Block::Builder, java.lang.String, int, int, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op$Result";
    %36 : java.type:"java.lang.String" = constant @"var.load";
    %37 : java.type:"int" = constant @20;
    %38 : java.type:"int" = constant @16;
    %39 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %40 : java.type:"int" = constant @7;
    %41 : java.type:"jdk.incubator.code.TypeElement" = invoke %40 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %42 : java.type:"java.util.Map" = constant @null;
    %43 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %44 : java.type:"jdk.incubator.code.Op$Result" = invoke %8 %36 %37 %38 %26 %39 %41 %42 %43 @java.ref:"TestOpBuilder2::::op(jdk.incubator.code.Block::Builder, java.lang.String, int, int, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op$Result";
    %45 : java.type:"java.lang.String" = constant @"var.load";
    %46 : java.type:"int" = constant @20;
    %47 : java.type:"int" = constant @20;
    %48 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %49 : java.type:"int" = constant @7;
    %50 : java.type:"jdk.incubator.code.TypeElement" = invoke %49 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %51 : java.type:"java.util.Map" = constant @null;
    %52 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %53 : java.type:"jdk.incubator.code.Op$Result" = invoke %8 %45 %46 %47 %35 %48 %50 %51 %52 @java.ref:"TestOpBuilder2::::op(jdk.incubator.code.Block::Builder, java.lang.String, int, int, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op$Result";
    %54 : java.type:"java.lang.String" = constant @"add";
    %55 : java.type:"int" = constant @20;
    %56 : java.type:"int" = constant @16;
    %57 : java.type:"java.util.List<jdk.incubator.code.Value>" = invoke %44 %53 @java.ref:"java.util.List::of(java.lang.Object, java.lang.Object):java.util.List";
    %58 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %59 : java.type:"int" = constant @7;
    %60 : java.type:"jdk.incubator.code.TypeElement" = invoke %59 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %61 : java.type:"java.util.Map" = constant @null;
    %62 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %63 : java.type:"jdk.incubator.code.Op$Result" = invoke %8 %54 %55 %56 %57 %58 %60 %61 %62 @java.ref:"TestOpBuilder2::::op(jdk.incubator.code.Block::Builder, java.lang.String, int, int, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op$Result";
    %64 : java.type:"java.lang.String" = constant @"conv";
    %65 : java.type:"int" = constant @20;
    %66 : java.type:"int" = constant @9;
    %67 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %68 : java.type:"int" = constant @1;
    %69 : java.type:"jdk.incubator.code.TypeElement" = invoke %68 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %70 : java.type:"java.util.Map" = constant @null;
    %71 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %72 : java.type:"jdk.incubator.code.Op$Result" = invoke %8 %64 %65 %66 %63 %67 %69 %70 %71 @java.ref:"TestOpBuilder2::::op(jdk.incubator.code.Block::Builder, java.lang.String, int, int, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op$Result";
    %73 : java.type:"java.lang.String" = constant @"return";
    %74 : java.type:"int" = constant @20;
    %75 : java.type:"int" = constant @9;
    %76 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %77 : java.type:"int" = constant @4;
    %78 : java.type:"jdk.incubator.code.TypeElement" = invoke %77 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %79 : java.type:"java.util.Map" = constant @null;
    %80 : java.type:"jdk.incubator.code.Body::Builder" = constant @null;
    %81 : java.type:"jdk.incubator.code.Op$Result" = invoke %8 %73 %74 %75 %72 %76 %78 %79 %80 @java.ref:"TestOpBuilder2::::op(jdk.incubator.code.Block::Builder, java.lang.String, int, int, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op$Result";
    %82 : java.type:"java.lang.String" = constant @"func";
    %83 : java.type:"java.lang.String" = constant @"file:///Users/mabbay/babylon/test/jdk/java/lang/reflect/code/writer/TestOpBuilder2.java";
    %84 : java.type:"int" = constant @18;
    %85 : java.type:"int" = constant @5;
    %86 : java.type:"jdk.incubator.code.Location" = new %83 %84 %85 @java.ref:"jdk.incubator.code.Location::(java.lang.String, int, int)";
    %87 : java.type:"jdk.incubator.code.Value" = constant @null;
    %88 : java.type:"jdk.incubator.code.Block$Reference" = constant @null;
    %89 : java.type:"int" = constant @4;
    %90 : java.type:"jdk.incubator.code.TypeElement" = invoke %89 @java.ref:"TestOpBuilder2::$type(int):jdk.incubator.code.TypeElement";
    %91 : java.type:"java.lang.String" = constant @"g";
    %92 : java.type:"jdk.incubator.code.Op" = invoke %82 %86 %87 %88 %90 %91 %7 @java.ref:"TestOpBuilder2::::op(java.lang.String, jdk.incubator.code.Location, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object):jdk.incubator.code.Op";
    invoke %92 @java.ref:"jdk.incubator.code.Op::seal():void";
    return %92;
};
    * */

    /*
    private static jdk.incubator.code.Op TestOpBuilder2::g(int, int):int();
    descriptor: ()Ljdk/incubator/code/Op;
    flags: (0x100a) ACC_PRIVATE, ACC_STATIC, ACC_SYNTHETIC
    Code:
      stack=11, locals=18, args_size=0
         0: getstatic     #1                  // Field jdk/incubator/code/dialect/java/JavaOp.JAVA_DIALECT_FACTORY:Ljdk/incubator/code/extern/DialectFactory;
         3: astore_0
         4: aload_0
         5: invokevirtual #7                  // Method jdk/incubator/code/extern/DialectFactory.opFactory:()Ljdk/incubator/code/extern/OpFactory;
         8: astore_1
         9: aload_0
        10: invokevirtual #13                 // Method jdk/incubator/code/extern/DialectFactory.typeElementFactory:()Ljdk/incubator/code/extern/TypeElementFactory;
        13: astore_2
        14: ldc           #17                 // String java.type.primitive
        16: ldc           #19                 // String int
        18: invokestatic  #21                 // Method jdk/incubator/code/extern/ExternalizedTypeElement.of:(Ljava/lang/String;)Ljdk/incubator/code/extern/ExternalizedTypeElement;
        21: invokestatic  #27                 // Method jdk/incubator/code/extern/ExternalizedTypeElement.of:(Ljava/lang/String;Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/extern/ExternalizedTypeElement;
        24: astore_3
        25: aload_2
        26: aload_3
        27: invokeinterface #30,  2           // InterfaceMethod jdk/incubator/code/extern/TypeElementFactory.constructType:(Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/TypeElement;
        32: astore        4
        34: aconst_null
        35: aload         4
        37: iconst_0
        38: anewarray     #36                 // class jdk/incubator/code/TypeElement
        41: invokestatic  #38                 // InterfaceMethod jdk/incubator/code/dialect/core/CoreType.functionType:(Ljdk/incubator/code/TypeElement;[Ljdk/incubator/code/TypeElement;)Ljdk/incubator/code/dialect/core/FunctionType;
        44: invokestatic  #44                 // Method jdk/incubator/code/Body$Builder.of:(Ljdk/incubator/code/Body$Builder;Ljdk/incubator/code/dialect/core/FunctionType;)Ljdk/incubator/code/Body$Builder;
        47: astore        5
        49: aload         5
        51: invokevirtual #49                 // Method jdk/incubator/code/Body$Builder.entryBlock:()Ljdk/incubator/code/Block$Builder;
        54: astore        6
        56: ldc           #17                 // String java.type.primitive
        58: ldc           #53                 // String void
        60: invokestatic  #21                 // Method jdk/incubator/code/extern/ExternalizedTypeElement.of:(Ljava/lang/String;)Ljdk/incubator/code/extern/ExternalizedTypeElement;
        63: invokestatic  #27                 // Method jdk/incubator/code/extern/ExternalizedTypeElement.of:(Ljava/lang/String;Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/extern/ExternalizedTypeElement;
        66: astore        7
        68: aload         6
        70: aload_2
        71: ldc           #55                 // String java.type.class
        73: ldc           #57                 // String TestOpBuilder2
        75: invokestatic  #21                 // Method jdk/incubator/code/extern/ExternalizedTypeElement.of:(Ljava/lang/String;)Ljdk/incubator/code/extern/ExternalizedTypeElement;
        78: aload         7
        80: invokestatic  #59                 // Method jdk/incubator/code/extern/ExternalizedTypeElement.of:(Ljava/lang/String;Ljdk/incubator/code/extern/ExternalizedTypeElement;Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/extern/ExternalizedTypeElement;
        83: invokeinterface #30,  2           // InterfaceMethod jdk/incubator/code/extern/TypeElementFactory.constructType:(Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/TypeElement;
        88: invokevirtual #62                 // Method jdk/incubator/code/Block$Builder.parameter:(Ljdk/incubator/code/TypeElement;)Ljdk/incubator/code/Block$Parameter;
        91: pop
        92: aload         6
        94: aload         4
        96: invokevirtual #62                 // Method jdk/incubator/code/Block$Builder.parameter:(Ljdk/incubator/code/TypeElement;)Ljdk/incubator/code/Block$Parameter;
        99: astore        8
       101: aload         6
       103: aload         4
       105: invokevirtual #62                 // Method jdk/incubator/code/Block$Builder.parameter:(Ljdk/incubator/code/TypeElement;)Ljdk/incubator/code/Block$Parameter;
       108: astore        9
       110: aload_2
       111: ldc           #68                 // String Var
       113: aload_3
       114: invokestatic  #27                 // Method jdk/incubator/code/extern/ExternalizedTypeElement.of:(Ljava/lang/String;Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/extern/ExternalizedTypeElement;
       117: invokeinterface #30,  2           // InterfaceMethod jdk/incubator/code/extern/TypeElementFactory.constructType:(Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/TypeElement;
       122: astore        10
       124: aload         6
       126: aload_1
       127: new           #70                 // class jdk/incubator/code/extern/ExternalizedOp
       130: dup
       131: ldc           #72                 // String var
       133: ldc           #74                 // String 11:5
       135: invokestatic  #76                 // Method jdk/incubator/code/Location.fromString:(Ljava/lang/String;)Ljdk/incubator/code/Location;
       138: aload         8
       140: invokestatic  #82                 // InterfaceMethod java/util/List.of:(Ljava/lang/Object;)Ljava/util/List;
       143: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       146: aload         10
       148: ldc           #90                 // String
       150: ldc           #92                 // String a
       152: invokestatic  #94                 // InterfaceMethod java/util/Map.of:(Ljava/lang/Object;Ljava/lang/Object;)Ljava/util/Map;
       155: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       158: invokespecial #99                 // Method jdk/incubator/code/extern/ExternalizedOp."<init>":(Ljava/lang/String;Ljdk/incubator/code/Location;Ljava/util/List;Ljava/util/List;Ljdk/incubator/code/TypeElement;Ljava/util/Map;Ljava/util/List;)V
       161: invokeinterface #103,  2          // InterfaceMethod jdk/incubator/code/extern/OpFactory.constructOp:(Ljdk/incubator/code/extern/ExternalizedOp;)Ljdk/incubator/code/Op;
       166: invokevirtual #109                // Method jdk/incubator/code/Block$Builder.op:(Ljdk/incubator/code/Op;)Ljdk/incubator/code/Op$Result;
       169: astore        11
       171: aload         6
       173: aload_1
       174: new           #70                 // class jdk/incubator/code/extern/ExternalizedOp
       177: dup
       178: ldc           #72                 // String var
       180: ldc           #74                 // String 11:5
       182: invokestatic  #76                 // Method jdk/incubator/code/Location.fromString:(Ljava/lang/String;)Ljdk/incubator/code/Location;
       185: aload         9
       187: invokestatic  #82                 // InterfaceMethod java/util/List.of:(Ljava/lang/Object;)Ljava/util/List;
       190: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       193: aload         10
       195: ldc           #90                 // String
       197: ldc           #113                // String b
       199: invokestatic  #94                 // InterfaceMethod java/util/Map.of:(Ljava/lang/Object;Ljava/lang/Object;)Ljava/util/Map;
       202: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       205: invokespecial #99                 // Method jdk/incubator/code/extern/ExternalizedOp."<init>":(Ljava/lang/String;Ljdk/incubator/code/Location;Ljava/util/List;Ljava/util/List;Ljdk/incubator/code/TypeElement;Ljava/util/Map;Ljava/util/List;)V
       208: invokeinterface #103,  2          // InterfaceMethod jdk/incubator/code/extern/OpFactory.constructOp:(Ljdk/incubator/code/extern/ExternalizedOp;)Ljdk/incubator/code/Op;
       213: invokevirtual #109                // Method jdk/incubator/code/Block$Builder.op:(Ljdk/incubator/code/Op;)Ljdk/incubator/code/Op$Result;
       216: astore        12
       218: aload         6
       220: aload_1
       221: new           #70                 // class jdk/incubator/code/extern/ExternalizedOp
       224: dup
       225: ldc           #115                // String var.load
       227: ldc           #117                // String 13:16
       229: invokestatic  #76                 // Method jdk/incubator/code/Location.fromString:(Ljava/lang/String;)Ljdk/incubator/code/Location;
       232: aload         11
       234: invokestatic  #82                 // InterfaceMethod java/util/List.of:(Ljava/lang/Object;)Ljava/util/List;
       237: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       240: aload         4
       242: invokestatic  #119                // InterfaceMethod java/util/Map.of:()Ljava/util/Map;
       245: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       248: invokespecial #99                 // Method jdk/incubator/code/extern/ExternalizedOp."<init>":(Ljava/lang/String;Ljdk/incubator/code/Location;Ljava/util/List;Ljava/util/List;Ljdk/incubator/code/TypeElement;Ljava/util/Map;Ljava/util/List;)V
       251: invokeinterface #103,  2          // InterfaceMethod jdk/incubator/code/extern/OpFactory.constructOp:(Ljdk/incubator/code/extern/ExternalizedOp;)Ljdk/incubator/code/Op;
       256: invokevirtual #109                // Method jdk/incubator/code/Block$Builder.op:(Ljdk/incubator/code/Op;)Ljdk/incubator/code/Op$Result;
       259: astore        13
       261: aload         6
       263: aload_1
       264: new           #70                 // class jdk/incubator/code/extern/ExternalizedOp
       267: dup
       268: ldc           #115                // String var.load
       270: ldc           #122                // String 13:20
       272: invokestatic  #76                 // Method jdk/incubator/code/Location.fromString:(Ljava/lang/String;)Ljdk/incubator/code/Location;
       275: aload         12
       277: invokestatic  #82                 // InterfaceMethod java/util/List.of:(Ljava/lang/Object;)Ljava/util/List;
       280: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       283: aload         4
       285: invokestatic  #119                // InterfaceMethod java/util/Map.of:()Ljava/util/Map;
       288: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       291: invokespecial #99                 // Method jdk/incubator/code/extern/ExternalizedOp."<init>":(Ljava/lang/String;Ljdk/incubator/code/Location;Ljava/util/List;Ljava/util/List;Ljdk/incubator/code/TypeElement;Ljava/util/Map;Ljava/util/List;)V
       294: invokeinterface #103,  2          // InterfaceMethod jdk/incubator/code/extern/OpFactory.constructOp:(Ljdk/incubator/code/extern/ExternalizedOp;)Ljdk/incubator/code/Op;
       299: invokevirtual #109                // Method jdk/incubator/code/Block$Builder.op:(Ljdk/incubator/code/Op;)Ljdk/incubator/code/Op$Result;
       302: astore        14
       304: aload         6
       306: aload_1
       307: new           #70                 // class jdk/incubator/code/extern/ExternalizedOp
       310: dup
       311: ldc           #124                // String add
       313: ldc           #117                // String 13:16
       315: invokestatic  #76                 // Method jdk/incubator/code/Location.fromString:(Ljava/lang/String;)Ljdk/incubator/code/Location;
       318: aload         13
       320: aload         14
       322: invokestatic  #126                // InterfaceMethod java/util/List.of:(Ljava/lang/Object;Ljava/lang/Object;)Ljava/util/List;
       325: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       328: aload         4
       330: invokestatic  #119                // InterfaceMethod java/util/Map.of:()Ljava/util/Map;
       333: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       336: invokespecial #99                 // Method jdk/incubator/code/extern/ExternalizedOp."<init>":(Ljava/lang/String;Ljdk/incubator/code/Location;Ljava/util/List;Ljava/util/List;Ljdk/incubator/code/TypeElement;Ljava/util/Map;Ljava/util/List;)V
       339: invokeinterface #103,  2          // InterfaceMethod jdk/incubator/code/extern/OpFactory.constructOp:(Ljdk/incubator/code/extern/ExternalizedOp;)Ljdk/incubator/code/Op;
       344: invokevirtual #109                // Method jdk/incubator/code/Block$Builder.op:(Ljdk/incubator/code/Op;)Ljdk/incubator/code/Op$Result;
       347: astore        15
       349: aload_2
       350: aload         7
       352: invokeinterface #30,  2           // InterfaceMethod jdk/incubator/code/extern/TypeElementFactory.constructType:(Ljdk/incubator/code/extern/ExternalizedTypeElement;)Ljdk/incubator/code/TypeElement;
       357: astore        16
       359: aload         6
       361: aload_1
       362: new           #70                 // class jdk/incubator/code/extern/ExternalizedOp
       365: dup
       366: ldc           #129                // String return
       368: ldc           #131                // String 13:9
       370: invokestatic  #76                 // Method jdk/incubator/code/Location.fromString:(Ljava/lang/String;)Ljdk/incubator/code/Location;
       373: aload         15
       375: invokestatic  #82                 // InterfaceMethod java/util/List.of:(Ljava/lang/Object;)Ljava/util/List;
       378: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       381: aload         16
       383: invokestatic  #119                // InterfaceMethod java/util/Map.of:()Ljava/util/Map;
       386: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       389: invokespecial #99                 // Method jdk/incubator/code/extern/ExternalizedOp."<init>":(Ljava/lang/String;Ljdk/incubator/code/Location;Ljava/util/List;Ljava/util/List;Ljdk/incubator/code/TypeElement;Ljava/util/Map;Ljava/util/List;)V
       392: invokeinterface #103,  2          // InterfaceMethod jdk/incubator/code/extern/OpFactory.constructOp:(Ljdk/incubator/code/extern/ExternalizedOp;)Ljdk/incubator/code/Op;
       397: invokevirtual #109                // Method jdk/incubator/code/Block$Builder.op:(Ljdk/incubator/code/Op;)Ljdk/incubator/code/Op$Result;
       400: pop
       401: aload_1
       402: new           #70                 // class jdk/incubator/code/extern/ExternalizedOp
       405: dup
       406: ldc           #133                // String func
       408: ldc           #135                // String 11:5:file:///Users/mabbay/babylon/test/jdk/java/lang/reflect/code/writer/TestOpBuilder2.java
       410: invokestatic  #76                 // Method jdk/incubator/code/Location.fromString:(Ljava/lang/String;)Ljdk/incubator/code/Location;
       413: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       416: invokestatic  #87                 // InterfaceMethod java/util/List.of:()Ljava/util/List;
       419: aload         16
       421: ldc           #90                 // String
       423: ldc           #137                // String g
       425: invokestatic  #94                 // InterfaceMethod java/util/Map.of:(Ljava/lang/Object;Ljava/lang/Object;)Ljava/util/Map;
       428: aload         5
       430: invokestatic  #82                 // InterfaceMethod java/util/List.of:(Ljava/lang/Object;)Ljava/util/List;
       433: invokespecial #99                 // Method jdk/incubator/code/extern/ExternalizedOp."<init>":(Ljava/lang/String;Ljdk/incubator/code/Location;Ljava/util/List;Ljava/util/List;Ljdk/incubator/code/TypeElement;Ljava/util/Map;Ljava/util/List;)V
       436: invokeinterface #103,  2          // InterfaceMethod jdk/incubator/code/extern/OpFactory.constructOp:(Ljdk/incubator/code/extern/ExternalizedOp;)Ljdk/incubator/code/Op;
       441: astore        17
       443: aload         17
       445: invokevirtual #139                // Method jdk/incubator/code/Op.seal:()V
       448: aload         17
       450: areturn
    * */
}
