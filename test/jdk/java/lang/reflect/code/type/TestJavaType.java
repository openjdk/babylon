/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import jdk.incubator.code.dialect.java.*;
import jdk.incubator.code.dialect.java.impl.JavaTypeUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;


/*
 * @test
 * @modules jdk.incubator.code/jdk.incubator.code.dialect.java.impl
 * @run junit TestJavaType
 */

public class TestJavaType {

    public static Object[][] JavaTypes() {
        return new Object[][]{
                {"boolean", "Z"},
                {"byte", "B"},
                {"char", "C"},
                {"short", "S"},
                {"int", "I"},
                {"long", "J"},
                {"float", "F"},
                {"double", "D"},
                {"void", "V"},
                {"int[]", "[I"},
                {"int[][][][]", "[[[[I"},
                {"java.lang.String", "Ljava/lang/String;"},
                {"java.lang.String[][]", "[[Ljava/lang/String;"},
                {"a.b.C$D", "La/b/C$D;"},
        };
    }

    @ParameterizedTest
    @MethodSource("JavaTypes")
    public void testJavaType(String tds, String bcd) {
        JavaType jt = typeFromFlatString(tds);
        Assertions.assertEquals(tds, jt.toString());
        Assertions.assertEquals(bcd, jt.toNominalDescriptor().descriptorString());
        Assertions.assertEquals(JavaType.type(ClassDesc.ofDescriptor(bcd)), jt);
    }

    public Object[][] classDescriptors() {
        return new Object[][]{
                {"java.lang.String", "java.lang.String"},
        };
    }

    @ParameterizedTest
    @MethodSource("classDescriptors")
    public void classDescriptor(String tds, String bcd) {
        ClassType jt = (ClassType) typeFromFlatString(tds);
        Assertions.assertEquals(tds, jt.toString());
        Assertions.assertEquals(bcd, jt.toClassName());
    }

    public static Object[][] basicJavaTypes() {
        return new Object[][]{
                {"boolean", "int"},
                {"byte", "int"},
                {"char", "int"},
                {"short", "int"},
                {"int", "int"},
                {"long", "long"},
                {"float", "float"},
                {"double", "double"},
                {"void", "void"},
                {"int[]", "java.lang.Object"},
                {"int[][][][]", "java.lang.Object"},
                {"java.lang.String", "java.lang.Object"},
                {"java.lang.String[][]", "java.lang.Object"},
                {"a.b.C$D", "java.lang.Object"},
                {"java.util.List<T>", "java.lang.Object"},
                {"java.util.List<T>[]", "java.lang.Object"},
        };
    }

    @ParameterizedTest
    @MethodSource("basicJavaTypes")
    public void testBasicJavaType(String tds, String btds) {
        JavaType jt = typeFromFlatString(tds);
        Assertions.assertEquals(tds, jt.toString());
        Assertions.assertEquals(btds, jt.toBasicType().toString());
    }


    public static Object[][] argumentJavaTypes() {
        return new Object[][]{
                {"java.util.List<T>", "T"},
                {"java.util.List<T>[]", "T"},
                {"java.util.List<java.util.function.Supplier<T>>", "java.util.function.Supplier<T>"},
                {"java.util.List<java.util.function.Supplier<T>>[][]", "java.util.function.Supplier<T>"},
                {"java.util.Map<K, V>", "K", "V"},
                {"ab<cd<S<T, V>, N>>", "cd<S<T, V>, N>"},
                {"java.util.Consumer<java.util.Function<String, Number>>", "java.util.Function<String, Number>"},
        };
    }

    @ParameterizedTest
    @MethodSource("argumentJavaTypes")
    public void testArgumentJavaType(String tds, String... argTypes) {
        JavaType jt = typeFromFlatString(tds);
        Assertions.assertEquals(tds, jt.toString());

        while (jt instanceof ArrayType) {
            jt = ((ArrayType)jt).componentType();
        }
        ClassType ct = (ClassType)jt;

        Assertions.assertEquals(ct.typeArguments().size(), argTypes.length);

        Assertions.assertEquals(Stream.of(argTypes).map(TestJavaType::typeFromFlatString).toList(), ct.typeArguments());
    }

    @ParameterizedTest
    @MethodSource("classDescs")
    public void testClassDescRoundTrip(ClassDesc classDesc) {
        Assertions.assertEquals(JavaType.type(classDesc).toNominalDescriptor(), classDesc);
    }

    public static Object[][] classDescs() throws ReflectiveOperationException {
        List<Object[]> classDescs = new ArrayList<>();
        for (Field f : ConstantDescs.class.getDeclaredFields()) {
            if (f.getName().startsWith("CD_")) {
                ClassDesc cd = (ClassDesc)f.get(null);
                classDescs.add(new Object[] { cd });
                if (!cd.equals(ConstantDescs.CD_void)) {
                    classDescs.add(new Object[]{cd.arrayType()});
                    classDescs.add(new Object[]{cd.arrayType().arrayType()});
                }
            }
        }
        return classDescs.stream().toArray(Object[][]::new);
    }

    @ParameterizedTest
    @MethodSource("types")
    public void testTypeRoundTrip(Type type) throws ReflectiveOperationException {
        JavaType javaType = JavaType.type(type);
        Assertions.assertEquals(javaType.resolve(MethodHandles.lookup()), type);
        Assertions.assertEquals(JavaType.JAVA_ONLY_TYPE_FACTORY.constructType(javaType.externalize()), javaType);
    }

    @ParameterizedTest
    @MethodSource("types")
    public void testTypeString(Type type) throws ReflectiveOperationException {
        JavaType javaType = JavaType.type(type);
        Assertions.assertEquals(replaceTypeVariables(javaType).toString()
                .replaceAll("::", "\\$"), type.getTypeName()
        );
    }

    JavaType replaceTypeVariables(JavaType type) {
        // This type transformation replaces type variables with simple class types.
        // This obtains a JavaType whose toString behaves the same as Type::getTypeName
        return switch (type) {
            case PrimitiveType p -> p;
            case WildcardType w -> JavaType.wildcard(w.boundKind(), replaceTypeVariables(w.boundType()));
            case ArrayType a -> JavaType.array(replaceTypeVariables(a.componentType()));
            case ClassType c -> {
                ClassType res = c.rawType();
                if (c.enclosingType().isPresent()) {
                    JavaType encl = replaceTypeVariables(c.enclosingType().get());
                    String nestedName = c.toClassName().substring(encl.toNominalDescriptor().displayName().length() + 1);
                    res = JavaType.qualified(replaceTypeVariables(c.enclosingType().get()), nestedName);
                }
                if (c.hasTypeArguments()) {
                    res = JavaType.parameterized(res,
                            c.typeArguments().stream().map(this::replaceTypeVariables).toList());
                }
                yield res;
            }
            case TypeVariableType t -> JavaType.type(ClassDesc.of(t.name()));
        };
    }

    public static Object[][] types() throws ReflectiveOperationException {
        List<Object[]> types = new ArrayList<>();
        for (Field f : TypeHolder.class.getDeclaredFields()) {
            types.add(new Object[] { f.getGenericType() });
        }
        return types.stream().toArray(Object[][]::new);
    }

    static class TypeHolder<X extends Number> {
        boolean p1;
        char p2;
        byte p3;
        short p4;
        int p5;
        long p6;
        float p7;
        double p8;

        boolean[] ap1;
        char[] ap2;
        byte[] ap3;
        short[] ap4;
        int[] ap5;
        long[] ap6;
        float[] ap7;
        double[] ap8;

        boolean[][] aap1;
        char[][] aap2;
        byte[][] aap3;
        short[][] aap4;
        int[][] aap5;
        long[][] aap6;
        float[][] aap7;
        double[][] aap8;

        String r1;
        Map<String, String> r2;
        Map<String, ?  extends String> r3;
        Map<? extends String, String> r4;
        Map<? extends String, ?  extends String> r5;
        Map<? extends List<? extends String>, ? super List<? extends String>> r6;
        Map<? extends List<? extends String>[], ? super List<? extends String>[]> r7;
        List<boolean[]> r8;
        List<char[]> r9;
        List<byte[]> r10;
        List<short[]> r11;
        List<int[]> r12;
        List<long[]> r13;
        List<float[]> r14;
        List<double[]> r15;

        String[] ar1;
        Map<String, String>[] ar2;
        Map<String, ?  extends String>[] ar3;
        Map<? extends String, String>[] ar4;
        Map<? extends String, ?  extends String>[] ar5;
        Map<? extends List<? extends String>, ? super List<? extends String>>[] ar6;
        Map<? extends List<? extends String>[], ? super List<? extends String>[]>[] ar7;
        List<boolean[]>[] ar8;
        List<char[]>[] ar9;
        List<byte[]>[] ar10;
        List<short[]>[] ar11;
        List<int[]>[] ar12;
        List<long[]>[] ar13;
        List<float[]>[] ar14;
        List<double[]>[] ar15;

        String[][] aar1;
        Map<String, String>[][] aar2;
        Map<String, ?  extends String>[][] aar3;
        Map<? extends String, String>[][] aar4;
        Map<? extends String, ?  extends String>[][] aar5;
        Map<? extends List<? extends String>, ? super List<? extends String>>[][] aar6;
        Map<? extends List<? extends String>[], ? super List<? extends String>[]>[][] aar7;
        List<boolean[]>[][] aar8;
        List<char[]>[][] aar9;
        List<byte[]>[][] aar10;
        List<short[]>[][] aar11;
        List<int[]>[][] aar12;
        List<long[]>[][] aar13;
        List<float[]>[][] aar14;
        List<double[]>[][] aar15;

        X x1;
        Map<X, X> x2;
        Map<X, ?  extends X> x3;
        Map<? extends X, X> x4;
        Map<? extends X, ?  extends X> x5;
        Map<? extends List<? extends X>, ? super List<? extends X>> x6;
        Map<? extends List<? extends X>[], ? super List<? extends X>[]> x7;
        List<X[]> x8;

        X[] ax1;
        Map<X, X>[] ax2;
        Map<X, ?  extends X>[] ax3;
        Map<? extends X, X>[] ax4;
        Map<? extends X, ?  extends X>[] ax5;
        Map<? extends List<? extends X>, ? super List<? extends X>>[] ax6;
        Map<? extends List<? extends X>[], ? super List<? extends X>[]>[] ax7;
        List<X[]>[] ax8;

        X[][] aax1;
        Map<X, X>[][] aax2;
        Map<X, ?  extends X>[][] aax3;
        Map<? extends X, X>[][] aax4;
        Map<? extends X, ?  extends X>[][] aax5;
        Map<? extends List<? extends X>, ? super List<? extends X>>[][] aax6;
        Map<? extends List<? extends X>[], ? super List<? extends X>[]>[][] aax7;
        List<X[]>[][] aax8;

        static class Outer<X> {
            class Inner<X> { }
        }

        Outer<String>.Inner<String> o1;
        Outer<? extends String>.Inner<String> o2;
        Outer<String>.Inner<? extends String> o3;
        Outer<? extends String>.Inner<? extends String> o4;
        Outer<? super String>.Inner<String> o5;
        Outer<String>.Inner<? super String> o6;
        Outer<? super String>.Inner<? super String> o7;
        Outer<?>.Inner<String> o8;
        Outer<String>.Inner<?> o9;
        Outer<?>.Inner<?> o10;

        Outer<int[]>.Inner<int[]> oa1;
        Outer<? extends int[]>.Inner<int[]> oa2;
        Outer<int[]>.Inner<? extends int[]> oa3;
        Outer<? extends int[]>.Inner<? extends int[]> oa4;
        Outer<? super int[]>.Inner<int[]> oa5;
        Outer<int[]>.Inner<? super int[]> oa6;
        Outer<? super int[]>.Inner<? super int[]> oa7;
        Outer<?>.Inner<int[]> oa8;
        Outer<int[]>.Inner<?> oa9;
        Outer<?>.Inner<?> oa10;

        Outer<String>.Inner<String>[] ao1;
        Outer<? extends String>.Inner<String>[] ao2;
        Outer<String>.Inner<? extends String>[] ao3;
        Outer<? extends String>.Inner<? extends String>[] ao4;
        Outer<? super String>.Inner<String>[] ao5;
        Outer<String>.Inner<? super String>[] ao6;
        Outer<? super String>.Inner<? super String>[] ao7;
        Outer<?>.Inner<String>[] ao8;
        Outer<String>.Inner<?>[] ao9;
        Outer<?>.Inner<?>[] ao10;

        Outer<int[]>.Inner<int[]>[] aoa1;
        Outer<? extends int[]>.Inner<int[]>[] aoa2;
        Outer<int[]>.Inner<? extends int[]>[] aoa3;
        Outer<? extends int[]>.Inner<? extends int[]>[] aoa4;
        Outer<? super int[]>.Inner<int[]>[] aoa5;
        Outer<int[]>.Inner<? super int[]>[] aoa6;
        Outer<? super int[]>.Inner<? super int[]>[] aoa7;
        Outer<?>.Inner<int[]>[] aoa8;
        Outer<int[]>.Inner<?>[] aoa9;
        Outer<?>.Inner<?>[] aoa10;

        Outer<String>.Inner<String>[][] aao1;
        Outer<? extends String>.Inner<String>[][] aao2;
        Outer<String>.Inner<? extends String>[][] aao3;
        Outer<? extends String>.Inner<? extends String>[][] aao4;
        Outer<? super String>.Inner<String>[][] aao5;
        Outer<String>.Inner<? super String>[][] aao6;
        Outer<? super String>.Inner<? super String>[][] aao7;
        Outer<?>.Inner<String>[][] aao8;
        Outer<String>.Inner<?>[][] aao9;
        Outer<?>.Inner<?>[][] aao10;

        Outer<int[]>.Inner<int[]>[][] aaoa1;
        Outer<? extends int[]>.Inner<int[]>[][] aaoa2;
        Outer<int[]>.Inner<? extends int[]>[][] aaoa3;
        Outer<? extends int[]>.Inner<? extends int[]>[][] aaoa4;
        Outer<? super int[]>.Inner<int[]>[][] aaoa5;
        Outer<int[]>.Inner<? super int[]>[][] aaoa6;
        Outer<? super int[]>.Inner<? super int[]>[][] aaoa7;
        Outer<?>.Inner<int[]>[][] aaoa8;
        Outer<int[]>.Inner<?>[][] aaoa9;
        Outer<?>.Inner<?>[][] aaoa10;
    }

    private static JavaType typeFromFlatString(String desc) {
        return JavaTypeUtils.toJavaType(JavaTypeUtils.parseExternalTypeString(desc));
    }

    static class InnerTypes {

        class Member {
            class One {
                class Two {
                    class Three { }
                }
            }
        }

        static class Nested { }

        void m() {
            class Local_I_M { }
        }

        static void s_m() {
            class Local_S_M { }
        }

        InnerTypes() {
            class Local_C { }
        }
    }

    @Test
    public void testInnerTypes() throws ReflectiveOperationException {
        var innertypes = JavaType.type(InnerTypes.class);
        var member = (ClassType)JavaType.type(InnerTypes.Member.class);
        Assertions.assertEquals(innertypes, member.enclosingType().get());

        var memberOne = (ClassType)JavaType.type(InnerTypes.Member.One.class);
        Assertions.assertEquals(member, memberOne.enclosingType().get());
        Assertions.assertEquals(InnerTypes.Member.One.class.getName(), memberOne.toClassName());

        var memberTwo = (ClassType)JavaType.type(InnerTypes.Member.One.Two.class);
        Assertions.assertEquals(memberOne, memberTwo.enclosingType().get());
        Assertions.assertEquals(InnerTypes.Member.One.Two.class.getName(), memberTwo.toClassName());

        var memberThree = (ClassType)JavaType.type(InnerTypes.Member.One.Two.Three.class);
        Assertions.assertEquals(memberTwo, memberThree.enclosingType().get());
        Assertions.assertEquals(InnerTypes.Member.One.Two.Three.class.getName(), memberThree.toClassName());

        var nested = (ClassType)JavaType.type(InnerTypes.Nested.class);
        Assertions.assertTrue(nested.enclosingType().isEmpty());

        var local_s_m = (ClassType)JavaType.type(Class.forName("TestJavaType$InnerTypes$1Local_S_M"));
        Assertions.assertTrue(local_s_m.enclosingType().isEmpty());

        var local_i_m = (ClassType)JavaType.type(Class.forName("TestJavaType$InnerTypes$1Local_I_M"));
        Assertions.assertEquals(innertypes, local_i_m.enclosingType().get());

        var local_c = (ClassType)JavaType.type(Class.forName("TestJavaType$InnerTypes$1Local_C"));
        Assertions.assertEquals(innertypes, local_c.enclosingType().get());
    }
}
