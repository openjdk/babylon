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

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.type.JavaType;
import java.util.stream.Stream;


/*
 * @test
 * @run testng TestJavaType
 */

public class TestJavaType {

    @DataProvider
    public Object[][] JavaTypes() {
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

    @Test(dataProvider = "JavaTypes")
    public void testJavaType(String tds, String bcd) {
        JavaType jt = JavaType.ofString(tds);
        Assert.assertEquals(jt.toString(), tds);
        Assert.assertEquals(jt.toNominalDescriptorString(), bcd);
        Assert.assertEquals(jt, JavaType.ofNominalDescriptorString(bcd));
    }

    @DataProvider
    public Object[][] classDescriptors() {
        return new Object[][]{
                {"java.lang.String", "java.lang.String"},
        };
    }

    @Test(dataProvider = "classDescriptors")
    public void classDescriptor(String tds, String bcd) {
        JavaType jt = JavaType.ofString(tds);
        Assert.assertEquals(jt.toString(), tds);
        Assert.assertEquals(jt.toClassName(), bcd);
    }


    @DataProvider
    public Object[][] basicJavaTypes() {
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

    @Test(dataProvider = "basicJavaTypes")
    public void testBasicJavaType(String tds, String btds) {
        JavaType jt = JavaType.ofString(tds);
        Assert.assertEquals(jt.toString(), tds);
        Assert.assertEquals(jt.toBasicType().toString(), btds);
    }


    @DataProvider
    public Object[][] argumentJavaTypes() {
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

    @Test(dataProvider = "argumentJavaTypes")
    public void testArgumentJavaType(String tds, String... argTypes) {
        JavaType jt = JavaType.ofString(tds);
        Assert.assertEquals(jt.toString(), tds);

        Assert.assertTrue(jt.hasTypeArguments());
        Assert.assertEquals(argTypes.length, jt.typeArguments().size());

        Assert.assertEquals(jt.typeArguments(), Stream.of(argTypes).map(JavaType::ofString).toList());
    }
}
