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

import java.lang.reflect.code.descriptor.*;
import java.lang.reflect.code.type.TypeDefinition;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestDescriptors
 */

public class TestDescriptors {

    @DataProvider
    public Object[][] TypeDescs() {
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

    @Test(dataProvider = "TypeDescs")
    public void testTypeDesc(String tds, String bcd) {
        TypeDefinition td = TypeDefinition.ofString(tds);
        Assert.assertEquals(td.toString(), tds);
    }

    @DataProvider
    public Object[][] classDescriptors() {
        return new Object[][]{
                {"java.lang.String", "java.lang.String"},
        };
    }

    @Test(dataProvider = "classDescriptors")
    public void classDescriptor(String tds, String bcd) {
        TypeDefinition td = TypeDefinition.ofString(tds);
        Assert.assertEquals(td.toString(), tds);
    }

    @DataProvider
    public Object[][] paramTypeDescs() {
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

    @Test(dataProvider = "paramTypeDescs")
    public void testParamTypeDesc(String tds, String... paramTypes) {
        TypeDefinition td = TypeDefinition.ofString(tds);
        Assert.assertEquals(td.toString(), tds);

        Assert.assertTrue(td.hasTypeArguments());
        Assert.assertEquals(paramTypes.length, td.typeArguments().size());

        Assert.assertEquals(td.typeArguments(), Stream.of(paramTypes).map(TypeDefinition::ofString).toList());
    }


    @DataProvider
    public Object[][] methodTypeDescriptors() {
        return new Object[][]{
                {"()boolean", "()Z"},
                {"()void", "()V"},
                {"(int)int", "(I)I"},
                {"(int, int)int", "(II)I"},
                {"(java.lang.String)int", "(Ljava/lang/String;)I"},
                {"(java.lang.String, int)int", "(Ljava/lang/String;I)I"},
                {"(int, java.lang.String)int", "(ILjava/lang/String;)I"},
                {"(int, java.lang.String)java.lang.String", "(ILjava/lang/String;)Ljava/lang/String;"},
                {"(byte, short, int, long)boolean", "(BSIJ)Z"},
                {"(Func<String, Number>, Entry<List<String>, val>, int, long)void", "(LFunc;LEntry;IJ)V"},
        };
    }

    @Test(dataProvider = "methodTypeDescriptors")
    public void testMethodTypeDescriptor(String mtds, String bcd) {
        MethodTypeDesc mtd = MethodTypeDesc.ofString(mtds);
        Assert.assertEquals(mtd.toString(), mtds);
        Assert.assertEquals(mtd.toNominalDescriptorString(), bcd);
        Assert.assertEquals(mtd.erase(), MethodTypeDesc.ofNominalDescriptorString(bcd));
    }

    @DataProvider
    public Object[][] methodDescriptors() {
        return new Object[][]{
                {"a::b()void", "a", "b"},
                {"a.b::c(int)int", "a.b", "c"},
                {"a.b.c::d(int, int)int", "a.b.c", "d"},
                {"java.io.PrintStream::println(java.lang.String)void", "java.io.PrintStream", "println"},
                {"MethodReferenceTest$A::m(java.lang.Object)java.lang.Object", "MethodReferenceTest$A", "m"},
                {"MethodReferenceTest$X::<new>(int)MethodReferenceTest$X", "MethodReferenceTest$X", "<new>"},
                {"MethodReferenceTest$A[]::<new>(int)MethodReferenceTest$A[]", "MethodReferenceTest$A[]", "<new>"}
        };
    }

    @Test(dataProvider = "methodDescriptors")
    public void testMethodDescriptor(String mds, String refType, String name) {
        MethodDesc md = MethodDesc.ofString(mds);
        Assert.assertEquals(md.toString(), mds);
        Assert.assertEquals(md.refType().toString(), refType);
        Assert.assertEquals(md.name(), name);
    }


    @DataProvider
    public Object[][] fieldDescriptors() {
        return new Object[][]{
                {"a.b::c()int", "a.b", "c", "int"},
                {"a.b.c::d()int", "a.b.c", "d", "int"},
                {"java.lang.System::out()java.io.PrintStream", "java.lang.System", "out", "java.io.PrintStream"},
        };
    }

    @Test(dataProvider = "fieldDescriptors")
    public void testFieldDescriptor(String fds, String refType, String name, String type) {
        FieldDesc fd = FieldDesc.ofString(fds);
        Assert.assertEquals(fd.toString(), fds);
        Assert.assertEquals(fd.refType().toString(), refType);
        Assert.assertEquals(fd.name(), name);
        Assert.assertEquals(fd.type().toString(), type);
    }


    @DataProvider
    public Object[][] recordTypeDescriptors() {
        return new Object[][]{
                {"()A"},
                {"(B b)A"},
                {"(B b, C c)A"},
                {"(p.Func<String, Number> f, Entry<List<String>, val> e, int i, long l)p.A<R>"},
        };
    }

    @Test(dataProvider = "recordTypeDescriptors")
    public void testRecordTypeDescriptor(String rtds) {
        RecordTypeDesc mtd = RecordTypeDesc.ofString(rtds);
        Assert.assertEquals(mtd.toString(), rtds);
    }

}
