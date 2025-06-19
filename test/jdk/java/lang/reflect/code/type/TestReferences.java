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

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.*;
import jdk.incubator.code.dialect.java.impl.JavaTypeUtils;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/*
 * @test
 * @modules jdk.incubator.code/jdk.incubator.code.dialect.java.impl
 * @run testng TestReferences
 */

public class TestReferences {

    interface X {
        Object value();
    }

    interface Y extends X {
        Boolean value();
    }

    @DataProvider
    public Object[][] methodRefs() {
        return new Object[][]{
                {"a::b():void", "a", "b"},
                {"a.b::c(int):int", "a.b", "c"},
                {"a.b.c::d(int, int):int", "a.b.c", "d"},
                {"a::b(Func<String, Number>, Entry<List<String>, val>, int, long):void", "a", "b"},
                {"java.io.PrintStream::println(java.lang.String):void", "java.io.PrintStream", "println"},
                {"MethodReferenceTest$A::m(java.lang.Object):java.lang.Object", "MethodReferenceTest$A", "m"},
                {"R<R::<T extends java.lang.Number>>::n():R::<T extends java.lang.Number>", "R<R::<T extends java.lang.Number>>", "n"}
        };
    }

    @Test(dataProvider = "methodRefs")
    public void testMethodRef(String mds, String refType, String name) {
        MethodRef mr = refFromFlatString(mds);
        Assert.assertEquals(mr.toString(), mds);
        Assert.assertEquals(mr.refType().toString(), refType);
        Assert.assertEquals(mr.name(), name);
    }


    @DataProvider
    public Object[][] externalizedMethodRefs() {
        return new Object[][]{
                {"java.ref:\"a::b():void\"", "a", "b"},
                {"java.ref:\"a.b::c(int):int\"", "a.b", "c"},
                {"java.ref:\"a.b.c::d(int, int):int\"", "a.b.c", "d"},
                {"java.ref:\"a::b(Func<String, Number>, Entry<List<String>, val>, int, long):void\"", "a", "b"},
                {"java.ref:\"java.io.PrintStream::println(java.lang.String):void\"", "java.io.PrintStream", "println"},
                {"java.ref:\"MethodReferenceTest$A::m(java.lang.Object):java.lang.Object\"", "MethodReferenceTest$A", "m"},
                {"java.ref:\"R<R::<T extends java.lang.Number>>::n():R::<T extends java.lang.Number>\"", "R<R::<T extends java.lang.Number>>", "n"}
        };
    }

    @Test(dataProvider = "externalizedMethodRefs")
    public void testExternalizedMethodRef(String mds, String refType, String name) {
        TypeElement.ExternalizedTypeElement emr = TypeElement.ExternalizedTypeElement.ofString(mds);
        MethodRef mr = (MethodRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(emr));
        Assert.assertEquals(JavaTypeUtils.flatten(mr.externalize()).toString(), mds);
        Assert.assertEquals(mr.refType().toString(), refType);
        Assert.assertEquals(mr.name(), name);
    }


    @DataProvider
    public Object[][] constructorRefs() {
        return new Object[][]{
                {"MethodReferenceTest$X::(int)", "MethodReferenceTest$X"},
                {"MethodReferenceTest$A[]::(int)", "MethodReferenceTest$A[]"},
        };
    }

    @Test(dataProvider = "constructorRefs")
    public void testConstructorRef(String cds, String refType) {
        ConstructorRef cr = refFromFlatString(cds);
        Assert.assertEquals(cr.toString(), cds);
        Assert.assertEquals(cr.refType().toString(), refType);
    }

    @DataProvider
    public Object[][] externalizedConstructorRefs() {
        return new Object[][]{
                {"java.ref:\"MethodReferenceTest$X::(int)\"", "MethodReferenceTest$X"},
                {"java.ref:\"MethodReferenceTest$A[]::(int)\"", "MethodReferenceTest$A[]"},
        };
    }

    @Test(dataProvider = "externalizedConstructorRefs")
    public void testExternalizedConstructorRef(String crs, String refType) {
        TypeElement.ExternalizedTypeElement ecr = TypeElement.ExternalizedTypeElement.ofString(crs);
        ConstructorRef cr = (ConstructorRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(ecr));

        Assert.assertEquals(JavaTypeUtils.flatten(cr.externalize()).toString(), crs);
        Assert.assertEquals(cr.refType().toString(), refType);
    }


    @DataProvider
    public Object[][] fieldRefs() {
        return new Object[][]{
                {"a.b::c:int", "a.b", "c", "int"},
                {"a.b.c::d:int", "a.b.c", "d", "int"},
                {"java.lang.System::out:java.io.PrintStream", "java.lang.System", "out", "java.io.PrintStream"},
                {"R<R::<T extends java.lang.Number>>::n:R::<T extends java.lang.Number>", "R<R::<T extends java.lang.Number>>", "n", "R::<T extends java.lang.Number>"}
        };
    }

    @Test(dataProvider = "fieldRefs")
    public void testFieldRef(String fds, String refType, String name, String type) {
        FieldRef fr = refFromFlatString(fds);
        Assert.assertEquals(fr.toString(), fds);
        Assert.assertEquals(fr.refType().toString(), refType);
        Assert.assertEquals(fr.name(), name);
        Assert.assertEquals(fr.type().toString(), type);
    }

    @DataProvider
    public Object[][] externalizedFieldRefs() {
        return new Object[][]{
                {"java.ref:\"a.b::c:int\"", "a.b", "c", "int"},
                {"java.ref:\"a.b.c::d:int\"", "a.b.c", "d", "int"},
                {"java.ref:\"java.lang.System::out:java.io.PrintStream\"", "java.lang.System", "out", "java.io.PrintStream"},
                {"java.ref:\"R<R::<T extends java.lang.Number>>::n:R::<T extends java.lang.Number>\"", "R<R::<T extends java.lang.Number>>", "n", "R::<T extends java.lang.Number>"}
        };
    }

    @Test(dataProvider = "externalizedFieldRefs")
    public void testExternalizedFieldRef(String frs, String refType, String name, String type) {
        TypeElement.ExternalizedTypeElement efr = TypeElement.ExternalizedTypeElement.ofString(frs);
        FieldRef fr = (FieldRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(efr));

        Assert.assertEquals(JavaTypeUtils.flatten(fr.externalize()).toString(), frs);
        Assert.assertEquals(fr.refType().toString(), refType);
        Assert.assertEquals(fr.name(), name);
        Assert.assertEquals(fr.type().toString(), type);
    }


    @DataProvider
    public Object[][] recordTypeRefs() {
        return new Object[][]{
                {"()A"},
                {"(B b)A"},
                {"(B b, C c)A"},
                {"(p.Func<String, Number> f, Entry<List<String>, val> e, int i, long l)p.A<R>"},
                {"(R::<T extends java.lang.Number> n)R<R::<T extends java.lang.Number>>"}
        };
    }

    @Test(dataProvider = "recordTypeRefs")
    public void testRecordTypeRef(String rtds) {
        RecordTypeRef rtr = refFromFlatString(rtds);
        Assert.assertEquals(rtr.toString(), rtds);
    }

    @DataProvider
    public Object[][] externalizedRecordTypeRefs() {
        return new Object[][]{
                {"java.ref:\"()A\""},
                {"java.ref:\"(B b)A\""},
                {"java.ref:\"(B b, C c)A\""},
                {"java.ref:\"(p.Func<String, Number> f, Entry<List<String>, val> e, int i, long l)p.A<R>\""},
                {"java.ref:\"(R::<T extends java.lang.Number> n)R<R::<T extends java.lang.Number>>\""}
        };
    }

    @Test(dataProvider = "externalizedRecordTypeRefs")
    public void testExternalizedRecordTypeRef(String rtds) {
        TypeElement.ExternalizedTypeElement ertr = TypeElement.ExternalizedTypeElement.ofString(rtds);
        RecordTypeRef rtr = (RecordTypeRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(ertr));
        Assert.assertEquals(JavaTypeUtils.flatten(rtr.externalize()).toString(), rtds);
    }

    @SuppressWarnings("unchecked")
    private static <R extends JavaRef> R refFromFlatString(String desc) {
        return (R)JavaTypeUtils.toJavaRef(JavaTypeUtils.parseExternalRefString(desc));
    }
}
