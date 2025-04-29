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
import jdk.incubator.code.type.*;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestReferences
 */

public class TestReferences {

    @DataProvider
    public Object[][] methodRefs() {
        return new Object[][]{
                {"a::b()void", "a", "b"},
                {"a.b::c(int)int", "a.b", "c"},
                {"a.b.c::d(int, int)int", "a.b.c", "d"},
                {"a::b(Func<String, Number>, Entry<List<String>, val>, int, long)void", "a", "b"},
                {"java.io.PrintStream::println(java.lang.String)void", "java.io.PrintStream", "println"},
                {"MethodReferenceTest$A::m(java.lang.Object)java.lang.Object", "MethodReferenceTest$A", "m"},
                {"R<#T<R, java.lang.Number>>::n()#T<R, java.lang.Number>", "R<#T<R, java.lang.Number>>", "n"}
        };
    }

    @Test(dataProvider = "methodRefs")
    public void testMethodRef(String mds, String refType, String name) {
        MethodRef mr = MethodRef.ofString(mds);
        Assert.assertEquals(mr.toString(), mds);
        Assert.assertEquals(mr.refType().externalize().toString(), refType);
        Assert.assertEquals(mr.name(), name);
    }


    @DataProvider
    public Object[][] externalizedMethodRefs() {
        return new Object[][]{
                {"&m<a, b, func<void>>", "a", "b"},
                {"&m<a.b, c, func<int, int>>", "a.b", "c"},
                {"&m<a.b.c, d, func<int, int, int>>", "a.b.c", "d"},
                {"&m<a, b, func<void, Func<String, Number>, Entry<List<String>, val>, int, long>>", "a", "b"},
                {"&m<java.io.PrintStream, println, func<void, java.lang.String>>", "java.io.PrintStream", "println"},
                {"&m<MethodReferenceTest$A, m, func<java.lang.Object, java.lang.Object>>", "MethodReferenceTest$A", "m"},
                {"&m<R<#T<R, java.lang.Number>>, n, func<#T<R, java.lang.Number>>>", "R<#T<R, java.lang.Number>>", "n"}
        };
    }

    @Test(dataProvider = "externalizedMethodRefs")
    public void testExternalizedMethodRef(String mds, String refType, String name) {
        TypeElement.ExternalizedTypeElement emr = TypeElement.ExternalizedTypeElement.ofString(mds);
        MethodRef mr = (MethodRef) CoreTypeFactory.CORE_TYPE_FACTORY.constructType(emr);
        Assert.assertEquals(mr.externalize().toString(), mds);
        Assert.assertEquals(mr.refType().externalize().toString(), refType);
        Assert.assertEquals(mr.name(), name);
    }


    @DataProvider
    public Object[][] constructorRefs() {
        return new Object[][]{
                {"MethodReferenceTest$X::<new>(int)", "MethodReferenceTest$X"},
                {"MethodReferenceTest$A[]::<new>(int)", "MethodReferenceTest$A[]"},
        };
    }

    @Test(dataProvider = "constructorRefs")
    public void testConstructorRef(String cds, String refType) {
        ConstructorRef cr = ConstructorRef.ofString(cds);
        Assert.assertEquals(cr.toString(), cds);
        Assert.assertEquals(cr.refType().externalize().toString(), refType);
    }

    @DataProvider
    public Object[][] externalizedConstructorRefs() {
        return new Object[][]{
                {"&c<func<MethodReferenceTest$X, int>>", "MethodReferenceTest$X"},
                {"&c<func<MethodReferenceTest$A[], int>>", "MethodReferenceTest$A[]"},
        };
    }

    @Test(dataProvider = "externalizedConstructorRefs")
    public void testExternalizedConstructorRef(String crs, String refType) {
        TypeElement.ExternalizedTypeElement ecr = TypeElement.ExternalizedTypeElement.ofString(crs);
        ConstructorRef cr = (ConstructorRef) CoreTypeFactory.CORE_TYPE_FACTORY.constructType(ecr);

        Assert.assertEquals(cr.externalize().toString(), crs);
        Assert.assertEquals(cr.refType().externalize().toString(), refType);
    }


    @DataProvider
    public Object[][] fieldRefs() {
        return new Object[][]{
                {"a.b::c()int", "a.b", "c", "int"},
                {"a.b.c::d()int", "a.b.c", "d", "int"},
                {"java.lang.System::out()java.io.PrintStream", "java.lang.System", "out", "java.io.PrintStream"},
                {"R<#T<R, java.lang.Number>>::n()#T<R, java.lang.Number>", "R<#T<R, java.lang.Number>>", "n", "#T<R, java.lang.Number>"}
        };
    }

    @Test(dataProvider = "fieldRefs")
    public void testFieldRef(String fds, String refType, String name, String type) {
        FieldRef fr = FieldRef.ofString(fds);
        Assert.assertEquals(fr.toString(), fds);
        Assert.assertEquals(fr.refType().externalize().toString(), refType);
        Assert.assertEquals(fr.name(), name);
        Assert.assertEquals(fr.type().externalize().toString(), type);
    }

    @DataProvider
    public Object[][] externalizedFieldRefs() {
        return new Object[][]{
                {"&f<a.b, c, int>", "a.b", "c", "int"},
                {"&f<a.b.c, d, int>", "a.b.c", "d", "int"},
                {"&f<java.lang.System, out, java.io.PrintStream>", "java.lang.System", "out", "java.io.PrintStream"},
                {"&f<R<#T<R, java.lang.Number>>, n, #T<R, java.lang.Number>>", "R<#T<R, java.lang.Number>>", "n", "#T<R, java.lang.Number>"}
        };
    }

    @Test(dataProvider = "externalizedFieldRefs")
    public void testExternalizedFieldRef(String frs, String refType, String name, String type) {
        TypeElement.ExternalizedTypeElement efr = TypeElement.ExternalizedTypeElement.ofString(frs);
        FieldRef fr = (FieldRef) CoreTypeFactory.CORE_TYPE_FACTORY.constructType(efr);

        Assert.assertEquals(fr.externalize().toString(), frs);
        Assert.assertEquals(fr.refType().externalize().toString(), refType);
        Assert.assertEquals(fr.name(), name);
        Assert.assertEquals(fr.type().externalize().toString(), type);
    }


    @DataProvider
    public Object[][] recordTypeRefs() {
        return new Object[][]{
                {"()A"},
                {"(B b)A"},
                {"(B b, C c)A"},
                {"(p.Func<String, Number> f, Entry<List<String>, val> e, int i, long l)p.A<R>"},
                {"(#T<R, java.lang.Number> n)R<#T<R, java.lang.Number>>"}
        };
    }

    @Test(dataProvider = "recordTypeRefs")
    public void testRecordTypeRef(String rtds) {
        RecordTypeRef rtr = RecordTypeRef.ofString(rtds);
        Assert.assertEquals(rtr.toString(), rtds);
    }

    @DataProvider
    public Object[][] externalizedRecordTypeRefs() {
        return new Object[][]{
                {"&r<A>"},
                {"&r<A, B, b>"},
                {"&r<A, B, b, C, c>"},
                {"&r<p.A<R>, p.Func<String, Number>, f, Entry<List<String>, val>, e, int, i, long, l>"},
                // @@@ Fails because of externalize().toString()
                {"&r<R<#T<R, java.lang.Number>>, #T<R, java.lang.Number>, n>"}
        };
    }

    @Test(dataProvider = "externalizedRecordTypeRefs")
    public void testExternalizedRecordTypeRef(String rtds) {
        TypeElement.ExternalizedTypeElement ertr = TypeElement.ExternalizedTypeElement.ofString(rtds);
        RecordTypeRef rtr = (RecordTypeRef) CoreTypeFactory.CORE_TYPE_FACTORY.constructType(ertr);
        Assert.assertEquals(rtr.externalize().toString(), rtds);
    }
}
