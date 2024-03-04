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

import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.RecordTypeRef;

/*
 * @test
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
                {"MethodReferenceTest$X::<new>(int)MethodReferenceTest$X", "MethodReferenceTest$X", "<new>"},
                {"MethodReferenceTest$A[]::<new>(int)MethodReferenceTest$A[]", "MethodReferenceTest$A[]", "<new>"}
        };
    }

    @Test(dataProvider = "methodRefs")
    public void testMethodRef(String mds, String refType, String name) {
        MethodRef mr = MethodRef.ofString(mds);
        Assert.assertEquals(mr.toString(), mds);
        Assert.assertEquals(mr.refType().toString(), refType);
        Assert.assertEquals(mr.name(), name);
    }


    @DataProvider
    public Object[][] fieldRefs() {
        return new Object[][]{
                {"a.b::c()int", "a.b", "c", "int"},
                {"a.b.c::d()int", "a.b.c", "d", "int"},
                {"java.lang.System::out()java.io.PrintStream", "java.lang.System", "out", "java.io.PrintStream"},
        };
    }

    @Test(dataProvider = "fieldRefs")
    public void testFieldRef(String fds, String refType, String name, String type) {
        FieldRef fr = FieldRef.ofString(fds);
        Assert.assertEquals(fr.toString(), fds);
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
        };
    }

    @Test(dataProvider = "recordTypeRefs")
    public void testRecordTypeRef(String rtds) {
        RecordTypeRef rtr = RecordTypeRef.ofString(rtds);
        Assert.assertEquals(rtr.toString(), rtds);
    }

}
