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

import java.util.ArrayList;
import java.util.List;
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
                {"a::b(void):void", "a", "b"},
                {"a.b::c(int):int", "a.b", "c"},
                {"a.b.c::d(int, int):int", "a.b.c", "d"},
                {"a::b(Func<String, Number>, Entry<List<String>, val>, int, long):void", "a", "b"},
                {"java.io.PrintStream::println(java.lang.String):void", "java.io.PrintStream", "println"},
                {"MethodReferenceTest$A::m(java.lang.Object):java.lang.Object", "MethodReferenceTest$A", "m"},
                {"R<R::<T extends java.lang.Number>>::n(void):R::<T extends java.lang.Number>", "R<R::<T extends java.lang.Number>>", "n"}
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
    public Object[][] constructorRefs() {
        return new Object[][]{
                {"MethodReferenceTest$X::(int)", "MethodReferenceTest$X"},
                {"[MethodReferenceTest$A]::(int)", "[MethodReferenceTest$A]"},
        };
    }

    @Test(dataProvider = "constructorRefs")
    public void testConstructorRef(String cds, String refType) {
        ConstructorRef cr = ConstructorRef.ofString(cds);
        Assert.assertEquals(cr.toString(), cds);
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
        FieldRef fr = FieldRef.ofString(fds);
        Assert.assertEquals(fr.toString(), fds);
        Assert.assertEquals(fr.refType().toString(), refType);
        Assert.assertEquals(fr.name(), name);
        Assert.assertEquals(fr.type().toString(), type);
    }

    @DataProvider
    public Object[][] recordTypeRefs() {
        return new Object[][]{
                {"A()"},
                {"A(b : B)"},
                {"A(b : B, c : C)"},
                {"p.A(f : p.Func<String, Number>, e : Entry<List<String>, val>, i : int, l : long)"},
                {"R(n : R::<T extends java.lang.Number>)"}
        };
    }

    @Test(dataProvider = "recordTypeRefs")
    public void testRecordTypeRef(String rtds) {
        RecordTypeRef rtr = RecordTypeRef.ofString(rtds);
        Assert.assertEquals(rtr.toString(), rtds);
    }
}
