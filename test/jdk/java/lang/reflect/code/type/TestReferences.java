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
import jdk.incubator.code.extern.ExternalizedTypeElement;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/*
 * @test
 * @modules jdk.incubator.code/jdk.incubator.code.dialect.java.impl
 * @run junit TestReferences
 */

public class TestReferences {

    interface X {
        Object value();
    }

    interface Y extends X {
        Boolean value();
    }

    public static Object[][] methodRefs() {
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

    @ParameterizedTest
    @MethodSource("methodRefs")
    public void testMethodRef(String mds, String refType, String name) {
        MethodRef mr = refFromFlatString(mds);
        Assertions.assertEquals(mds, mr.toString());
        Assertions.assertEquals(refType, mr.refType().toString());
        Assertions.assertEquals(name, mr.name());
    }


    public static Object[][] externalizedMethodRefs() {
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

    @ParameterizedTest
    @MethodSource("externalizedMethodRefs")
    public void testExternalizedMethodRef(String mds, String refType, String name) {
        ExternalizedTypeElement emr = ExternalizedTypeElement.ofString(mds);
        MethodRef mr = (MethodRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(emr));
        Assertions.assertEquals(mds, JavaTypeUtils.flatten(mr.externalize()).toString());
        Assertions.assertEquals(refType, mr.refType().toString());
        Assertions.assertEquals(name, mr.name());
    }


    public static Object[][] constructorRefs() {
        return new Object[][]{
                {"MethodReferenceTest$X::(int)", "MethodReferenceTest$X"},
                {"MethodReferenceTest$A[]::(int)", "MethodReferenceTest$A[]"},
        };
    }

    @ParameterizedTest
    @MethodSource("constructorRefs")
    public void testConstructorRef(String cds, String refType) {
        MethodRef cr = refFromFlatString(cds);
        Assertions.assertEquals(cds, cr.toString());
        Assertions.assertEquals(refType, cr.refType().toString());
    }

    public static Object[][] externalizedConstructorRefs() {
        return new Object[][]{
                {"java.ref:\"MethodReferenceTest$X::(int)\"", "MethodReferenceTest$X"},
                {"java.ref:\"MethodReferenceTest$A[]::(int)\"", "MethodReferenceTest$A[]"},
        };
    }

    @ParameterizedTest
    @MethodSource("externalizedMethodRefs")
    public void testExternalizedMethodRef(String crs, String refType) {
        ExternalizedTypeElement ecr = ExternalizedTypeElement.ofString(crs);
        MethodRef cr = (MethodRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(ecr));

        Assertions.assertEquals(crs, JavaTypeUtils.flatten(cr.externalize()).toString());
        Assertions.assertEquals(refType, cr.refType().toString());
    }


    public static Object[][] fieldRefs() {
        return new Object[][]{
                {"a.b::c:int", "a.b", "c", "int"},
                {"a.b.c::d:int", "a.b.c", "d", "int"},
                {"java.lang.System::out:java.io.PrintStream", "java.lang.System", "out", "java.io.PrintStream"},
                {"R<R::<T extends java.lang.Number>>::n:R::<T extends java.lang.Number>", "R<R::<T extends java.lang.Number>>", "n", "R::<T extends java.lang.Number>"}
        };
    }

    @ParameterizedTest
    @MethodSource("fieldRefs")
    public void testFieldRef(String fds, String refType, String name, String type) {
        FieldRef fr = refFromFlatString(fds);
        Assertions.assertEquals(fds, fr.toString());
        Assertions.assertEquals(refType, fr.refType().toString());
        Assertions.assertEquals(name, fr.name());
        Assertions.assertEquals(type, fr.type().toString());
    }

    public static Object[][] externalizedFieldRefs() {
        return new Object[][]{
                {"java.ref:\"a.b::c:int\"", "a.b", "c", "int"},
                {"java.ref:\"a.b.c::d:int\"", "a.b.c", "d", "int"},
                {"java.ref:\"java.lang.System::out:java.io.PrintStream\"", "java.lang.System", "out", "java.io.PrintStream"},
                {"java.ref:\"R<R::<T extends java.lang.Number>>::n:R::<T extends java.lang.Number>\"", "R<R::<T extends java.lang.Number>>", "n", "R::<T extends java.lang.Number>"}
        };
    }

    @ParameterizedTest
    @MethodSource("externalizedFieldRefs")
    public void testExternalizedFieldRef(String frs, String refType, String name, String type) {
        ExternalizedTypeElement efr = ExternalizedTypeElement.ofString(frs);
        FieldRef fr = (FieldRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(efr));

        Assertions.assertEquals(frs, JavaTypeUtils.flatten(fr.externalize()).toString());
        Assertions.assertEquals(refType, fr.refType().toString());
        Assertions.assertEquals(name, fr.name());
        Assertions.assertEquals(type, fr.type().toString());
    }


    public static Object[][] recordTypeRefs() {
        return new Object[][]{
                {"()A"},
                {"(B b)A"},
                {"(B b, C c)A"},
                {"(p.Func<String, Number> f, Entry<List<String>, val> e, int i, long l)p.A<R>"},
                {"(R::<T extends java.lang.Number> n)R<R::<T extends java.lang.Number>>"}
        };
    }

    @ParameterizedTest
    @MethodSource("recordTypeRefs")
    public void testRecordTypeRef(String rtds) {
        RecordTypeRef rtr = refFromFlatString(rtds);
        Assertions.assertEquals(rtds, rtr.toString());
    }

    public static Object[][] externalizedRecordTypeRefs() {
        return new Object[][]{
                {"java.ref:\"()A\""},
                {"java.ref:\"(B b)A\""},
                {"java.ref:\"(B b, C c)A\""},
                {"java.ref:\"(p.Func<String, Number> f, Entry<List<String>, val> e, int i, long l)p.A<R>\""},
                {"java.ref:\"(R::<T extends java.lang.Number> n)R<R::<T extends java.lang.Number>>\""}
        };
    }

    @ParameterizedTest
    @MethodSource("externalizedRecordTypeRefs")
    public void testExternalizedRecordTypeRef(String rtds) {
        ExternalizedTypeElement ertr = ExternalizedTypeElement.ofString(rtds);
        RecordTypeRef rtr = (RecordTypeRef) JavaTypeUtils.toJavaRef(JavaTypeUtils.inflate(ertr));
        Assertions.assertEquals(rtds, JavaTypeUtils.flatten(rtr.externalize()).toString());
    }

    @SuppressWarnings("unchecked")
    private static <R extends JavaRef> R refFromFlatString(String desc) {
        return (R)JavaTypeUtils.toJavaRef(JavaTypeUtils.parseExternalRefString(desc));
    }
}
