/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.lang.classfile.Attributes;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.attribute.CodeAttribute;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/*
 * @test
 * @modules jdk.incubator.code
 * @enablePreview
 * @run junit TestNestedTryCatch
 */
public class TestNestedTryCatch {

    static void nop() {}

    @Reflect
    static void f() {
        try {
            try {
                nop();
            } catch (Exception _) {
                return;
            }
        } catch (Exception _) {
            return;
        }
    }

    @Test
    public void testInnerHandlerStartIsInsideRegion() throws Exception {
        CoreOp.FuncOp f = Op.ofMethod(TestNestedTryCatch.class.getDeclaredMethod("f")).get();
        byte[] classdata = BytecodeGenerator.generateClassData(MethodHandles.lookup(), f);
        CodeModel code = ClassFile.of().parse(classdata).methods().getFirst().code().orElseThrow();
        // inner exception handler starts with astore_0 and the astore_0 instruction must be inside of the outer try region
        // the only non-intrusive test is to look for empty region as a sign of the exception region missplacement
        code.exceptionHandlers().forEach(ec ->
                Assertions.assertNotSame(ec.tryStart(), ec.tryEnd(),
                        () -> "Empty exception region in:\n" + code.toDebugString()));
    }
}
