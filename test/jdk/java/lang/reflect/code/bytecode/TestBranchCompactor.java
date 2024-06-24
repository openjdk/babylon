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

import java.lang.classfile.Attributes;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.components.ClassPrinter;
import static java.lang.classfile.Opcode.*;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.reflect.code.bytecode.BranchCompactor;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/*
 * @test
 * @enablePreview
 * @run testng TestBranchCompactor
 */
public class TestBranchCompactor {

    @Test
    public void testBranchCompactor() {
        var cc = ClassFile.of(ClassFile.StackMapsOption.DROP_STACK_MAPS);
        var clm = cc.parse(cc.build(ClassDesc.of("c"), clb -> clb.withMethodBody("m", ConstantDescs.MTD_void, 0,
                cb -> cb.transforming(new BranchCompactor(), cob -> {
                    var l = cob.newLabel();
                    cob.goto_(l) //compact
                       .lineNumber(1)
                       .labelBinding(l)
                       .nop();

                    l = cob.newLabel();
                    cob.goto_w(l) //compact
                       .lineNumber(2)
                       .labelBinding(l);

                    l = cob.newLabel();
                    cob.goto_(l) //compact
                       .labelBinding(l);

                    cob.iconst_0();
                    l = cob.newLabel();
                    cob.ifeq(l) //do not compact
                       .labelBinding(l);

                    l = cob.newLabel();
                    cob.goto_(l) //do not compact
                       .nop()
                       .labelBinding(l)
                       .return_();
                }))));
        var code = clm.methods().get(0).code().get();
        ClassPrinter.toYaml(code, ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
        Assert.assertEquals(
                code.elementList().stream().mapMulti((e, ec) -> {if (e instanceof Instruction i) ec.accept(i.opcode());}).toList(),
                List.of(NOP, ICONST_0, IFEQ, GOTO, NOP, RETURN));
        Assert.assertEquals(code.findAttribute(Attributes.lineNumberTable()).get().lineNumbers().size(), 2);
    }
}
