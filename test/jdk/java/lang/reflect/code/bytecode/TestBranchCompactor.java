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

import java.lang.classfile.ClassFile;
import java.lang.classfile.components.ClassPrinter;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.reflect.code.bytecode.BranchCompactor;
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
        var cc = ClassFile.of();
        var clm = cc.parse(cc.build(ClassDesc.of("c"), clb -> clb.withMethodBody("m", ConstantDescs.MTD_void, 0,
                cb -> cb.transforming(new BranchCompactor(), cob -> {
                    var l1 = cob.newLabel();
                    cob.goto_(l1);
                    cob.labelBinding(l1);
                    l1 = cob.newLabel();
                    cob.goto_(l1);
                    cob.labelBinding(l1);
                    cob.iconst_0();
                    cob.ifThenElse(tb -> {
                        var l2 = tb.newLabel();
                        tb.goto_(l2);
                        tb.labelBinding(l2);
                        l2 = tb.newLabel();
                        tb.goto_(l2);
                        tb.labelBinding(l2);
                    }, eb -> {
                        var l2 = eb.newLabel();
                        eb.goto_(l2);
                        eb.labelBinding(l2);
                        l2 = eb.newLabel();
                        eb.goto_(l2);
                        eb.labelBinding(l2);
                    });
                    l1 = cob.newLabel();
                    cob.goto_(l1);
                    cob.labelBinding(l1);
                    l1 = cob.newLabel();
                    cob.goto_(l1);
                    cob.labelBinding(l1);
                    cob.return_();
                }))));

        ClassPrinter.toYaml(clm, ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
        //only iconst_0 and return_ should remain
        Assert.assertEquals(clm.methods().get(0).code().get().elementList().size(), 2);
    }
}
