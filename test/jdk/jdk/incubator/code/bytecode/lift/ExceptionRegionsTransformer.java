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

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

final class ExceptionRegionsTransformer implements CodeTransformer {

    private boolean modified;

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {
        var t = new ExceptionRegionsTransformer();
        do {
            t.modified = false;
            func = func.transform(t);
        } while (t.modified);
        return func;
    }

    @Override
    public void acceptBlock(Block.Builder builder, Block b) {
        if (!builder.context().queryValue(b.terminatingOp().result()).isPresent()) {
            CodeTransformer.super.acceptBlock(builder, b);
        }
    }

    @Override
    public Block.Builder acceptOp(Block.Builder builder, Op op) {
        if (op instanceof JavaOp.ExceptionRegionExit exit
                && exit.endReference().targetBlock().ops().size() == 1
                && exit.endReference().targetBlock().terminatingOp() instanceof JavaOp.ExceptionRegionEnter enter
                && !exit.endReference().targetBlock().predecessors().isEmpty()
                && exit.endReference().targetBlock().predecessors().stream().allMatch(predecessor ->
                    predecessor.terminatingOp() instanceof JavaOp.ExceptionRegionExit e
                            && e.endReference().targetBlock() == exit.endReference().targetBlock()
                            && e.enterOp() == exit.enterOp())
                && exit.enterOp().catchReferences().stream().map(Block.Reference::targetBlock).toList()
                        .equals(enter.catchReferences().stream().map(Block.Reference::targetBlock).toList())
                && enter.result().isDominatedBy(exit.enterOp().result())) {
            var cc = builder.context();
            cc.mapValue(enter.result(), cc.getValue(exit.enterOp().result()));
            cc.mapValues(exit.endReference().targetBlock().parameters(), cc.getValues(exit.endReference().arguments()));
            builder.add(CoreOp.branch(cc.getReferenceOrCreate(enter.startReference())));
            modified = true;
        } else {
            builder.add(op);
        }
        return builder;
    }
}
