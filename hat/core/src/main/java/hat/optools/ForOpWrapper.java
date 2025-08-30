/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package hat.optools;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.stream.Stream;

public class ForOpWrapper extends LoopOpWrapper<JavaOp.ForOp> {
    ForOpWrapper( MethodHandles.Lookup lookup,JavaOp.ForOp op) {
        super(lookup,op);
    }

    public Stream<OpWrapper<?>> initWrappedYieldOpStream() {
       return  op.init().entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o->OpWrapper.wrap(lookup,o) );
    }

    @Override
    public Stream<OpWrapper<?>> conditionWrappedYieldOpStream() {
        return  op.cond().entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o->OpWrapper.wrap(lookup,o) );
    }

    public Stream<OpWrapper<?>> mutateRootWrappedOpStream() {
          return RootSet.rootsWithoutVarFuncDeclarationsOrYields(lookup,op.bodies().get(2).entryBlock());
    }

    @Override
    public Stream<OpWrapper<?>> loopWrappedRootOpStream() {
        var list = new ArrayList<>(RootSet.rootsWithoutVarFuncDeclarationsOrYields(lookup,op.bodies().get(3).entryBlock()).toList());
        if (list.getLast() instanceof JavaContinueOpWrapper) {
            list.removeLast();
        }
        return list.stream();
    }
}
