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
package optkl;

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;
import optkl.util.BiMap;

import java.lang.invoke.MethodHandles;

public interface MappedIfaceBufferInvokeQuery extends Query<JavaOp.InvokeOp,Invoke, MappedIfaceBufferInvokeQuery> {
    interface OK extends Match<JavaOp.InvokeOp, Invoke, MappedIfaceBufferInvokeQuery>{
         boolean mutatesBuffer();
    }
    record Impl(MethodHandles.Lookup lookup) implements MappedIfaceBufferInvokeQuery {
        @Override
        public Res<JavaOp.InvokeOp,Invoke, MappedIfaceBufferInvokeQuery> test(CodeElement<?, ?> ce) {
            if (Invoke.invoke(lookup, ce) instanceof Invoke invoke) {
                if (invoke.isInstance() && invoke.returns(MappedIfaceBufferInvokeQuery.class) || invoke.returnsPrimitive()){
                    record MatchImpl(MappedIfaceBufferInvokeQuery query, Invoke helper, boolean mutatesBuffer) implements OK{
                        @Override
                        public Match<JavaOp.InvokeOp, Invoke, MappedIfaceBufferInvokeQuery> remap(BiMap<CodeElement<?, ?>, CodeElement<?, ?>> biMap) {
                            return new MatchImpl(MatchImpl.this.query, Invoke.invoke(query().lookup(), biMap.getTo(MatchImpl.this.helper.op())),MatchImpl.this.mutatesBuffer);
                        }
                    }
                    return new MatchImpl(this, invoke, invoke.returnsVoid());
                }
            }
            return Query.FAILED;
        }
    }
    static MappedIfaceBufferInvokeQuery create(MethodHandles.Lookup lookup) {
         return new Impl(lookup);
    }
}
