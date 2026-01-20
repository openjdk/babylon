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
package hat.phases;

import hat.KernelContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper.FieldAccess;
import optkl.Query;
import optkl.util.BiMap;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.util.function.Predicate;

public interface KernelContextThreadIdFieldAccessQuery extends Query<JavaOp.FieldAccessOp, FieldAccess, KernelContextThreadIdFieldAccessQuery> {
    Regex threadIdRegex = Regex.of("[glb][is][xyz]");
    interface Match extends SimpleMatch<JavaOp.FieldAccessOp, FieldAccess, KernelContextThreadIdFieldAccessQuery> {
         String id();
    }
    record Impl(MethodHandles.Lookup lookup) implements KernelContextThreadIdFieldAccessQuery {
        @Override
        public Res<JavaOp.FieldAccessOp,FieldAccess, KernelContextThreadIdFieldAccessQuery> matches(CodeElement<?, ?> ce, Predicate<FieldAccess> predicate) {
            if (FieldAccess.fieldAccess(lookup, ce) instanceof FieldAccess fieldAccess && predicate.test(fieldAccess)
                && fieldAccess.isInstance() && fieldAccess.refType(KernelContext.class) && fieldAccess.named(threadIdRegex)){
                    record MatchImpl(KernelContextThreadIdFieldAccessQuery query, FieldAccess helper, String id) implements Match{
                        @Override
                        public SimpleMatch<JavaOp.FieldAccessOp, FieldAccess, KernelContextThreadIdFieldAccessQuery> remap(BiMap<CodeElement<?, ?>, CodeElement<?, ?>> biMap) {
                            return new MatchImpl(MatchImpl.this.query, FieldAccess.fieldAccess(query().lookup(), biMap.getTo(MatchImpl.this.helper.op())),MatchImpl.this.id);
                        }
                    }
                    return new MatchImpl(this, fieldAccess, fieldAccess.name() );
                }
            return Query.FAILED;
        }
    }
    static KernelContextThreadIdFieldAccessQuery create(MethodHandles.Lookup lookup) {
         return new Impl(lookup);
    }
}
