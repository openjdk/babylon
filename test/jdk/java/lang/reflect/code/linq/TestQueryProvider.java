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

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.type.JavaType;

import static java.lang.reflect.code.descriptor.MethodTypeDesc.methodType;
import static java.lang.reflect.code.op.CoreOps._return;
import static java.lang.reflect.code.op.CoreOps.func;
import static java.lang.reflect.code.type.JavaType.type;

public final class TestQueryProvider extends QueryProvider {
    public TestQueryProvider() {
    }

    @Override
    public <T> Queryable<T> query(Class<T> elementType) {
        return new TestQueryable<>(elementType, this);
    }

    @Override
    protected Queryable<?> createQuery(JavaType elementType, CoreOps.FuncOp expression) {
        return new TestQueryable<>(elementType, this, expression);
    }

    @Override
    protected QueryResult<?> createQueryResult(JavaType resultType, CoreOps.FuncOp expression) {
        return new TestQueryResult<>(resultType, expression);
    }

    static final class TestQueryable<T> implements Queryable<T> {
        final JavaType elementType;
        final TestQueryProvider provider;
        final CoreOps.FuncOp expression;

        TestQueryable(Class<T> tableClass, TestQueryProvider qp) {
            this.elementType = type(tableClass);
            this.provider = qp;

            JavaType queryableType = type(Queryable.TYPE, elementType);
            // Initial expression is an identity function
            var funDescriptor = methodType(queryableType, queryableType);
            this.expression = func("query", funDescriptor)
                    .body(b -> b.op(_return(b.parameters().get(0))));
        }

        TestQueryable(JavaType elementType, TestQueryProvider provider, CoreOps.FuncOp expression) {
            this.elementType = elementType;
            this.provider = provider;
            this.expression = expression;
        }

        @Override
        public QueryProvider provider() {
            return provider;
        }

        @Override
        public JavaType elementType() {
            return elementType;
        }

        @Override
        public CoreOps.FuncOp expression() {
            return expression;
        }
    }

    record TestQueryResult<T>(JavaType resultType, CoreOps.FuncOp expression) implements QueryResult<T> {
    }
}
