/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.dialect;

import hat.NDRange;
import hat.optools.OpTk;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Regex;

import java.util.List;

public final class HATBlockThreadIdOp extends HATThreadOp {
    public HATBlockThreadIdOp(int dimension, TypeElement resultType) {
        super("BlockThreadId", resultType,dimension, List.of());
    }

    public HATBlockThreadIdOp(HATBlockThreadIdOp op, CodeContext copyContext) {
        super(op, copyContext);
    }

    @Override
    public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
        return new HATBlockThreadIdOp(this, copyContext);
    }

    public final static Regex regex = NDRange.Block.idxRegex;

    public static HATBlockThreadIdOp of(int dimension, TypeElement resultType){
        return new HATBlockThreadIdOp(dimension,resultType);
    }
}
