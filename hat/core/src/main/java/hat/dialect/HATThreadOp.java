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

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import optkl.util.ops.Precedence;

import java.util.List;
import java.util.Map;

public abstract sealed class HATThreadOp extends HATOp implements Precedence.LoadOrConv
        permits HATThreadOp.HATBlockThreadIdOp, HATThreadOp.HATGlobalSizeOp, HATThreadOp.HATGlobalThreadIdOp, HATThreadOp.HATLocalSizeOp, HATThreadOp.HATLocalThreadIdOp {
   final  private String name;
   final  private TypeElement resultType;
   final  private int dimension;

    public HATThreadOp(String name, TypeElement resultType,int dimension, List<Value> operands) {
        super(operands);
        this.name = name;
        this.resultType = resultType;
        this.dimension = dimension;
    }

    protected HATThreadOp(HATThreadOp that, CodeContext cc) {
        super(that, cc);
        this.name =that.name;
        this.resultType = that.resultType;
        this.dimension = that.dimension;
    }

    public int getDimension() {
        return dimension;
    }


    @Override
    final public TypeElement resultType() {
        return resultType;
    }

    @Override
    final public Map<String, Object> externalize() {
        return Map.of("hat.dialect." + name, this.getDimension());
    }

    public static final class HATLocalThreadIdOp extends HATThreadOp {

        public HATLocalThreadIdOp(int dimension, TypeElement resultType) {
            super("LocalThreadId",resultType,dimension, List.of());
        }

        public HATLocalThreadIdOp(HATLocalThreadIdOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATLocalThreadIdOp(this, copyContext);
        }

        public static HATLocalThreadIdOp of(int dimension, TypeElement resultType){
            return new HATLocalThreadIdOp(dimension,resultType);
        }
    }

    public static final class HATBlockThreadIdOp extends HATThreadOp {
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


        public static HATBlockThreadIdOp of(int dimension, TypeElement resultType){
            return new HATBlockThreadIdOp(dimension,resultType);
        }
    }

    public static final class HATLocalSizeOp extends HATThreadOp {

        public HATLocalSizeOp(int dimension, TypeElement resultType) {
            super("GlobalThreadSize",resultType,dimension, List.of());
        }

        public HATLocalSizeOp(HATLocalSizeOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATLocalSizeOp(this, copyContext);
        }

        public static HATThreadOp of(int dimension, TypeElement resultType){
            return new HATLocalSizeOp(dimension, resultType);
        }
    }

    public static final class HATGlobalThreadIdOp extends HATThreadOp {

        public HATGlobalThreadIdOp(int dimension, TypeElement resultType) {
            super("GlobalThreadId",resultType,dimension, List.of());
        }

        public HATGlobalThreadIdOp(HATGlobalThreadIdOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATGlobalThreadIdOp(this, copyContext);
        }

        public static HATGlobalThreadIdOp of(int dimension, TypeElement resultType){
            return new HATGlobalThreadIdOp(dimension, resultType);
        }
    }

    public static final class HATGlobalSizeOp extends HATThreadOp {
        public HATGlobalSizeOp(int dimension, TypeElement resultType) {
            super("GlobalThreadSize",resultType,dimension, List.of());
        }

        public HATGlobalSizeOp(HATGlobalSizeOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATGlobalSizeOp(this, copyContext);
        }


        static public HATGlobalSizeOp of(int dimension, TypeElement resultType){
            return new HATGlobalSizeOp(dimension,resultType);
        }
    }
}