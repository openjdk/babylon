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
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.util.ops.Precedence;
import optkl.util.ops.StatementLikeOp;
import optkl.util.ops.VarLikeOp;

import java.util.List;
import java.util.Map;


public abstract sealed class HATVectorOp extends HATOp implements VarLikeOp
        permits HATVectorOp.HATVectorBinaryOp, HATVectorOp.HATVectorLoadOp, HATVectorOp.HATVectorMakeOfOp, HATVectorOp.HATVectorOfOp, HATVectorOp.HATVectorSelectLoadOp, HATVectorOp.HATVectorSelectStoreOp, HATVectorOp.HATVectorStoreView, HATVectorOp.HATVectorVarLoadOp, HATVectorOp.HATVectorVarOp {
    // TODO all these fields should be final
    private  String varName;
    private final TypeElement resultType;
    private final int vectorN;
    private final TypeElement vectorElementType;

    public HATVectorOp(String varName, TypeElement resultType, TypeElement vectorElementType, int vectorN, List<Value> operands) {
        super(operands);
        this.varName = varName;
        this.resultType = resultType;
        this.vectorN = vectorN;
        this.vectorElementType = vectorElementType;
    }

    protected HATVectorOp(HATVectorOp that, CodeContext cc) {
        super(that, cc);
        this.varName = that.varName;
        this.resultType = that.resultType;
        this.vectorN = that.vectorN;
        this.vectorElementType = that.vectorElementType;
    }
   // @Override
    final public String varName() {
        return varName;
    }
    final public TypeElement vectorElementType(){return vectorElementType;}

    //TODO all these fields should be final why do we allow this to be mutated.
    public void varName(String varName) {
        this.varName = varName;
    }

    public String mapLane(int lane) {
        return switch (lane) {
            case 0 -> "x";
            case 1 -> "y";
            case 2 -> "z";
            case 3 -> "w";
            default -> throw new InternalError("Invalid lane: " + lane);
        };
    }
    @Override
    final public TypeElement resultType() {
        return resultType;
    }
    public String buildType() {
        return vectorElementType.toString() + vectorN;
    }

    public int vectorN() {
        return vectorN;
    }

    public abstract sealed static class HATVectorBinaryOp extends HATVectorOp
            permits HATVectorBinaryOp.HATVectorAddOp, hat.dialect.HATVectorOp.HATVectorBinaryOp.HATVectorDivOp, hat.dialect.HATVectorOp.HATVectorBinaryOp.HATVectorMulOp, hat.dialect.HATVectorOp.HATVectorBinaryOp.HATVectorSubOp {

        private final BinaryOpEnum operationType;

        public HATVectorBinaryOp(String varName, TypeElement resultType, BinaryOpEnum operationType, TypeElement vectorElementType, int width, List<Value> operands) {
            super(varName, resultType, vectorElementType, width, operands);
            //this.elementType = typeElement;
            this.operationType = operationType;

        }

        public HATVectorBinaryOp(HATVectorBinaryOp op, CodeContext copyContext) {
            super(op, copyContext);
           // this.elementType = op.elementType;
            this.operationType = op.operationType;
        }

      //  @Override
        //public TypeElement resultType() {
          //  return this.elementType;
       // }

        public BinaryOpEnum operationType() {
            return operationType;
        }

        public static final class HATVectorAddOp extends HATVectorBinaryOp implements Precedence.Additive {

            public HATVectorAddOp(String varName, TypeElement typeElement, TypeElement vectorElementType, int width, List<Value> operands) {
                super(varName, typeElement, BinaryOpEnum.ADD, vectorElementType, width, operands);
            }

            public HATVectorAddOp(HATVectorAddOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATVectorAddOp(this, copyContext);
            }
        }

        public static final class HATVectorDivOp extends HATVectorBinaryOp implements Precedence.Multiplicative {

            public HATVectorDivOp(String varName, TypeElement typeElement, TypeElement vectorElementType, int width, List<Value> operands) {
                super(varName, typeElement, BinaryOpEnum.DIV, vectorElementType, width, operands);
            }

            public HATVectorDivOp(HATVectorDivOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATVectorDivOp(this, copyContext);
            }
        }

        public static final class HATVectorMulOp extends HATVectorBinaryOp implements Precedence.Multiplicative {

            public HATVectorMulOp(String varName, TypeElement typeElement, TypeElement vectorElementType, int width, List<Value> operands) {
                super(varName, typeElement, BinaryOpEnum.MUL, vectorElementType, width, operands);
            }

            public HATVectorMulOp(HATVectorMulOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATVectorMulOp(this, copyContext);
            }
        }

        public static final class HATVectorSubOp extends HATVectorBinaryOp implements Precedence.Additive {

            public HATVectorSubOp(String varName, TypeElement typeElement, TypeElement vectorElementType, int width, List<Value> operands) {
                super(varName, typeElement, BinaryOpEnum.SUB, vectorElementType, width, operands);
            }

            public HATVectorSubOp(HATVectorSubOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATVectorSubOp(this, copyContext);
            }
        }
    }

    public static final class HATVectorLoadOp extends HATVectorOp implements Precedence.LoadOrConv {
        private final int loadN;
        private final boolean isSharedOrPrivate;

        public HATVectorLoadOp(String varName, TypeElement typeElement, TypeElement vectorType, int loadN, boolean isShared, List<Value> operands) {
            super(varName, typeElement, vectorType, loadN, operands);
            //this.typeElement = typeElement;
            this.loadN = loadN;
           // this.vectorType = vectorType;
            this.isSharedOrPrivate = isShared;
        }

        public HATVectorLoadOp(HATVectorLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
            //this.typeElement = op.typeElement;
            this.loadN = op.loadN;
           // this.vectorType = op.vectorType;
            this.isSharedOrPrivate = op.isSharedOrPrivate;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorLoadOp(this, copyContext);
        }

       // @Override
      //  public TypeElement resultType() {
        //    return typeElement;
       // }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.vectorLoadView." + varName(), resultType());
        }

        public boolean isSharedOrPrivate() {
            return this.isSharedOrPrivate;
        }
    }

    public static final class HATVectorOfOp extends HATVectorOp {

       // private final TypeElement typeElement;
        private final int loadN;

        public HATVectorOfOp(TypeElement resultType, TypeElement vectorTypeElement, int loadN, List<Value> operands) {
            super("", resultType, vectorTypeElement, loadN, operands);
           // this.typeElement = typeElement;
            this.loadN = loadN;
        }

        public HATVectorOfOp(HATVectorOfOp op, CodeContext copyContext) {
            super(op, copyContext);
           // this.typeElement = op.typeElement;
            this.loadN = op.loadN;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorOfOp(this, copyContext);
        }

       // @Override
       // public TypeElement resultType() {
         //   return typeElement;
       // }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.vectorOf." + varName(), resultType());
        }


    }

    public static final class HATVectorMakeOfOp extends HATVectorOp {

      //  private final TypeElement typeElement;
        private final int loadN;

        public HATVectorMakeOfOp(String varName, TypeElement resultType, int loadN, List<Value> operands) {
            super(varName, resultType, resultType, loadN, operands);
        //    this.typeElement = typeElement;
            this.loadN = loadN;
        }

        public HATVectorMakeOfOp(HATVectorMakeOfOp op, CodeContext copyContext) {
            super(op, copyContext);
          //  this.typeElement = op.typeElement;
            this.loadN = op.loadN;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorMakeOfOp(this, copyContext);
        }

       // @Override
       // public TypeElement resultType() {
         //   return typeElement;
       // }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.makeOf." + varName(), resultType());
        }

    }

    public static final class HATVectorSelectLoadOp extends HATVectorOp implements Precedence.LoadOrConv {

        private final int lane;

        public HATVectorSelectLoadOp(String varName, TypeElement resultType, int lane, List<Value> operands) {
            super(varName, resultType, resultType, -1, operands);
            this.lane = lane;
        }

        public HATVectorSelectLoadOp(HATVectorSelectLoadOp that, CodeContext cc) {
            super(that, cc);
            this.lane = that.lane;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorSelectLoadOp(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.vselect." + lane, resultType());
        }

        public String mapLane() {
            return super.mapLane(lane);
        }

    }

    public static final class HATVectorSelectStoreOp extends HATVectorOp {
        private final int lane;
        private final CoreOp.VarOp resultVarOp;

        public HATVectorSelectStoreOp(String varName,  int lane, CoreOp.VarOp resultVarOp, List<Value> operands) {
            super(varName, JavaType.VOID, JavaType.VOID, -1, operands);
            this.lane = lane;
            this.resultVarOp = resultVarOp;
        }

        public HATVectorSelectStoreOp(HATVectorSelectStoreOp that, CodeContext cc) {
            super(that, cc);
            this.lane = that.lane;
            this.resultVarOp = that.resultVarOp;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorSelectStoreOp(this, copyContext);
        }
        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.vselect.store." + lane, resultType());
        }

        public String mapLane() {
            return super.mapLane(lane);
        }

        public CoreOp.VarOp resultValue() {
            return resultVarOp;
        }

    }

    public static final class HATVectorStoreView extends HATVectorOp {

        private final boolean isSharedOrPrivate;

        public HATVectorStoreView(String varName, TypeElement resultType, int storeN, TypeElement vectorElementType, boolean isSharedOrPrivate, List<Value> operands) {
            super(varName, resultType, vectorElementType, storeN, operands);
            this.isSharedOrPrivate = isSharedOrPrivate;
        }

        public HATVectorStoreView(HATVectorStoreView op, CodeContext copyContext) {
            super(op, copyContext);
            this.isSharedOrPrivate = op.isSharedOrPrivate;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorStoreView(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect." + vectorElementType().toString() + vectorN() + "StoreView." + varName(), resultType());
        }

        public boolean isSharedOrPrivate() {
            return this.isSharedOrPrivate;
        }

    }

    public static final class HATVectorVarLoadOp extends HATVectorOp {

        public HATVectorVarLoadOp(String varName, TypeElement resultType, TypeElement vectorElementType, int width, List<Value> operands) {
            super(varName, resultType, vectorElementType, width, operands);
        }

        public HATVectorVarLoadOp(HATVectorVarLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorVarLoadOp(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.vectorVarLoadOp." + varName(), resultType());
        }

    }

    public static final class HATVectorVarOp extends HATVectorOp implements StatementLikeOp {

        private final int loadN;

        public HATVectorVarOp(String varName, VarType resultType, TypeElement vectorElementType, int loadN, List<Value> operands) {
            super(varName, resultType, vectorElementType, loadN, operands);
            this.loadN = loadN;
        }

        public HATVectorVarOp(HATVectorVarOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.loadN = op.loadN;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorVarOp(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.vectorVarOp." + varName(), resultType());
        }

    }
}