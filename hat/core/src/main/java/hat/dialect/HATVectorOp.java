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

import optkl.IfaceValue.Vector;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
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
    private final Vector.Shape vectorShape;

    public HATVectorOp(String varName, TypeElement resultType, Vector.Shape vectorShape, List<Value> operands) {
        super(operands);
        this.varName = varName;
        this.resultType = resultType;
        this.vectorShape = vectorShape;
      //  if (!vectorShape.typeElement().equals(resultType)){
        //    System.out.println("resulttype = "+resultType+ " vectorshape.typeElement = "+vectorShape.typeElement());
       // }
    }

    protected HATVectorOp(HATVectorOp that, CodeContext cc) {
        super(that, cc);
        this.varName = that.varName;
        this.resultType = that.resultType;
        this.vectorShape = that.vectorShape;
    }
    @Override
    final public String varName() {
        return varName;
    }
    final public Vector.Shape vectorShape(){return vectorShape;}

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
        return vectorShape.typeElement().toString() + vectorShape.lanes();
    }

    public abstract sealed static class HATVectorBinaryOp extends HATVectorOp
            permits HATVectorBinaryOp.HATVectorAddOp, hat.dialect.HATVectorOp.HATVectorBinaryOp.HATVectorDivOp, hat.dialect.HATVectorOp.HATVectorBinaryOp.HATVectorMulOp, hat.dialect.HATVectorOp.HATVectorBinaryOp.HATVectorSubOp {

        private final BinaryOpEnum operationType;
        public HATVectorBinaryOp(String varName,  BinaryOpEnum operationType, Vector.Shape vectorShape, List<Value> operands) {
            super(  varName /* this is clearly wrong binary ops have no name */,
                    vectorShape.typeElement(), // also why does the base type need this twice?
                    vectorShape,
                    operands);
            this.operationType = operationType;
        }

        public HATVectorBinaryOp(HATVectorBinaryOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.operationType = op.operationType;
        }

        public BinaryOpEnum operationType() {
            return operationType;
        }

        public static final class HATVectorAddOp extends HATVectorBinaryOp implements Precedence.Additive {
            public HATVectorAddOp(String varName, Vector.Shape vectorShape, List<Value> operands) {
                super(varName, BinaryOpEnum.ADD, vectorShape, operands);
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
            public HATVectorDivOp(String varName, Vector.Shape vectorShape, List<Value> operands) {
                super(varName,BinaryOpEnum.DIV, vectorShape, operands);
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

            public HATVectorMulOp(String varName, Vector.Shape vectorShape, List<Value> operands) {
                super(varName, BinaryOpEnum.MUL, vectorShape, operands);
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

            public HATVectorSubOp(String varName, Vector.Shape vectorShape, List<Value> operands) {
                super(varName, BinaryOpEnum.SUB, vectorShape, operands);
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
    public interface Shared{
    }

    public interface Private{
    }

    public static abstract sealed class HATVectorLoadOp extends HATVectorOp implements Precedence.LoadOrConv permits HATVectorLoadOp.HATPrivateVectorLoadOp, HATVectorLoadOp.HATSharedVectorLoadOp {


        protected HATVectorLoadOp(String varName, TypeElement typeElement, Vector.Shape vectorShape,  List<Value> operands) {
            super(varName, typeElement, vectorShape, operands);
        }

        protected HATVectorLoadOp(HATVectorLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
        }


        public static final class HATPrivateVectorLoadOp extends HATVectorLoadOp implements Private{
            public HATPrivateVectorLoadOp(String varName, TypeElement typeElement, Vector.Shape vectorShape,  List<Value> operands) {
                super(varName, typeElement, vectorShape,  operands);
            }
            public HATPrivateVectorLoadOp(HATPrivateVectorLoadOp op, CodeContext copyContext) {
                super(op, copyContext);
            }


            @Override
            public Op transform(CodeContext cc, CodeTransformer ot) {
                return new HATPrivateVectorLoadOp(this, cc);
            }
            @Override
            public Map<String, Object> externalize() {
                return Map.of("hat.dialect.vectorPrivateLoad." + varName(), resultType());
            }
        }
        public static final class HATSharedVectorLoadOp extends HATVectorLoadOp implements Shared{
            public HATSharedVectorLoadOp(String varName, TypeElement typeElement, Vector.Shape vectorShape, List<Value> operands) {
                super(varName, typeElement, vectorShape, operands);
            }
            public HATSharedVectorLoadOp(HATSharedVectorLoadOp op, CodeContext copyContext) {
                super(op, copyContext);
            }


            @Override
            public Op transform(CodeContext cc, CodeTransformer ot) {
                return new HATSharedVectorLoadOp(this,cc);
            }
            @Override
            public Map<String, Object> externalize() {
                return Map.of("hat.dialect.vectorSharedLoad." + varName(), resultType());
            }

        }
    }

    public static final class HATVectorOfOp extends HATVectorOp {
        public HATVectorOfOp(TypeElement resultType, Vector.Shape vectorShape, List<Value> operands) {
            super("", resultType, vectorShape, operands);
        }

        public HATVectorOfOp(HATVectorOfOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorOfOp(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.vectorOf." + varName(), resultType());
        }

    }

    public static final class HATVectorMakeOfOp extends HATVectorOp {
        public HATVectorMakeOfOp(String varName, TypeElement resultType, int vectorWidth, List<Value> operands) {
            super(varName, resultType,  Vector.Shape.of(resultType,vectorWidth), operands);
        }

        public HATVectorMakeOfOp(HATVectorMakeOfOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorMakeOfOp(this, copyContext);
        }
        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.makeOf." + varName(), resultType());
        }

    }

    public static final class HATVectorSelectLoadOp extends HATVectorOp implements Precedence.LoadOrConv {

        private final int lane;

        public HATVectorSelectLoadOp(String varName, TypeElement resultType, int lane, List<Value> operands) {
            super(varName, resultType, Vector.Shape.of(JavaType.VOID, -1), operands); // looks like we have a hiearchy mixup
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
        private final String resolvedName;

        public HATVectorSelectStoreOp(String varName, int lane, String resolvedName, List<Value> operands) {
            super(varName, JavaType.VOID, Vector.Shape.of(JavaType.VOID, -1), operands); // This seems so wrong.
            this.lane = lane;
            this.resolvedName = resolvedName;
        }

        public HATVectorSelectStoreOp(HATVectorSelectStoreOp that, CodeContext cc) {
            super(that, cc);
            this.lane = that.lane;
            this.resolvedName = that.resolvedName;
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
        public String resolvedName(){
            return resolvedName;
        }

    }

    public static abstract sealed class HATVectorStoreView extends HATVectorOp
            permits HATVectorStoreView.HATPrivateVectorStoreView, HATVectorStoreView.HATSharedVectorStoreView {

        protected HATVectorStoreView(String varName, TypeElement resultType, Vector.Shape vectorShape, /* boolean isSharedOrPrivate*/ List<Value> operands) {
            super(varName, resultType, vectorShape, operands);
        }

        protected HATVectorStoreView(HATVectorStoreView op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HATSharedVectorStoreView extends HATVectorStoreView implements Shared{

            public HATSharedVectorStoreView(String varName, TypeElement resultType, Vector.Shape vectorShape, List<Value> operands) {
                super(varName, resultType, vectorShape, operands);
            }

            public HATSharedVectorStoreView(HATSharedVectorStoreView op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATSharedVectorStoreView(this, copyContext);
            }

            @Override
            public Map<String, Object> externalize() {
                return Map.of("hat.dialect." + vectorShape().typeElement() + "SharedStoreView." + varName(), resultType());
            }

        }
        public static final class HATPrivateVectorStoreView extends HATVectorStoreView implements Private{

            public HATPrivateVectorStoreView(String varName, TypeElement resultType, Vector.Shape vectorShape, List<Value> operands) {
                super(varName, resultType, vectorShape, operands);
            }

            public HATPrivateVectorStoreView(HATPrivateVectorStoreView op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATPrivateVectorStoreView(this, copyContext);
            }

            @Override
            public Map<String, Object> externalize() {
                return Map.of("hat.dialect." + vectorShape().typeElement() + "PrivateStoreView." + varName(), resultType());
            }
        }

    }

    public static final class HATVectorVarLoadOp extends HATVectorOp {

        public HATVectorVarLoadOp(String varName, TypeElement resultType, Vector.Shape vectorShape,  List<Value> operands) {
            super(varName, resultType, vectorShape, operands);
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
        public HATVectorVarOp(String varName, VarType resultType,  Vector.Shape vectorShape, List<Value> operands) {
            super(varName, resultType, vectorShape, operands);
        }

        public HATVectorVarOp(HATVectorVarOp op, CodeContext copyContext) {
            super(op, copyContext);
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