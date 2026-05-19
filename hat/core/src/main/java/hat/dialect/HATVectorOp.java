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
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.util.ops.Precedence;
import optkl.util.ops.StatementLikeOp;
import optkl.util.ops.VarLikeOp;

import java.util.List;
import java.util.Map;


public abstract sealed class HATVectorOp extends HATOp implements VarLikeOp {

    // TODO all these fields should be final
    private  String varName;
    private final CodeType resultType;
    private final Vector.Shape vectorShape;

    protected HATVectorOp(String varName, CodeType resultType, Vector.Shape vectorShape, List<Value> operands) {
        super(operands);
        this.varName = varName;
        this.resultType = resultType;
        this.vectorShape = vectorShape;
    }

    protected HATVectorOp(HATVectorOp that, CodeContext cc) {
        super(that, cc);
        this.varName = that.varName;
        this.resultType = that.resultType;
        this.vectorShape = that.vectorShape;
    }
    @Override
    public final String varName() {
        return varName;
    }

    public final Vector.Shape vectorShape(){return vectorShape;}

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
    public final CodeType resultType() {
        return resultType;
    }
    public String buildType() {
        return vectorShape.codeType().toString() + vectorShape.lanes();
    }

    public abstract static sealed class HATVectorBinaryOp extends HATVectorOp {

        private final BinaryOpEnum operationType;
        protected HATVectorBinaryOp(String varName,  BinaryOpEnum operationType, Vector.Shape vectorShape, List<Value> operands) {
            super(  varName /* this is clearly wrong binary ops have no name */,
                    vectorShape.codeType(), // also why does the base type need this twice?
                    vectorShape,
                    operands);
            this.operationType = operationType;
        }

        protected HATVectorBinaryOp(HATVectorBinaryOp op, CodeContext copyContext) {
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

    public abstract static sealed class HATVectorLoadOp extends HATVectorOp implements Precedence.LoadOrConv {


        protected HATVectorLoadOp(String varName, CodeType codeType, Vector.Shape vectorShape,  List<Value> operands) {
            super(varName, codeType, vectorShape, operands);
        }

        protected HATVectorLoadOp(HATVectorLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
        }


        public static final class HATPrivateVectorLoadOp extends HATVectorLoadOp implements Private{
            public HATPrivateVectorLoadOp(String varName, CodeType codeType, Vector.Shape vectorShape,  List<Value> operands) {
                super(varName, codeType, vectorShape,  operands);
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
            public HATSharedVectorLoadOp(String varName, CodeType codeType, Vector.Shape vectorShape, List<Value> operands) {
                super(varName, codeType, vectorShape, operands);
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
        public HATVectorOfOp(CodeType resultType, Vector.Shape vectorShape, List<Value> operands) {
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
        public HATVectorMakeOfOp(String varName, CodeType resultType, int vectorWidth, List<Value> operands) {
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

        public HATVectorSelectLoadOp(String varName, CodeType resultType, int lane, List<Value> operands) {
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

    public abstract static sealed class HATVectorStoreView extends HATVectorOp {

        protected HATVectorStoreView(String varName, CodeType resultType, Vector.Shape vectorShape, /* boolean isSharedOrPrivate*/ List<Value> operands) {
            super(varName, resultType, vectorShape, operands);
        }

        protected HATVectorStoreView(HATVectorStoreView op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HATSharedVectorStoreView extends HATVectorStoreView implements Shared{

            public HATSharedVectorStoreView(String varName, CodeType resultType, Vector.Shape vectorShape, List<Value> operands) {
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
                return Map.of("hat.dialect." + vectorShape().codeType() + "SharedStoreView." + varName(), resultType());
            }

        }
        public static final class HATPrivateVectorStoreView extends HATVectorStoreView implements Private{

            public HATPrivateVectorStoreView(String varName, CodeType resultType, Vector.Shape vectorShape, List<Value> operands) {
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
                return Map.of("hat.dialect." + vectorShape().codeType() + "PrivateStoreView." + varName(), resultType());
            }
        }

    }

    public static final class HATVectorVarLoadOp extends HATVectorOp {

        public HATVectorVarLoadOp(String varName, CodeType resultType, Vector.Shape vectorShape,  List<Value> operands) {
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