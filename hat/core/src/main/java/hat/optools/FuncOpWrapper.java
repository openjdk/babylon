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

import hat.OpsAndTypes;

import java.lang.foreign.GroupLayout;

import hat.util.BiMap;
import jdk.incubator.code.Block;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.stream.Stream;

public class FuncOpWrapper extends OpWrapper<CoreOp.FuncOp> {

    public static class ParamTable {
        public static class Info {
            public final int idx;
            public final Block.Parameter parameter;
            public final JavaType javaType;
            public final CoreOp.VarOp varOp;
            public GroupLayout layout = null;
            public Class<?> clazz = null;
            Info(int idx, Block.Parameter parameter, CoreOp.VarOp varOp) {
                this.idx = idx;
                this.parameter = parameter;
                this.javaType = (JavaType) parameter.type();
                this.varOp = varOp;
            }
            public boolean isPrimitive() {
                return javaType instanceof PrimitiveType;
            }
            public void setLayout(GroupLayout layout) {
                this.layout = layout;
            }
            public void setClass(Class<?> clazz) {
                this.clazz = clazz;
            }
        }

        final private Map<Block.Parameter, Info> parameterToInfo = new LinkedHashMap<>();
        final private Map<CoreOp.VarOp, ParamTable.Info> varOpToInfo = new LinkedHashMap<>();

        final private List<Info> list = new ArrayList<>();

        public ParamTable.Info info(Block.Parameter parameter) {
            return parameterToInfo.get(parameter);
        }

        public boolean isParameterVarOp(CoreOp.VarOp op) {
            return varOpToInfo.containsKey(op);
        }

        void add(Map.Entry<Block.Parameter, CoreOp.VarOp> parameterToVarOp) {
            //We add a new ParameterInfo to both maps using parameter and varOp as keys
            varOpToInfo.put(parameterToVarOp.getValue(),
                    // always called but convenient because computeIfAbsent returns what we added :)
                    parameterToInfo.computeIfAbsent(parameterToVarOp.getKey(), (parameterKey) -> {
                        var info = new ParamTable.Info(list.size(), parameterKey, parameterToVarOp.getValue());
                        list.add(info);
                        return info;
                    })
            );
        }

        public List<Info> list() {
            return list;
        }

        public Stream<Info> stream() {
            return list.stream();
        }
    }

    public final ParamTable paramTable = new ParamTable();

    public ParamTable paramTable() {
        return paramTable;
    }

    public ParamTable.Info paramInfo(int idx) {
        return paramTable.stream().toList().get(idx);
    }

    public Block.Parameter parameter(int idx) {
        return paramInfo(idx).parameter;
    }

    public BiMap<Block.Parameter, CoreOp.VarOp> parameterVarOpMap = new BiMap<>();
    public BiMap<Block.Parameter, JavaOp.InvokeOp> parameterInvokeOpMap = new BiMap<>();
    public BiMap<Block.Parameter, OpsAndTypes.HatPtrOp<?>> parameterHatPtrOpMap = new BiMap<>();
    public FuncOpWrapper( MethodHandles.Lookup lookup,CoreOp.FuncOp op) {
        super(lookup,op);
        op.parameters().forEach(parameter -> {
            Optional<Op.Result> optionalResult = parameter.uses().stream().findFirst();
            optionalResult.ifPresentOrElse(result -> {
                var resultOp = result.op();
                if (resultOp instanceof CoreOp.VarOp varOp) {
                    parameterVarOpMap.add(parameter, varOp);
                    paramTable.add(Map.entry(parameter, varOp));
                }else if (resultOp instanceof JavaOp.InvokeOp invokeOp) {
                    parameterInvokeOpMap.add(parameter,invokeOp);
                }else if (resultOp instanceof OpsAndTypes.HatPtrOp hatPtrOp) {
                    parameterHatPtrOpMap.add(parameter,hatPtrOp);
                }else{
                    //System.out.println("neither varOp or an invokeOp "+resultOp.getClass().getName());
                }
            }, () -> {
                throw new IllegalStateException("no use of param");
            });
        });
    }


    public interface WrappedInvokeOpTransformer extends BiFunction<Block.Builder, InvokeOpWrapper, Block.Builder> {
        Block.Builder apply(Block.Builder block, InvokeOpWrapper op);
    }

    public FuncOpWrapper transformInvokes(WrappedInvokeOpTransformer wrappedOpTransformer) {
        return OpWrapper.wrap(lookup,op.transform((b, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                wrappedOpTransformer.apply(b, OpWrapper.wrap(lookup,invokeOp));
            } else {
                b.op(op);
            }
            return b;
        }));
    }

    public static class WrappedOpReplacer<T extends Op, WT extends OpWrapper<T>>{
        final private Block.Builder builder;
        final private CopyContext context;
        final private WT current;
        private boolean replaced = false;
        WrappedOpReplacer(Block.Builder builder, WT current){
            this.builder = builder;
            this.context = this.builder.context();
            this.current = current;
        }
        public List<Value> currentOperandValues(){
            return context.getValues(current.op.operands());
        }

        public void replace(Op replacement){
            context.mapValue(current.op.result(), builder.op(replacement));
            replaced = true;
        }

        public WT current() {
            return  current;
        }

        public TypeElement currentResultType() {
            return current.op.resultType();
        }
    }
    public String functionName() {
        return op.funcName();
    }

    public TypeElement functionReturnTypeDesc() {
        return op.resultType();
    }

}
