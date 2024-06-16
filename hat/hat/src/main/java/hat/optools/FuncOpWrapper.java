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

import hat.HatOps;
import hat.KernelContext;
import hat.buffer.Buffer;

import java.lang.foreign.GroupLayout;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

public class FuncOpWrapper extends OpWrapper<CoreOp.FuncOp> {
    protected void collectParams(Block block) {

        block.parameters().stream()
                .map(parameter -> {
                    var op = parameter.uses().stream().findFirst().orElseThrow().op();
                    return Map.entry(parameter, (CoreOp.VarOp) op);
                })
                .forEach(paramTable::add);
    }

    public JavaType getReturnType() {
        return (JavaType) op().body().yieldType();
    }

    public static class ParamTable {
        public static class Info {

            public static boolean isIfaceBuffer(Class<?> hopefullyABufferClass) {

                if (Buffer.class.isAssignableFrom(hopefullyABufferClass)) {
                    return true;
                } else {
                    Class<?> enclosingClass = hopefullyABufferClass.getEnclosingClass();
                    if (enclosingClass != null) {
                        return isIfaceBuffer(enclosingClass);
                    } else {
                        return false;
                    }
                }
            }

            public static boolean isIfaceBuffer(JavaType javaType) {
                if (javaType instanceof PrimitiveType) {
                    return false;
                }
                try {
                    String className = javaType.toString();
                    Class<?> hopefullyABufferClass = Class.forName(className);
                    return isIfaceBuffer(hopefullyABufferClass);
                } catch (ClassNotFoundException e) {
                    return false;
                }

            }

            public boolean isIfaceBuffer() {
                return isIfaceBuffer(javaType);

            }

            public boolean isKernelContext() {
                if (javaType instanceof PrimitiveType) {
                    return false;
                }
                try {
                    String className = javaType.toString();
                    Class<?> hopefullyAKernelContext = Class.forName(className);
                    return KernelContext.class.isAssignableFrom(hopefullyAKernelContext);
                } catch (ClassNotFoundException e) {
                    return false;
                }
            }

            public final int idx;
            public final Block.Parameter parameter;

            public final JavaType javaType;
            public final CoreOp.VarOp varOp;
            Value dirtyVar;
            public GroupLayout layout = null;
            public Class<?> clazz = null;

            Info(int idx, Block.Parameter parameter, CoreOp.VarOp varOp) {
                this.idx = idx;
                this.parameter = parameter;
                this.javaType = (JavaType) parameter.type();
                this.varOp = varOp;
                this.dirtyVar = null;
            }

            public void setDirtyVar(Value dirtyVar) {
                this.dirtyVar = dirtyVar;
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
                    parameterToInfo.computeIfAbsent(parameterToVarOp.getKey(), (parameterKey) -> { // always called but convenient because computeIfAbsent returns what we added :)
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

    public static class BiMap<T1 extends Block.Parameter,T2 extends Op>{
        public Map<T1, T2> t1ToT2 = new LinkedHashMap<>();
        public Map<T2,T1> t2ToT1 = new LinkedHashMap<>();
        public void add(T1 t1,T2 t2){
            t1ToT2.put(t1,t2);
            t2ToT1.put(t2,t1);
        }
        public T1 get(T2 t2){
            return t2ToT1.get(t2);
        }
        public T2 get(T1 t1){
            return t1ToT2.get(t1);
        }

        public boolean containsKey(T1 t1) {
            return t1ToT2.containsKey(t1);
        }
        public boolean containsKey(T2 t2) {
            return t2ToT1.containsKey(t2);
        }
    }

    public BiMap<Block.Parameter, CoreOp.VarOp> parameterVarOpMap = new BiMap<>();
    public BiMap<Block.Parameter, CoreOp.InvokeOp> parameterInvokeOpMap = new BiMap<>();
    public BiMap<Block.Parameter, HatOps.HatPtrOp> parameterHatPtrOpMap = new BiMap<>();
    public FuncOpWrapper(CoreOp.FuncOp op) {
        super(op);
        op().parameters().forEach(parameter -> {
            Optional<Op.Result> optionalResult = parameter.uses().stream().findFirst();
            optionalResult.ifPresentOrElse(result -> {
                var resultOp = result.op();
                if (resultOp instanceof CoreOp.VarOp varOp) {
                    parameterVarOpMap.add(parameter, varOp);
                    paramTable.add(Map.entry(parameter, varOp));
                }else if (resultOp instanceof CoreOp.InvokeOp invokeOp) {
                    parameterInvokeOpMap.add(parameter,invokeOp);
                }else if (resultOp instanceof HatOps.HatPtrOp hatPtrOp) {
                    parameterHatPtrOpMap.add(parameter,hatPtrOp);
                }else{
                    //System.out.println("neither varOp or an invokeOp "+resultOp.getClass().getName());
                }
            }, () -> {
                throw new IllegalStateException("no use of param");
            });
        });
    }

    public FuncOpWrapper lower() {
        return OpWrapper.wrap(op().transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        }));
    }

    public FuncOpWrapper ssa() {
        return OpWrapper.wrap(SSA.transform(op()));
    }


    public Stream<OpWrapper<?>> wrappedRootOpStream() {
        return wrappedRootOpStream(firstBlockOfFirstBody());
    }

    public CoreOp.FuncOp transform(String newName, OpTransformer opTransformer) {
        return op().transform(newName, opTransformer);
    }

    public boolean isParameterVarOp(VarDeclarationOpWrapper varDeclarationOpWrapper) {
        return paramTable().isParameterVarOp(varDeclarationOpWrapper.op());
    }

    public boolean isParameterVarOp(CoreOp.VarOp varOp) {
        return paramTable().isParameterVarOp(varOp);
    }



    public interface WrappedInvokeOpTransformer extends BiFunction<Block.Builder, InvokeOpWrapper, Block.Builder> {
        Block.Builder apply(Block.Builder block, InvokeOpWrapper op);
    }

    public FuncOpWrapper transformInvokes(WrappedInvokeOpTransformer wrappedOpTransformer) {
        return OpWrapper.wrap(op().transform((b, op) -> {
            if (op instanceof CoreOp.InvokeOp invokeOp) {
                wrappedOpTransformer.apply(b, OpWrapper.wrap(invokeOp));
            } else {
                b.op(op);
            }
            return b;
        }));
    }

    public FuncOpWrapper transformIfaceInvokes(BiConsumer<Block.Builder,InvokeOpWrapper> wrappedOpTransformer) {
        return OpWrapper.wrap(op().transform((b, op) -> {
            if (op instanceof CoreOp.InvokeOp invokeOp) {
                InvokeOpWrapper wrapped = OpWrapper.wrap(invokeOp);
                if (wrapped.isIfaceBufferMethod()) {
                    wrappedOpTransformer.accept(b,wrapped);
                }else{
                    b.op(op);
                }
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
        public List<Value> operandValues(){
            return context.getValues(current.operands());
        }

        public void replace(Op replacement){
            context.mapValue(current.result(), builder.op(replacement));
            replaced = true;
        }

        public WT current() {
            return  current;
        }

        public TypeElement resultType() {
            return current.resultType();
        }
    }

    public <T extends Op, WT extends OpWrapper<T>>FuncOpWrapper transform(Consumer<WrappedOpReplacer<T,WT>> wrappedOpTransformer) {
        return OpWrapper.wrap(op().transform((b, op) -> {
            var replacer = new WrappedOpReplacer<T,WT>(b, (WT)OpWrapper.wrap(op));
            wrappedOpTransformer.accept(replacer);
            if (!replacer.replaced) {
               b.op(op);
            }
            return b;
        }));
    }

    public  <T extends Op, WT extends OpWrapper<T>>FuncOpWrapper findMapAndReplace(
            Predicate<OpWrapper<?>> predicate,
            Function<OpWrapper<?>, WT> mapper,
            Consumer<WrappedOpReplacer<T,WT>> wrappedOpTransformer) {
        return OpWrapper.wrap(op().transform((b, op) -> {
            var opWrapper = OpWrapper.wrap(op);
            if (predicate.test(opWrapper)) {
                var replacer = new WrappedOpReplacer<T,WT>(b, mapper.apply(opWrapper));
                wrappedOpTransformer.accept(replacer);
                if (!replacer.replaced) {
                    b.op(op);
                }
            }else{
                b.op(op);
            }
            return b;
        }));
    }
    public String functionName() {
        return op().funcName();
    }

    public TypeElement functionReturnTypeDesc() {
        return op().resultType();
    }

}
