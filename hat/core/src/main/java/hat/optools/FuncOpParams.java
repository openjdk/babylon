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

import hat.util.BiMap;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.lang.foreign.GroupLayout;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

public class FuncOpParams {
    public static class Info {
        public final int idx;
        public final Block.Parameter parameter;
        public final JavaType javaType;
        public final CoreOp.VarOp varOp;
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

        public void setClass(Class<?> clazz) {
            this.clazz = clazz;
        }
    }

    final public BiMap<Block.Parameter, CoreOp.VarOp> parameterVarOpMap = new BiMap<>();
    final public BiMap<Block.Parameter, JavaOp.InvokeOp> parameterInvokeOpMap = new BiMap<>();

    final private Map<Block.Parameter, Info> parameterToInfo = new LinkedHashMap<>();
    final private Map<CoreOp.VarOp, Info> varOpToInfo = new LinkedHashMap<>();

    final private List<Info> list = new ArrayList<>();

    public Info info(Block.Parameter parameter) {
        return parameterToInfo.get(parameter);
    }

    void add(Map.Entry<Block.Parameter, CoreOp.VarOp> parameterToVarOp) {
        //We add a new ParameterInfo to both maps using parameter and varOp as keys
        varOpToInfo.put(parameterToVarOp.getValue(),
                // always called but convenient because computeIfAbsent returns what we added :)
                parameterToInfo.computeIfAbsent(parameterToVarOp.getKey(), (parameterKey) -> {
                    var info = new Info(list.size(), parameterKey, parameterToVarOp.getValue());
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

    final public CoreOp funcOp;

    public FuncOpParams(CoreOp.FuncOp funcOp) {
        this.funcOp = funcOp;
        funcOp.parameters().forEach(parameter -> {
            Optional<Op.Result> optionalResult = parameter.uses().stream().findFirst();
            optionalResult.ifPresentOrElse(result -> {
                if (result.op() instanceof CoreOp.VarOp varOp) {
                    parameterVarOpMap.add(parameter, varOp);
                    add(Map.entry(parameter, varOp));
                } else if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                    parameterInvokeOpMap.add(parameter, invokeOp);
                }
            }, () -> {
                throw new IllegalStateException("FuncOp has unused params ");
            });
        });
    }
}
