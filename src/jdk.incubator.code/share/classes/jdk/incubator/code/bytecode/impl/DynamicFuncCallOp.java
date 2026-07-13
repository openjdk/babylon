/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.bytecode.impl;

import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import jdk.incubator.code.AbstractOp;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Value;

public final class DynamicFuncCallOp extends AbstractOp {
    private final CodeType resultType;
    private final String funcName;
    private final DirectMethodHandleDesc bootstrapMethod;
    private final String invocationName;
    private final MethodTypeDesc invocationType;
    private final MethodTypeDesc interfaceMethodType;
    private final MethodTypeDesc dynamicMethodType;

    public DynamicFuncCallOp(CodeType resultType,
                             List<Value> operands,
                             String funcName,
                             DirectMethodHandleDesc bootstrapMethod,
                             String invocationName,
                             MethodTypeDesc invocationType,
                             MethodTypeDesc interfaceMethodType,
                             MethodTypeDesc dynamicMethodType) {
        super(operands);
        this.resultType = resultType;
        this.funcName = funcName;
        this.bootstrapMethod = bootstrapMethod;
        this.invocationName = invocationName;
        this.invocationType = invocationType;
        this.interfaceMethodType = interfaceMethodType;
        this.dynamicMethodType = dynamicMethodType;
    }

    DynamicFuncCallOp(DynamicFuncCallOp that, CodeContext cc) {
        super(that, cc);
        this.resultType = that.resultType;
        this.funcName = that.funcName;
        this.bootstrapMethod = that.bootstrapMethod;
        this.invocationName = that.invocationName;
        this.invocationType = that.invocationType;
        this.interfaceMethodType = that.interfaceMethodType;
        this.dynamicMethodType = that.dynamicMethodType;
    }

    @Override
    public DynamicFuncCallOp transform(CodeContext cc, CodeTransformer ct) {
        return new DynamicFuncCallOp(this, cc);
    }

    @Override
    public CodeType resultType() {
        return resultType;
    }

    @Override
    public Map<String, Object> externalize() {
        // for debug print
        LinkedHashMap<String, Object> attributes = new LinkedHashMap<>();
        attributes.put("func", funcName);
        attributes.put("bootstrap", bootstrapMethod);
        attributes.put("invocationName", invocationName);
        attributes.put("invocationType", invocationType);
        attributes.put("interfaceMethodType", interfaceMethodType);
        attributes.put("dynamicMethodType", dynamicMethodType);
        return attributes;
    }

    public String funcName() {
        return funcName;
    }

    public DirectMethodHandleDesc bootstrapMethod() {
        return bootstrapMethod;
    }

    public String invocationName() {
        return invocationName;
    }

    public MethodTypeDesc invocationType() {
        return invocationType;
    }

    public MethodTypeDesc interfaceMethodType() {
        return interfaceMethodType;
    }

    public MethodTypeDesc dynamicMethodType() {
        return dynamicMethodType;
    }
}
