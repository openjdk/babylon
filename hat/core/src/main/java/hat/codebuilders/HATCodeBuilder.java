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
package hat.codebuilders;


import hat.dialect.*;
import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

public abstract class HATCodeBuilder<T extends HATCodeBuilder<T>> extends CodeBuilder<T> {

    public final T oracleCopyright(){
        return blockComment("""
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
                * questions."""
      );
    }


    //public final T s32Declaration(String name) {
      //  return s32Type(name);
   // }

    public final T varName(CoreOp.VarOp varOp) {
        return identifier(varOp.varName());
    }
    public final  T funcName(CoreOp.FuncCallOp funcCallOp){
        return identifier(funcCallOp.funcName());
    }
    public final T funcName(CoreOp.FuncOp funcOp) {
        return identifier(funcOp.funcName());
    }
    public final T fieldName(JavaOp.FieldAccessOp fieldAccessOp) {
        return identifier(OpTk.fieldName(fieldAccessOp));
    }
    public final T funcName(JavaOp.InvokeOp invokeOp){
        return identifier(OpTk.funcName(invokeOp));
    }

    public final T varName(HATMemoryOp hatLocalVarOp) {
        identifier(hatLocalVarOp.varName());
        return self();
    }

    public final T varName(HATVectorVarOp hatVectorVarOp) {
        identifier(hatVectorVarOp.varName());
        return self();
    }

    public final T varName(HATVectorLoadOp vectorLoadOp) {
        identifier(vectorLoadOp.varName());
        return self();
    }

    public final T varName(HATVectorStoreView hatVectorStoreView) {
        identifier(hatVectorStoreView.varName());
        return self();
    }

    public final T varName(HATVectorBinaryOp hatVectorBinaryOp) {
        identifier(hatVectorBinaryOp.varName());
        return self();
    }

    public final T varName(HATVectorVarLoadOp hatVectorVarLoadOp) {
        identifier(hatVectorVarLoadOp.varName());
        return self();
    }

    public final T varName(HATF16VarOp hatF16VarOp) {
        identifier(hatF16VarOp.varName());
        return self();
    }

    protected final T camel(String value) {
        return identifier(Character.toString(Character.toLowerCase(value.charAt(0)))).identifier(value.substring(1));
    }

    public final T camelJoin(String prefix, String suffix) {
        return camel(prefix).identifier(Character.toString(Character.toUpperCase(suffix.charAt(0)))).identifier(suffix.substring(1));
    }

    T symbol(Op op) {
        return switch (op) {
            case JavaOp.ModOp o -> percent();
            case JavaOp.MulOp o -> mul();
            case JavaOp.DivOp o -> div();
            case JavaOp.AddOp o -> plus();
            case JavaOp.SubOp o -> minus();
            case JavaOp.LtOp o -> lt();
            case JavaOp.GtOp o -> gt();
            case JavaOp.LeOp o -> lte();
            case JavaOp.GeOp o -> gte();
            case JavaOp.AshrOp o -> cchevron().cchevron();
            case JavaOp.LshlOp o -> ochevron().ochevron();
            case JavaOp.LshrOp o -> cchevron().cchevron();
            case JavaOp.NeqOp o -> pling().equals();
            case JavaOp.NegOp o -> minus();
            case JavaOp.EqOp o -> equals().equals();
            case JavaOp.NotOp o -> pling();
            case JavaOp.AndOp o -> ampersand();
            case JavaOp.OrOp o -> bar();
            case JavaOp.XorOp o -> hat();
            case JavaOp.ConditionalAndOp o -> condAnd();
            case JavaOp.ConditionalOrOp o -> condOr();
            default -> throw new IllegalStateException("Unexpected value: " + op);
        };
    }

}
