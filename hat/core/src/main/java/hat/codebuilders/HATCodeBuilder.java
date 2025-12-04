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
import jdk.incubator.code.dialect.java.JavaType;

import java.util.Arrays;
import java.util.function.Consumer;

public abstract class HATCodeBuilder<T extends HATCodeBuilder<T>> extends CodeBuilder<T> {





    public T oracleCopyright(){
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

    public T suffix_t(String name) {
        return identifier(name).identifier("_t");
    }

    public T suffix_u(String name) {
        return identifier(name).identifier("_u");
    }

    public T suffix_s(String name) {
        return identifier(name).identifier("_s");
    }

    public T suffix_t(Class<?> klass) {
        return suffix_t(klass.getSimpleName());
    }

    public T suffix_u(Class<?> klass) {
        return suffix_u(klass.getSimpleName());
    }

    public T suffix_s(Class<?> klass) {
        return suffix_s(klass.getSimpleName());
    }

    public T intDeclaration(String name) {
        return intType().space().identifier(name);
    }

    public T floatDeclaration(String name) {
        return floatType().space().identifier(name);
    }

    public T doubleDeclaration(String name) {
        return doubleType().space().identifier(name);
    }

    public T longDeclaration(String name) {
        return longType().space().identifier(name);
    }

    public T booleanDeclaration(String name) {
        return booleanType().space().identifier(name);
    }

    public T byteDeclaration(String name) {
        return charType().space().identifier(name);
    }

    public T shortDeclaration(String name) {
        return shortType().space().identifier(name);
    }

    public T structOrUnion(boolean isStruct) {
        return (isStruct ? structKeyword() : union());
    }


    public T typedefKeyword() {
        return keyword("typedef");
    }


    public T structKeyword() {
        return keyword("struct");
    }

    public T union() {
        return keyword("union");
    }


    public T externC() {
        return externKeyword().space().dquote("C");
    }

    public T hashDefineKeyword() {
        return hash().keyword("define");
    }

    public T hashIfdefKeyword() {
        return hash().keyword("ifdef");
    }

    public T hashIfndefKeyword() {
        return hash().keyword("ifndef");
    }

    protected T hashEndif() {
        return hash().keyword("endif").nl();
    }

    protected T hashIfdef(String value) {
        return hashIfdefKeyword().space().constant(value).nl();
    }

    protected T hashIfndef(String value) {
        return hashIfndefKeyword().space().constant(value).nl();
    }

    public T hashIfdef(String value, Consumer<T> consumer) {
        return hashIfdef(value).accept(consumer).hashEndif();
    }

    protected T hashIfndef(String value, Consumer<T> consumer) {
        return hashIfndef(value).accept(consumer).hashEndif();
    }
    public T varName(CoreOp.VarOp varOp) {
        identifier(varOp.varName());
        return self();
    }

    public T varName(HATMemoryOp hatLocalVarOp) {
        identifier(hatLocalVarOp.varName());
        return self();
    }

    public T varName(HATVectorVarOp hatVectorVarOp) {
        identifier(hatVectorVarOp.varName());
        return self();
    }

    public T varName(HATVectorLoadOp vectorLoadOp) {
        identifier(vectorLoadOp.varName());
        return self();
    }

    public T varName(HATVectorStoreView hatVectorStoreView) {
        identifier(hatVectorStoreView.varName());
        return self();
    }

    public T varName(HATVectorBinaryOp hatVectorBinaryOp) {
        identifier(hatVectorBinaryOp.varName());
        return self();
    }

    public T varName(HATVectorVarLoadOp hatVectorVarLoadOp) {
        identifier(hatVectorVarLoadOp.varName());
        return self();
    }

    public T varName(HATF16VarOp hatF16VarOp) {
        identifier(hatF16VarOp.varName());
        return self();
    }

    public T builtin_float2bfloat16() {
        identifier("floatTobfloat16");
        return self();
    }

    public T builtin_bfloat16ToFloat() {
        identifier("bfloat16Tofloat");
        return self();
    }

    public T pragmaKeyword() {
        return keyword("pragma");
    }

    public T includeKeyword() {
        return keyword("include");
    }

    public T hashDefine(String name, String... values) {
        hashDefineKeyword().space().identifier(name);
        for (String value : values) {
            space().constant(value);
        }
        return nl();
    }

    public T hashDefine(String name, Consumer<T> consumer) {
        hashDefineKeyword().space().identifier(name);
        space();
        consumer.accept(self());
        return nl();
    }

    public T pragma(String name, String... values) {
        hash().pragmaKeyword().space().identifier(name);
        for (String value : values) {
            space().constant(value);
        }
        return nl();
    }
    public T includeSys(String... values) {
        for (String value : values) {
            hash().includeKeyword().space().lt().identifier(value).gt().nl();
        }
        return self();
    }
    public T include(String... values) {
        for (String value : values) {
            hash().includeKeyword().space().dquote().identifier(value).dquote().nl();
        }
        return nl();
    }

    public T externKeyword() {
        return keyword("extern");
    }

    protected T camel(String value) {
        return identifier(Character.toString(Character.toLowerCase(value.charAt(0)))).identifier(value.substring(1));
    }

    public T camelJoin(String prefix, String suffix) {
        return camel(prefix).identifier(Character.toString(Character.toUpperCase(suffix.charAt(0)))).identifier(suffix.substring(1));
    }

    public final T unsignedCharType() {
        return typeName("unsigned").space().charType();
    }
    public final T unsignedCharType(String identifier) {
        return unsignedCharType().space().identifier(identifier);
    }
    public final T unsignedCharPtrType() {
        return unsignedCharType().space().asterisk();
    }
    public final T unsignedCharPtrType(String identifier) {
        return unsignedCharPtrType().identifier(identifier);
    }

    public T charTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> charType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedCharTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedCharType(), _ -> identifier(name)));
        return self();
    }

    public T shortTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> shortType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedShortTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedShortType(), _ -> identifier(name)));
        return self();
    }

    public T intTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> intType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedIntTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedIntType(), _ -> identifier(name)));
        return self();
    }

    public T floatTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> floatType(), _ -> identifier(name)));
        return self();
    }

    public T longTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> longType(), _ -> identifier(name)));
        return self();
    }

    public T unsignedLongTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedLongType(), _ -> identifier(name)));
        return self();
    }

    public T doubleTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> doubleType(), _ -> identifier(name)));
        return self();
    }

    private T typedef(Consumer<T> lhs, Consumer<T> rhs) {
        return semicolonNlTerminated(_ -> typedefKeyword().space().accept(lhs).space().accept(rhs));
    }

    public final T unsignedIntType() {
        return typeName("unsigned").space().intType();
    }

    public final T unsignedIntType(String identifier ) {
        return unsignedIntType().space().identifier(identifier);
    }

    public final T unsignedLongType() {
        return typeName("unsigned").space().longType();
    }

    public final T unsignedShortType() {
        return typeName("unsigned").space().shortType();
    }
    public final T unsignedShortType(String identifier) {
        return unsignedShortType().space().identifier(identifier);
    }

    //Unused?
    public T declareVarFromJavaType(JavaType type, String varName) {
        if (type.equals(JavaType.INT)) {
            intDeclaration(varName);
        } else if (type.equals(JavaType.LONG)) {
            longDeclaration(varName);
        } else  if (type.equals(JavaType.FLOAT)) {
            floatDeclaration(varName);
        } else if (type.equals(JavaType.DOUBLE)) {
            doubleDeclaration(varName);
        }
        return self();
    }
    public T funcName(CoreOp.FuncCallOp funcCallOp){
        return identifier(funcCallOp.funcName());
    }
    public T funcName(CoreOp.FuncOp funcOp) {
        return identifier(funcOp.funcName());
    }
    public T fieldName(JavaOp.FieldAccessOp fieldAccessOp) {
        return identifier(OpTk.fieldName(fieldAccessOp));
    }
    public T funcName(JavaOp.InvokeOp invokeOp){
        return identifier(OpTk.funcName(invokeOp));
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

    public T buildStructSingleMember(String structName, String member, String type) {
        typedefKeyword().space().structKeyword().space().suffix_s(structName)
                .obrace().nl()
                .in()
                    .typeName(type).space().typeName(member).semicolon().nl()
                .out().cbrace().space().suffix_t(structName).semicolon().nl();
        return self();
    }

    public T buildForLoopHeader(String loopVar, String init, String loopBound) {
        forKeyword().paren(_ -> intType().space().identifier(loopVar).space().equals().identifier(init).semicolon().space()
                        .identifier(loopVar).lt().identifier(loopBound).semicolon().space()
                        .identifier(loopVar).plusplus());
        return self();
    }

    public T builtin_byteCopy(){
        return identifier("byteCopy");
    }

    /**
     * <code>
     *  void byteCopy(void *dest, const void* src, size_t size) {
     *      unsigned char *c = (unsigned char*)dest;
     *      unsigned char *s = (unsigned char*)src;
     *      for (int i = 0; i < size; i++) {
     *          *c++ = *s++;
     *      }
     *  }
     * </code>
     * @return
     */
    public T build_builtin_byteCopy() {
        voidType().space().builtin_byteCopy()
                .paren(_-> voidPtrType("dest").commaSpace().voidPtrType("src").commaSpace().size_t("size"));
        braceNlIndented(_ ->
                         unsignedCharPtrType("c").equals().paren( _ -> unsignedCharPtrType()).identifier("dest").semicolonNl()
                        .unsignedCharPtrType("s").equals().paren( _ -> unsignedCharPtrType()).identifier("src").semicolonNl()
                        .buildForLoopHeader("i", "0", "size").braceNlIndented(_ ->
                                         dereference("c").plusplus().equals().dereference("s").plusplus().semicolon()
                                 )
        );
        nl();
        return self();
    }

    /**
     * <code>
     *  float bfloat16Tofloat(ushort bf16) {
     *      uint bitsRecovered = bf16 << 16;
     *      float r = bitsRecovered;
     *      byteCopy(&r, &bitsRecovered, sizeof(r));
     *      return r;
     * }
     * </code>
     *
     * @param parameterName
     * @return
     */
    public T build_builtin_bfloat16ToFloat(String parameterName) {
        floatType().space().builtin_bfloat16ToFloat().paren(_ -> unsignedShortType(parameterName))
                .brace( _ ->
                        nl()
                        .unsignedIntType("bits").equals().identifier(parameterName).leftShift(16).semicolonNl()
                        .floatType("r").equals().identifier("bits").semicolonNl()
                        .builtin_byteCopy().paren(_ -> addressOf("r").commaSpace().addressOf("bits").comma().sizeof("r")).semicolonNl()
                        .returnKeyword("r").semicolonNl()
                );
        nl();
        return self();
    }

    /**
     * <code>
     * ushort floatTobfloat16(float f) {
     *      uint bits;
     *      byteCopy(&bits, &f, sizeof(bits));
     *      short bf16 = bits >> 16;
     *      return bf16;
     * }
     * </code>
     * @param parameterName
     * @return
     */
    public T build_builtin_float2bfloat16(String parameterName) {
        shortType().space().builtin_float2bfloat16().paren(_ -> floatType().space().identifier(parameterName))
                .brace( _ -> nl()
                        .unsignedIntType("bits").semicolonNl()
                        .builtin_byteCopy().paren(_ ->
                                addressOf("bits").commaSpace().addressOf(parameterName).commaSpace().sizeof("bits")).semicolonNl()
                                .shortType("bf16").equals().identifier("bits").rightShift(16).semicolonNl()
                                .returnKeyword("bf16").semicolonNl()
                );
        nl();
        return self();
    }
}
