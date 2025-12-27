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
import hat.dialect.HATF16Op;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATVectorOp;
import jdk.incubator.code.dialect.java.ClassType;

import java.util.function.Consumer;

public  class C99HATCodeBuilder<T extends C99HATCodeBuilder<T>> extends HATCodeBuilder<T> {

    public final T varName(HATMemoryVarOp hatLocalVarOp) {
        identifier(hatLocalVarOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorVarOp hatVectorVarOp) {
        identifier(hatVectorVarOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorLoadOp vectorLoadOp) {
        identifier(vectorLoadOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorStoreView hatVectorStoreView) {
        identifier(hatVectorStoreView.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {
        identifier(hatVectorBinaryOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorVarLoadOp hatVectorVarLoadOp) {
        identifier(hatVectorVarLoadOp.varName());
        return self();
    }

    public final T varName(HATF16Op.HATF16VarOp hatF16VarOp) {
        identifier(hatF16VarOp.varName());
        return self();
    }

    public final T suffix_t(ClassType type){
        String name = type.toClassName();
        int dotIdx = name.lastIndexOf('.');
        int dollarIdx = name.lastIndexOf('$');
        int idx = Math.max(dotIdx, dollarIdx);
        if (idx > 0) {
            name = name.substring(idx + 1);
        }
        return suffix_t(name);
    }

    public final T suffix_t(String name) {
        return identifier(name).identifier("_t");
    }

    public final T suffix_u(String name) {
        return identifier(name).identifier("_u");
    }

    public final T suffix_s(String name) {
        return identifier(name).identifier("_s");
    }

    public final T suffix_t(Class<?> klass) {
        return suffix_t(klass.getSimpleName());
    }

    public final T suffix_u(Class<?> klass) {
        return suffix_u(klass.getSimpleName());
    }

    public final T suffix_s(Class<?> klass) {
        return suffix_s(klass.getSimpleName());
    }

    public final T structOrUnion(boolean isStruct) {
        return (isStruct ? structKeyword() : union());
    }

    public final T typedefKeyword() {
        return keyword("typedef");
    }

    public final T structKeyword() {
        return keyword("struct");
    }

    public final T union() {
        return keyword("union");
    }

    public final T externC() {
        return externKeyword().space().dquote("C");
    }

    public final T hashDefineKeyword() {
        return hash().keyword("define");
    }

    public final T hashIfdefKeyword() {
        return hash().keyword("ifdef");
    }

    public final T hashIfndefKeyword() {
        return hash().keyword("ifndef");
    }

    public final T hashEndif() {
        return hash().keyword("endif").nl();
    }

    public final T hashIfdef(String value) {
        return hashIfdefKeyword().space().constant(value).nl();
    }

    public final T hashIfndef(String value) {
        return hashIfndefKeyword().space().constant(value).nl();
    }

    public final T hashIfdef(String value, Consumer<T> consumer) {
        return hashIfdef(value).accept(consumer).hashEndif();
    }

    public final T hashIfndef(String value, Consumer<T> consumer) {
        return hashIfndef(value).accept(consumer).hashEndif();
    }

    public final T pragmaKeyword() {
        return keyword("pragma");
    }

    public final T includeKeyword() {
        return keyword("include");
    }

    public final T hashDefine(String name, String... values) {
        hashDefineKeyword().space().identifier(name);
        for (String value : values) {
            space().constant(value);
        }
        return nl();
    }

    public final T hashDefine(String name, Consumer<T> consumer) {
        hashDefineKeyword().space().identifier(name);
        space();
        consumer.accept(self());
        return nl();
    }

    public final T pragma(String name, String... values) {
        hash().pragmaKeyword().space().identifier(name);
        for (String value : values) {
            space().constant(value);
        }
        return nl();
    }

    public final T includeSys(String... values) {
        for (String value : values) {
            hash().includeKeyword().space().lt().identifier(value).gt().nl();
        }
        return self();
    }

    public final T include(String... values) {
        for (String value : values) {
            hash().includeKeyword().space().dquote().identifier(value).dquote().nl();
        }
        return nl();
    }

    public final T externKeyword() {
        return keyword("extern");
    }

    public final T u08Type() {
        return typeName("unsigned").space().s08Type();
    }

    public final T u08Type(String identifier) {
        return u08Type().space().identifier(identifier);
    }

    public final T u08PtrType() {
        return u08Type().space().asterisk();
    }

    public final T u08PtrType(String identifier) {
        return u08PtrType().identifier(identifier);
    }

    public final T u32Type() {
        return typeName("unsigned").space().s32Type();
    }

    public final T u32Type(String identifier ) {
        return u32Type().space().identifier(identifier);
    }

    public final T u64Type() {
        return typeName("unsigned").space().s64Type();
    }

    public final T u16Type() {
        return typeName("unsigned").space().s16Type();
    }

    public final T u16Type(String identifier) {
        return u16Type().space().identifier(identifier);
    }

    public final T bfloat16Type(String identifier) {
        return suffix_t("BFLOAT16_UNION").space().identifier(identifier);
    }

    public final  T typedefStructOrUnion(boolean isStruct, Class<?> klass, Consumer<T> consumer) {
        return typedefKeyword()
                .space()
                .structOrUnion(isStruct)
                .space()
                .either(isStruct, _ -> suffix_s(klass), _ -> suffix_u(klass))
                .braceNlIndented(consumer)
                .suffix_t(klass).semicolonNl();
    }

    public final T typedefStruct(String name, Consumer<T> consumer) {
        return typedefKeyword()
                .space()
                .structKeyword()
                .space()
                .suffix_s(name)
                .braceNlIndented(consumer)
                .suffix_t(name)
                .semicolonNl();
    }

    public final T typedefUnion(String name, Consumer<T> consumer) {
        return typedefKeyword()
                .space()
                .union()
                .space()
                .suffix_s(name)
                .braceNlIndented(consumer)
                .suffix_t(name)
                .semicolonNl();
    }

    public final T typedefStruct(Class<?>clazz, Consumer<T> consumer) {
        return typedefStruct(clazz.getSimpleName(), consumer);
    }

    public final T typedefSingleValueStruct(String structName, String type) {
        return typedefStruct(structName,_-> typeName(type).space().identifier("value").semicolon());
    }

    public final T unionBfloat16() {
        return typedefUnion("BFLOAT16_UNION", _ -> {
            typeName("float").space().identifier("f").semicolon().nl();
            u16Type("s").sizeArray(2).semicolon();
        });
    }

    public final T funcDef(Consumer<T> type, Consumer<T> name, Consumer<T> args, Consumer<T> body){
        type.accept(self());
        space();
        name.accept(self());
        paren(args);
        braceNlIndented(body);
        return nl();
    }

    public final T assign(Consumer<T> lhs, Consumer<T> rhs){
        lhs.accept(self());
        space().equals().space();
        rhs.accept(self());
        return self();
    }

    public final T cast(Consumer<T> type){
        return paren(_-> type.accept(self()));
    }

    public final T returnKeyword(Consumer<T> exp){
        return returnKeyword().space().paren(_-> exp.accept(self())).semicolon();
    }

    public final T call(Consumer<T> name,Consumer<T> ...args) {
        name.accept(self());
        return paren(_->commaSpaceSeparated(args));
    }

    public final T call(String name,Consumer<T> ...args) {
        return call(_->identifier(name),args);
    }

    public final T forLoop(Consumer<T> init, Consumer<T> test, Consumer<T>mutate, Consumer<T>body) {
        return  forKeyword()
                .paren(_->{
                    init.accept(self());
                    semicolon().space();
                    test.accept(self());
                    semicolon().space();mutate.accept(self());
                })
                .braceNlIndented(body::accept);
    }

    public final T sizeof() {
        return emitText("sizeof");
    }

    public final T sizeof(String identifier) {
        return sizeof(_->identifier(identifier));
    }

    public final T sizeof(Consumer<T> consumer) {
        return sizeof().paren(consumer);
    }

    public final T voidPtrType() {
        return voidType().space().asterisk();
    }

    public final T voidPtrType(String identifier) {
        return voidPtrType().identifier(identifier);
    }

    public final T sizeType() {
        return typeName("size_t");
    }

    public final T sizeType(String identifier) {
        return sizeType().space().identifier(identifier);
    }
}
