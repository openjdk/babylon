/*
 * Copyright (c) 2024-2026, Oracle and/or its affiliates. All rights reserved.
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
package optkl.codebuilders;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.ClassType;
import optkl.exceptions.CodeGenException;

import java.util.List;
import java.util.function.Consumer;

public  class C99CodeBuilder<T extends C99CodeBuilder<T>> extends ScopeAwareJavaOrC99StyleCodeBuilder<T> {

    public C99CodeBuilder(ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(scopedCodeBuilderContext);
    }
    public C99CodeBuilder() {
        super((ScopedCodeBuilderContext) null);
    }
    public C99CodeBuilder(C99CodeBuilder<T> c99CodeBuilder) {
        super(c99CodeBuilder.scopedCodeBuilderContext());
        preformatted(c99CodeBuilder.getText());
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
        return id(name).id("_t");
    }

    public final T suffix_u(String name) {
        return id(name).id("_u");
    }

    public final T suffix_s(String name) {
        return id(name).id("_s");
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
        return externKeyword().sp().dquote("C");
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
        return hashIfdefKeyword().sp().constant(value).nl();
    }

    public final T hashIfndef(String value) {
        return hashIfndefKeyword().sp().constant(value).nl();
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
        hashDefineKeyword().sp().id(name);
        for (String value : values) {
            sp().constant(value);
        }
        return nl();
    }

    public final T hashDefine(String name, Consumer<T> consumer) {
        hashDefineKeyword().sp().id(name);
        sp();
        consumer.accept(self());
        return nl();
    }

    public final T macro(String name, List<String> params, Consumer<T> body) {
        hashDefineKeyword().sp().id(name);
        return paren( _ -> commaSpaceSeparated(params, this::id))
               .sp()
               .paren( _ -> body.accept(self()))
               .nl();
    }

    public final T maxMacro(String name) {
        List<String> params = List.of("a", "b");
        return macro(name, params, _ -> maxMacroBody(params));
    }

    public final T minMacro(String name) {
        List<String> params = List.of("a", "b");
        return macro(name, params, _ -> minMacroBody(params));
    }

    public final T maxMacroBody(List<String> params) {
        final String a = params.getFirst();
        final String b = params.get(1);
        paren(_ -> paren(_ -> id(a))
                .gt()
                .paren(_ -> id(b)));
        questionMark()
                .paren(_ -> id(a))
                .colon()
                .paren(_ -> id(b));
        return self();
    }

    public final T minMacroBody(List<String> params) {
        final String a = params.getFirst();
        final String b = params.get(1);
        paren(_ -> paren( _ -> id(a))
                .lt()
                .paren( _ -> id(b)));
        questionMark()
                .paren( _ -> id(a))
                .colon()
                .paren( _ -> id(b));
        return self();
    }

    public final T pragma(String name, String... values) {
        hash().pragmaKeyword().sp().id(name);
        for (String value : values) {
            sp().constant(value);
        }
        return nl();
    }

    public final T includeSys(String... values) {
        for (String value : values) {
            hash().includeKeyword().sp().lt().id(value).gt().nl();
        }
        return self();
    }

    public final T include(String... values) {
        for (String value : values) {
            hash().includeKeyword().sp().dquote().id(value).dquote().nl();
        }
        return nl();
    }

    public final T namespace(String namespace) {
        return using().sp().namespace().sp().id(namespace).semicolon().nl();
    }

    public final T externKeyword() {
        return keyword("extern");
    }

    public final T u08Type() {
        return type("unsigned").sp().s08Type();
    }

    public final T u08Type(String identifier) {
        return u08Type().sp().id(identifier);
    }

    public final T u08PtrType() {
        return u08Type().sp().asterisk();
    }

    public final T u08PtrType(String identifier) {
        return u08PtrType().id(identifier);
    }

    public final T u32Type() {
        return type("unsigned").sp().s32Type();
    }

    public final T u32Type(String identifier ) {
        return u32Type().sp().id(identifier);
    }

    public final T u64Type() {
        return type("unsigned").sp().s64Type();
    }

    public final T u16Type() {
        return type("unsigned").sp().s16Type();
    }

    public final T u16Type(String identifier) {
        return u16Type().sp().id(identifier);
    }

    public final T bfloat16Type(String identifier) {
        return suffix_t("BFLOAT16_UNION").sp().id(identifier);
    }

    public final  T typedefStructOrUnion(boolean isStruct, Class<?> klass, Consumer<T> consumer) {
        return typedefKeyword()
                .sp()
                .structOrUnion(isStruct)
                .sp()
                .either(isStruct, _ -> suffix_s(klass), _ -> suffix_u(klass))
                .braceNlIndented(consumer)
                .suffix_t(klass).snl();
    }

    public final T typedefStruct(String name, Consumer<T> consumer) {
        return typedefKeyword()
                .sp()
                .structKeyword()
                .sp()
                .suffix_s(name)
                .braceNlIndented(consumer)
                .suffix_t(name)
                .snl();
    }

    public final T typedefUnion(String name, Consumer<T> consumer) {
        return typedefKeyword()
                .sp()
                .union()
                .sp()
                .suffix_s(name)
                .braceNlIndented(consumer)
                .suffix_t(name)
                .snl();
    }

    public final T typedefStruct(Class<?>clazz, Consumer<T> consumer) {
        return typedefStruct(clazz.getSimpleName(), consumer);
    }

    public final T typedefSingleValueStruct(String structName, String type) {
        return typedefStruct(structName,_-> type(type).sp().id("value").semicolon());
    }

    public final T unionBfloat16() {
        return typedefUnion("BFLOAT16_UNION", _ -> {
            type("float").sp().id("f").snl();
            u16Type("s").sbrace( _ -> intConst(2)).snl();
            u32Type("i").semicolon();
        });
    }

    public final T funcDef(Consumer<T> type, Consumer<T> name, Consumer<T> args, Consumer<T> body){
        type.accept(self());
        sp();
        name.accept(self());
        paren(args);
        braceNlIndented(body);
        return nl();
    }


    public final T call(Consumer<T> name,Consumer<T> ...args) {
        name.accept(self());
        return paren(_->commaSpaceSeparated(args));
    }

    public final T ifTrueCondition(Consumer<T> condition, Consumer<T> ... trueStatements) {
        return ifKeyword().sp().paren(_-> condition.accept(self())).sp().brace(_ ->
                nl().indent( _ -> {
                    for (Consumer<T> statement : trueStatements) {
                        statement.accept(self());
                        snl();
                    }
                })).nl();
    }

    public final T call(String name,Consumer<T> ...args) {
        return call(_-> id(name),args);
    }


    public final T sizeof() {
        return emitText("sizeof");
    }

    public final T sizeof(String identifier) {
        return sizeof(_-> id(identifier));
    }

    public final T sizeof(Consumer<T> consumer) {
        return sizeof().paren(consumer);
    }

    public final T voidPtrType() {
        return voidType().sp().asterisk();
    }

    public final T voidPtrType(String identifier) {
        return voidPtrType().id(identifier);
    }

    public final T sizeType() {
        return type("size_t");
    }

    public final T sizeType(String identifier) {
        return sizeType().sp().id(identifier);
    }

}
