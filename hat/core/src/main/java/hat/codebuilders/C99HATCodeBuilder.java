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
import jdk.incubator.code.dialect.java.ClassType;

import java.util.Arrays;
import java.util.function.Consumer;

public  class C99HATCodeBuilder<T extends C99HATCodeBuilder<T>> extends HATCodeBuilder<T> {

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

    public final T unsignedCharType() {
        return typeName("unsigned").space().charType();
    }
    public final T unsignedCharType(String identifier) {
        return unsignedCharType().space().identifier(identifier);
    }
    public final T u08PtrType() {
        return unsignedCharType().space().asterisk();
    }
    public final T u08PtrType(String identifier) {
        return u08PtrType().identifier(identifier);
    }

    public final T charTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> charType(), _ -> identifier(name)));
        return self();
    }

    public final T unsignedCharTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedCharType(), _ -> identifier(name)));
        return self();
    }

    public final T shortTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> shortType(), _ -> identifier(name)));
        return self();
    }

    public final T unsignedShortTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedShortType(), _ -> identifier(name)));
        return self();
    }

    public final T intTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> intType(), _ -> identifier(name)));
        return self();
    }

    public final T unsignedIntTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> u32Type(), _ -> identifier(name)));
        return self();
    }

    public final T floatTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> f32Type(), _ -> identifier(name)));
        return self();
    }

    public final T longTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> longType(), _ -> identifier(name)));
        return self();
    }

    public final T unsignedLongTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> unsignedLongType(), _ -> identifier(name)));
        return self();
    }

    public final T doubleTypeDefs(String... names) {
        Arrays.stream(names).forEach(name -> typedef(_ -> doubleType(), _ -> identifier(name)));
        return self();
    }

     public final  T typedef(Consumer<T> lhs, Consumer<T> rhs) {
        return semicolonNlTerminated(_ -> typedefKeyword().space().accept(lhs).space().accept(rhs));
    }

    public final T u32Type() {
        return typeName("unsigned").space().intType();
    }

    public final T u32Type(String identifier ) {
        return u32Type().space().identifier(identifier);
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
}
