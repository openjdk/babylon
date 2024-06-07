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
package hat.backend.c99codebuilders;


import hat.text.CodeBuilder;

import java.util.Arrays;
import java.util.function.Consumer;

public abstract class C99CodeBuilder<T extends C99CodeBuilder<T>> extends CodeBuilder<T> {
    public T suffix_t(String name) {
        return identifier(name).identifier("_t");
    }

    public T suffix_u(String name) {
        return identifier(name).identifier("_u");
    }

    public T suffix_s(String name) {
        return identifier(name).identifier("_s");
    }


    public T intDeclaration(String name) {
        return intType().space().identifier(name);
    }

    public T floatDeclaration(String name) {
        return floatType().space().identifier(name);
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

    T hashDefineKeyword() {
        return hash().keyword("define");
    }

    T hashIfdefKeyword() {
        return hash().keyword("ifdef");
    }

    T hashIfndefKeyword() {
        return hash().keyword("ifndef");
    }

    protected T hashEndif() {
        return hash().keyword("endif").nl();
    }

    T hashIfdef(String value) {
        return hashIfdefKeyword().space().append(value).nl();
    }

    protected T hashIfndef(String value) {
        return hashIfndefKeyword().space().append(value).nl();
    }

    T hashIfdef(String value, Consumer<T> consumer) {
        return hashIfdef(value).accept(consumer).hashEndif();
    }

    protected T hashIfndef(String value, Consumer<T> consumer) {
        return hashIfndef(value).accept(consumer).hashEndif();
    }
  /*  public T defonce(String name, Runnable r) {
        return ifndef(name+"_ONCE_DEF",()->{
            define(name+"_ONCE_DEF").nl();
            r.run();
        });
    }*/

    T pragmaKeyword() {
        return keyword("pragma");
    }

    public T hashDefine(String name, String... values) {
        hashDefineKeyword().space().identifier(name);
        for (String value : values) {
            space().append(value);
        }
        return nl();
    }

    public T pragma(String name, String... values) {
        hash().pragmaKeyword().space().identifier(name);
        for (String value : values) {
            space().append(value);
        }
        return nl();
    }

    T externKeyword() {
        return keyword("extern");
    }

    protected T camel(String value) {
        return identifier(Character.toString(Character.toLowerCase(value.charAt(0)))).identifier(value.substring(1));
    }

    T camelJoin(String prefix, String suffix) {
        return camel(prefix).identifier(Character.toString(Character.toUpperCase(suffix.charAt(0)))).identifier(suffix.substring(1));
    }

    public final T unsignedCharType() {
        return typeName("unsigned").space().charType();
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
        return semicolonTerminatedLine(_ -> typedefKeyword().space().accept(lhs).space().accept(rhs));
    }

    public final T unsignedIntType() {
        return typeName("unsigned").space().intType();
    }

    public final T unsignedLongType() {
        return typeName("unsigned").space().longType();
    }

    public final T unsignedShortType() {
        return typeName("unsigned").space().shortType();
    }


}
