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
package hat.text;


import java.util.function.Consumer;

public abstract class JavaCodeBuilder<T extends JavaCodeBuilder<T>> extends CodeBuilder<T> {

    T importKeyword() {
        return keyword("import");
    }

    T classKeyword() {
        return keyword("class");
    }

    public T classKeyword(String name) {
        return classKeyword().space().append(name);
    }

    T extendsKeyword() {
        return keyword("extends");
    }

    public T extendsKeyword(String name) {
        return extendsKeyword().space().append(name);
    }

    public T importKeyword(String name) {
        return importKeyword().space().append(name).semicolon().nl();
    }

    T publicKeyword() {
        return keyword("public");
    }

    T interfaceKeyword() {
        return keyword("interface");
    }

    T publicInterface(String name) {
        return publicKeyword().space().interfaceKeyword().space().append(name);
    }

    public T publicInterface(String name, Consumer<T> consumer) {
        return publicInterface(name).obrace().indent(consumer).cbrace();
    }

    public T publicFinal(String name) {
        return publicKeyword().space().finalKW().space().append(name);
    }

    public T finalKW() {
        return keyword("final");
    }

    T publicKeyword(String name) {
        return publicKeyword().space().append(name);
    }


    public T arity(String arityPattern, int start, int n) {
        ochevron();
        for (int i = start; i < n; i++) {
            if (i > start) {
                comma().space();
            }
            append(arityPattern + i);
        }
        return cchevron();
    }
}
