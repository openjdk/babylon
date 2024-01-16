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

package jdk.code.tools.renderer;

/**
 * Created by gfrost
 */
public class CommonRenderer<T extends CommonRenderer<T>> extends TextRenderer<T> {

    public CommonRenderer() {
    }

    public CommonRenderer(TextRenderer<?> renderer) {
        super(renderer);
    }

    public T semicolon() {
        return op(";");
    }

    public T comma() {
        return op(",");
    }

    public T commaSeparatedList() {
        return startList();
    }

    public T commaSeparator() {
        if (!first()) {
            comma().space();
        }
        return self();
    }

    public T commaSpaceSeparatedList() {
        return startList();
    }

    public T commaSpaceSeparator() {
        if (!first()) {
            comma().space();
        }
        return self();
    }

    public T spaceSeparatedList() {
        return startList();
    }

    public T spaceSeparator() {
        if (!first()) {
            space();
        }
        return self();
    }

    public T newlineSeparatedList() {
        return startList();
    }

    public T newlineSeparator() {
        if (!first()) {
            nl();
        }
        return self();
    }

    public T semicolonSeparatedList() {
        return startList();
    }

    public T semicolonSeparator() {
        if (!first()) {
            semicolon();
        }
        return self();
    }

    public T semicolonSpaceSeparatedList() {
        return startList();
    }

    public T semicolonSpaceSeparator() {
        if (!first()) {
            semicolon().space();
        }
        return self();
    }

    public T dot() {
        return op(".");
    }


    public T equal() {
        return op("=");
    }


    public T plusplus() {
        return op("++");
    }


    public T minusminus() {
        return op("--");
    }

    public T lineComment(String line) {
        return op("//").space().comment(line).nl();
    }


    public T blockComment(String block) {
        return op("/*").nl().comment(block).nl().op("*/").nl();
    }

    public T newKeyword() {
        return keyword("new");
    }


    public T staticKeyword() {
        return keyword("static");
    }


    public T constKeyword() {
        return keyword("const");
    }

    public T ifKeyword() {
        return keyword("if");

    }


    public T whileKeyword() {
        return keyword("while");
    }


    public T breakKeyword() {
        return keyword("break");

    }


    public T continueKeyword() {
        return keyword("continue");
    }


    public T query() {
        return op("?");
    }


    public T colon() {
        return op(":");
    }


    public T nullKeyword() {
        return keyword("null");

    }


    public T elseKeyword() {
        return keyword("else");
    }


    public T returnKeyword() {
        return keyword("return");
    }


    public T switchKeyword() {
        return keyword("switch");
    }


    public T caseKeyword() {
        return keyword("case");
    }


    public T defaultKeyword() {
        return keyword("default");
    }

    public T doKeyword() {
        return keyword("do");
    }

    public T forKeyword() {
        return keyword("for");
    }

    public T ampersand() {
        return op("&");
    }

    public T braced(NestedRendererSAM<T> nb) {
        return nb.build(obrace().nl().in()).out().cbrace().self();
    }

    public T osbrace() {
        return open("[");
    }


    public T csbrace() {
        return close("]");
    }


    public T parenthesized(NestedRendererSAM<T> nb) {
        return nb.build(oparen().in()).out().cparen().self();
    }

    public T underscore() {
        return op("_");
    }

    public T oparen() {
        return open("(");
    }

    public T cparen() {
        return close(")");
    }

    public T obrace() {
        return open("{");
    }

    public T cbrace() {
        return close("}");
    }

    public T at() {
        return op("@").self();
    }

    public T caret() {
        return op("^").self();
    }

    public T percent() {
        return op("%").self();
    }

    public T pipe() {
        return op("|").self();
    }

    public T rarrow() {
        return op("->").self();
    }

    public T larrow() {
        return op("<-").self();
    }

    public T lt() {
        return op("<").self();
    }

    public T gt() {
        return op(">").self();
    }

    public T asterisk() {
        return op("*").self();
    }
}
