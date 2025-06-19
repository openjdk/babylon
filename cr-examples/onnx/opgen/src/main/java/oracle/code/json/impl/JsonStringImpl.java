/*
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
 * questions.
 */

package oracle.code.json.impl;

import oracle.code.json.JsonString;

import java.util.Objects;

/**
 * JsonString implementation class
 */
public final class JsonStringImpl implements JsonString {

    private final char[] doc;
    private final int startOffset;
    private final int endOffset;
    private final String str;// = StableSupplier.of(this::unescape);

    public JsonStringImpl(String str) {
        doc = ("\"" + str + "\"").toCharArray();
        startOffset = 0;
        endOffset = doc.length;
        this.str = unescape();
    }

    public JsonStringImpl(char[] doc, int start, int end) {
        this.doc = doc;
        startOffset = start;
        endOffset = end;
        str = unescape();
    }

    @Override
    public String value() {
        var ret = str;
        return str.substring(1, ret.length() - 1);
    }

    @Override
    public String toString() {
        return str;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof JsonString ojs &&
                Objects.equals(value(), ojs.value());
    }

    @Override
    public int hashCode() {
        return Objects.hash(value());
    }

    private String unescape() {
        return Utils.unescape(doc, startOffset, endOffset);
    }
}
