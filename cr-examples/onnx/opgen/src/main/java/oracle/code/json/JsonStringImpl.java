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

package oracle.code.json;

import java.util.Objects;

/**
 * JsonString implementation class
 */
final class JsonStringImpl implements JsonString, JsonValueImpl {

    private final JsonDocumentInfo docInfo;
    private final int startOffset;
    private final int endOffset;
    private final int endIndex;
    private String theString;
    private String source;

    JsonStringImpl(String str) {
        docInfo = new JsonDocumentInfo(("\"" + str + "\"").toCharArray());
        startOffset = 0;
        endOffset = docInfo.getEndOffset();
        theString = unescape(startOffset + 1, endOffset - 1);
        endIndex = 0;
    }

    JsonStringImpl(JsonDocumentInfo doc, int offset, int index) {
        docInfo = doc;
        startOffset = offset;
        endIndex = index + 1;
        endOffset = docInfo.getOffset(endIndex) + 1;
    }

    @Override
    public String value() {
        if (theString == null) {
            theString = unescape(startOffset + 1, endOffset - 1);
        }
        return theString;
    }

    @Override
    public int getEndIndex() {
        return endIndex + 1; // We are interested in the index after '"'
    }

    @Override
    public boolean equals(Object o) {
        return this == o ||
            o instanceof JsonStringImpl ojsi &&
            Objects.equals(toString(), ojsi.toString());
    }

    @Override
    public int hashCode() {
        return Objects.hash(toString());
    }

    @Override
    public String toUntyped() {
        return value();
    }

    @Override
    public String toString() {
        if (source == null) {
            source = docInfo.substring(startOffset, endOffset);
        }
        return source;
    }

    String unescape(int startOffset, int endOffset) {
        var sb = new StringBuilder();
        var escape = false;
        int offset = startOffset;
        for (; offset < endOffset; offset++) {
            var c = docInfo.charAt(offset);
            if (escape) {
                switch (c) {
                    case '"', '\\', '/' -> {}
                    case 'b' -> c = '\b';
                    case 'f' -> c = '\f';
                    case 'n' -> c = '\n';
                    case 'r' -> c = '\r';
                    case 't' -> c = '\t';
                    case 'u' -> {
                        c = JsonParser.codeUnit(docInfo, offset + 1);
                        offset += 4;
                    }
                    // TBD: should be replaced with appropriate runtime exception
                    default -> throw new RuntimeException("Illegal escape sequence");
                }
                escape = false;
            } else if (c == '\\') {
                escape = true;
                continue;
            }
            sb.append(c);
        }
        return sb.toString();
    }
}
