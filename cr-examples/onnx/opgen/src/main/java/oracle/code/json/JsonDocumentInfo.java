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

final class JsonDocumentInfo  {

    // Access to the underlying JSON contents
    final char[] doc;
    // tokens array/index are finalized by JsonParser::parse
    final int[] tokens;
    int index;
    // For exception message on failure
    int line = 0;
    int lineStart = 0;

    JsonDocumentInfo(char[] in) {
        doc = in;
        tokens = new int[doc.length];
        index = 0;
    }

    // Convenience to walk a token during inflation
    boolean shouldWalkToken(char c) {
        return switch (c) {
            case '"', '{', '['  -> true;
            default -> false;
        };
    }

    // gets offset in the input from the array index
    int getOffset(int index) {
        Objects.checkIndex(index, this.index);
        return tokens[index];
    }

    // Json Boolean, Null, and Number have an end index that is 1 greater
    // If the root is a primitive JSON value, -1 is returned as there are no indices
    int nextIndex(int index) {
        if (index + 1 < this.index) {
            return index + 1;
        } else {
            return -1;
        }
    }

    // Json Array and Object have an end index that corresponds to the closing bracket
    int nextIndex(int startIdx, char startToken, char endToken) {
        var index = startIdx + 1;
        int depth = 0;
        while (index < this.index) {
            var c = charAtIndex(index);
            if (c == startToken) {
                depth++;
            } else if (c == endToken) {
                depth--;
            }
            if (depth < 0) {
                break;
            }
            index++;
        }
        return index;
    }

    // for convenience
    char charAtIndex(int index) {
        return doc[getOffset(index)];
    }

    int getIndexCount() {
        return index;
    }

    int getEndOffset() {
        return doc.length;
    }

    // gets the char at the specified offset in the input
    char charAt(int offset) {
        return doc[offset];
    }

    // gets the substring at the specified start/end offsets in the input
    String substring(int startOffset, int endOffset) {
        return new String(doc, startOffset, endOffset - startOffset);
    }

    // Utility method to compose parse exception message
    String composeParseExceptionMessage(String message, int line, int lineStart, int offset) {
        return message + ": (%s) at Row %d, Col %d."
                .formatted(substring(offset, Math.min(offset + 8, doc.length)), line, offset - lineStart);
    }
}
