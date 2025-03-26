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

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Objects;

/**
 * JsonNumber implementation class
 */
final class JsonNumberImpl implements JsonNumber, JsonValueImpl {

    private final JsonDocumentInfo docInfo;
    private final int startOffset;
    private final int endOffset;
    private final int endIndex;
    private Number theNumber;
    private String numString;

    JsonNumberImpl(Number num) {
        theNumber = num;
        numString = num.toString();
        startOffset = 0;
        endOffset = 0;
        endIndex = 0;
        docInfo = null;
    }

    JsonNumberImpl(JsonDocumentInfo doc, int offset, int index) {
        docInfo = doc;
        startOffset = offset;
        endIndex = docInfo.nextIndex(index);
        endOffset = endIndex != -1 ? docInfo.getOffset(endIndex) : docInfo.getEndOffset();
    }

    @Override
    public Number value() {
        if (theNumber == null) {
            theNumber = toNum(string());
        }
        return theNumber;
    }

    private String string() {
        if (numString == null) { // Trim back only
            numString = docInfo.substring(startOffset, endOffset).stripTrailing();
        }
        return numString;
    }

    @Override
    public int getEndIndex() {
        return endIndex;
    }

    @Override
    public boolean equals(Object o) {
        return this == o ||
            o instanceof JsonNumberImpl ojni &&
            Objects.equals(string(), ojni.string());
    }

    @Override
    public int hashCode() {
        return Objects.hash(string());
    }

    Number toNum(String numStr) {
        try {
            return Long.parseLong(numStr);
        } catch (NumberFormatException e) {
        }

        try {
            return new BigInteger(numStr);
        } catch (NumberFormatException e) {
        }

        if (Double.valueOf(numStr) instanceof double d && !Double.isInfinite(d)) {
            return d;
        }

        return new BigDecimal(numStr);

//        // Determine if fp
//        boolean fp = false;
//        for (char c : numStr.toCharArray()) {
//            if (c == 'e' || c == 'E' || c =='.') {
//                fp = true;
//                break;
//            }
//        }
//
//        // Make conversion
//        if (!fp) {
//            // integral numbers
//            try {
//                return Integer.valueOf(numStr);
//            } catch (NumberFormatException _) {
//                // int overflow. try long
//                try {
//                    return Long.valueOf(numStr);
//                } catch (NumberFormatException _) {
//                    // long overflow. convert to Double
//                }
//            }
//        }
//        var num = Double.valueOf(numStr);
//        if (Double.isInfinite(num)) {
//            throw new NumberFormatException("The number is infinitely large in magnitude");
//        }
//        return num;
    }

    @Override
    public Number toUntyped() {
        return value();
    }

    @Override
    public String toString() {
        return string();
    }
}
