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

import java.util.HashSet;

// Responsible for parsing the Json document which validates the contents
// and builds the tokens array in JsonDocumentInfo which is used for lazy inflation
final class JsonParser { ;

    // Parse the JSON and return the built DocumentInfo w/ tokens array
    static JsonDocumentInfo parseRoot(JsonDocumentInfo docInfo) {
        int end = parseValue(docInfo, 0, 0);
        if (!checkWhitespaces(docInfo, end, docInfo.getEndOffset())) {
            throw failure(docInfo,"Unexpected character(s)", end);
        }
        return docInfo;
    }

    static int parseValue(JsonDocumentInfo docInfo, int offset, int depth) {
        offset = skipWhitespaces(docInfo, offset);

        return switch (docInfo.charAt(offset)) {
            case '{' -> parseObject(docInfo, offset, depth + 1);
            case '[' -> parseArray(docInfo, offset, depth + 1);
            case '"' -> parseString(docInfo, offset);
            case 't', 'f' -> parseBoolean(docInfo, offset);
            case 'n' -> parseNull(docInfo, offset);
            case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-' -> parseNumber(docInfo, offset);
            default -> throw failure(docInfo, "Unexpected character(s)", offset);
        };
    }

    static int parseObject(JsonDocumentInfo docInfo, int offset, int depth) {
        checkDepth(docInfo, offset, depth);
        var keys = new HashSet<String>();
        docInfo.tokens[docInfo.index++] = offset;
        // Walk past the '{'
        offset = JsonParser.skipWhitespaces(docInfo, offset + 1);
        // Check for empty case
        if (docInfo.charAt(offset) == '}') {
            docInfo.tokens[docInfo.index++] = offset;
            return ++offset;
        }
        while (offset < docInfo.getEndOffset()) {
            // Get the key
            if (docInfo.charAt(offset) != '"') {
                throw failure(docInfo, "Invalid key", offset);
            }
            // Member equality done via unescaped String
            // see https://datatracker.ietf.org/doc/html/rfc8259#section-8.3
            docInfo.tokens[docInfo.index++] = offset++; // Move past the starting quote
            var escape = false;
            boolean useBldr = false;
            var start = offset;
            StringBuilder sb = null; // only init if we need to use for escapes
            boolean foundClosing = false;
            for (; offset < docInfo.getEndOffset(); offset++) {
                var c = docInfo.charAt(offset);
                if (escape) {
                    var length = 0;
                    switch (c) {
                        // Allowed JSON escapes
                        case '"', '\\', '/' -> {}
                        case 'b' -> c = '\b';
                        case 'f' -> c = '\f';
                        case 'n' -> c = '\n';
                        case 'r' -> c = '\r';
                        case 't' -> c = '\t';
                        case 'u' -> {
                            if (offset + 4 < docInfo.getEndOffset()) {
                                c = codeUnit(docInfo, offset + 1);
                                length = 4;
                            } else {
                                throw failure(docInfo,
                                        "Illegal Unicode escape sequence", offset);
                            }
                        }
                        default -> throw failure(docInfo,
                                "Illegal escape", offset);
                    }
                    if (!useBldr) {
                        useBldr = true;
                        sb = new StringBuilder(docInfo.substring(start, offset - 1));
                    }
                    offset+=length;
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                    continue;
                } else if (c == '\"') {
                    docInfo.tokens[docInfo.index++] = offset++;
                    foundClosing = true;
                    break;
                } else if (c < ' ') {
                    throw failure(docInfo,
                            "Unescaped control code", offset);
                }
                if (useBldr) {
                    sb.append(c);
                }
            }
            if (!foundClosing) {
                throw failure(docInfo, "Closing quote missing", offset);
            }
            var keyStr = useBldr ? sb.toString() :
                    docInfo.substring(start, offset - 1);

            // Check for duplicates
            if (keys.contains(keyStr)) {
                throw failure(docInfo,
                        "The duplicate key: '%s' was already parsed".formatted(keyStr), offset);
            }
            keys.add(keyStr);

            // Move from key to ':'
            offset = JsonParser.skipWhitespaces(docInfo, offset);
            docInfo.tokens[docInfo.index++] = offset;
            if (docInfo.charAt(offset) != ':') {
                throw failure(docInfo,
                        "Unexpected character(s) found after key", offset);
            }

            // Move from ':' to JsonValue
            offset = JsonParser.skipWhitespaces(docInfo, offset + 1);
            offset = JsonParser.parseValue(docInfo, offset, depth);

            // Walk to either ',' or '}'
            offset = JsonParser.skipWhitespaces(docInfo, offset);
            var c = docInfo.charAt(offset);
            if (c == '}') {
                docInfo.tokens[docInfo.index++] = offset;
                return ++offset;
            } else if (docInfo.charAt(offset) != ',') {
                break;
            }

            // Add the comma, and move to the next key
            docInfo.tokens[docInfo.index++] = offset;
            offset = JsonParser.skipWhitespaces(docInfo, offset + 1);
        }
        throw failure(docInfo,
                "Unexpected character(s) found after value", offset);
    }

    static int parseArray(JsonDocumentInfo docInfo, int offset, int depth) {
        checkDepth(docInfo, offset, depth);
        docInfo.tokens[docInfo.index++] = offset;
        // Walk past the '['
        offset = JsonParser.skipWhitespaces(docInfo, offset + 1);
        // Check for empty case
        if (docInfo.charAt(offset) == ']') {
            docInfo.tokens[docInfo.index++] = offset;
            return ++offset;
        }

        while (offset < docInfo.getEndOffset()) {
            // Get the JsonValue
            offset = JsonParser.parseValue(docInfo, offset, depth);
            // Walk to either ',' or ']'
            offset = JsonParser.skipWhitespaces(docInfo, offset);
            var c = docInfo.charAt(offset);
            if (c == ']') {
                docInfo.tokens[docInfo.index++] = offset;
                return ++offset;
            } else if (c != ',') {
                break;
            }

            // Add the comma, and move to the next value
            docInfo.tokens[docInfo.index++] = offset;
            offset = JsonParser.skipWhitespaces(docInfo, offset + 1);
        }
        throw failure(docInfo,
                "Unexpected character(s) found after value", offset);
    }

    static int parseString(JsonDocumentInfo docInfo, int offset) {
        docInfo.tokens[docInfo.index++] = offset++; // Move past the starting quote
        var escape = false;

        for (; offset < docInfo.getEndOffset(); offset++) {
            var c = docInfo.charAt(offset);
            if (escape) {
                switch (c) {
                    // Allowed JSON escapes
                    case '"', '\\', '/', 'b', 'f', 'n', 'r', 't' -> {}
                    case 'u' -> {
                        if (offset + 4 < docInfo.getEndOffset()) {
                            checkEscapeSequence(docInfo, offset + 1);
                            offset += 4;
                        } else {
                            throw failure(docInfo,
                                    "Illegal Unicode escape sequence", offset);
                        }
                    }
                    default -> throw failure(docInfo,
                            "Illegal escape", offset);
                }
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '\"') {
                docInfo.tokens[docInfo.index++] = offset;
                return ++offset;
            } else if (c < ' ') {
                throw failure(docInfo,
                        "Unescaped control code", offset);
            }
        }
        throw failure(docInfo, "Closing quote missing", offset);
    }

    // Validate unicode escape sequence
    static void checkEscapeSequence(JsonDocumentInfo docInfo, int offset) {
        for (int index = 0; index < 4; index++) {
            char c = docInfo.charAt(offset + index);
            if ((c < 'a' || c > 'f') && (c < 'A' || c > 'F') && (c < '0' || c > '9')) {
                throw failure(docInfo, "Invalid Unicode escape", offset);
            }
        }
    }

    // Validate and construct corresponding value of unicode escape sequence
    static char codeUnit(JsonDocumentInfo docInfo, int offset) {
        char val = 0;
        for (int index = 0; index < 4; index ++) {
            char c = docInfo.charAt(offset + index);
            val <<= 4;
            val += (char) (
                    switch (c) {
                        case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' -> c - '0';
                        case 'a', 'b', 'c', 'd', 'e', 'f' -> c - 'a' + 10;
                        case 'A', 'B', 'C', 'D', 'E', 'F' -> c - 'A' + 10;
                        default -> throw new InternalError();
                    });
        }
        return val;
    }

    static int parseBoolean(JsonDocumentInfo docInfo, int offset) {
        var start = docInfo.charAt(offset);
        if (start == 't') {
            if (offset + 3 >= docInfo.getEndOffset() || !docInfo.substring(offset + 1, offset + 4).equals("rue")) {
                throw failure(docInfo, "Unexpected character(s)", offset);
            }
            return offset + 4;
        } else {
            if (offset + 4 >= docInfo.getEndOffset() || !docInfo.substring(offset + 1, offset + 5).equals("alse")) {
                throw failure(docInfo, "Unexpected character(s)", offset);
            }
            return offset + 5;
        }
    }

    static int parseNull(JsonDocumentInfo docInfo, int offset) {
        if (offset + 3 >= docInfo.getEndOffset() || !docInfo.substring(offset + 1, offset + 4).equals("ull")) {
            throw failure(docInfo, "Unexpected character(s)", offset);
        }
        return offset + 4;
    }

    static int parseNumber(JsonDocumentInfo docInfo, int offset) {
        boolean sawDecimal = false;
        boolean sawExponent = false;
        boolean sawZero = false;
        boolean sawWhitespace = false;
        boolean havePart = false;
        boolean sawInvalid = false;
        boolean sawSign = false;
        var start = offset;
        for (; offset < docInfo.getEndOffset() && !sawWhitespace && !sawInvalid; offset++) {
            switch (docInfo.charAt(offset)) {
                case '-' -> {
                    if (offset != start && !sawExponent || sawSign) {
                        throw failure(docInfo,
                                "Invalid '-' position", offset);
                    }
                    sawSign = true;
                }
                case '+' -> {
                    if (!sawExponent || havePart || sawSign) {
                        throw failure(docInfo,
                                "Invalid '+' position", offset);
                    }
                    sawSign = true;
                }
                case '0' -> {
                    if (!havePart) {
                        sawZero = true;
                    }
                    havePart = true;
                }
                case '1', '2', '3', '4', '5', '6', '7', '8', '9' -> {
                    if (!sawDecimal && !sawExponent && sawZero) {
                        throw failure(docInfo,
                                "Invalid '0' position", offset);
                    }
                    havePart = true;
                }
                case '.' -> {
                    if (sawDecimal) {
                        throw failure(docInfo,
                                "Invalid '.' position", offset);
                    } else {
                        if (!havePart) {
                            throw failure(docInfo,
                                    "Invalid '.' position", offset);
                        }
                        sawDecimal = true;
                        havePart = false;
                    }
                }
                case 'e', 'E' -> {
                    if (sawExponent) {
                        throw failure(docInfo,
                                "Invalid '[e|E]' position", offset);
                    } else {
                        if (!havePart) {
                            throw failure(docInfo,
                                    "Invalid '[e|E]' position", offset);
                        }
                        sawExponent = true;
                        havePart = false;
                        sawSign = false;
                    }
                }
                case ' ', '\t', '\r', '\n' -> {
                    sawWhitespace = true;
                    offset --;
                }
                default -> {
                    offset--;
                    sawInvalid = true;
                }
            }
        }
        if (!havePart) {
            throw failure(docInfo,
                    "Input expected after '[.|e|E]'", offset);
        }
        return offset;
    }

    // Utility functions
    static int skipWhitespaces(JsonDocumentInfo docInfo, int offset) {
        while (offset < docInfo.getEndOffset()) {
            if (notWhitespace(docInfo, offset)) {
                break;
            }
            offset ++;
        }
        return offset;
    }

    static boolean checkWhitespaces(JsonDocumentInfo docInfo, int offset, int endOffset) {
        int end = Math.min(endOffset, docInfo.getEndOffset());
        while (offset < end) {
            if (notWhitespace(docInfo, offset)) {
                return false;
            }
            offset ++;
        }
        return true;
    }

    static boolean notWhitespace(JsonDocumentInfo docInfo, int offset) {
        return !isWhitespace(docInfo, offset);
    }

    static boolean isWhitespace(JsonDocumentInfo docInfo, int offset) {
        return switch (docInfo.charAt(offset)) {
            case ' ', '\t','\r' -> true;
            case '\n' -> {
                docInfo.line+=1;
                docInfo.lineStart = offset + 1;
                yield true;
            }
            default -> false;
        };
    }

    static JsonParseException failure(JsonDocumentInfo docInfo, String message, int offset) {
        var errMsg = docInfo.composeParseExceptionMessage(
                message, docInfo.line, docInfo.lineStart, offset);
        return new JsonParseException(errMsg, docInfo.line, offset - docInfo.lineStart);
    }

    private static void checkDepth(JsonDocumentInfo docInfo, int offset, int depth) {
        if (depth > Json.MAX_DEPTH) {
            throw failure(docInfo, "Max depth exceeded", offset);
        }
    }

    // no instantiation of this parser
    private JsonParser(){}
}
