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

package java.lang.reflect.code.parser.impl;

import java.util.ArrayList;
import java.util.List;

/**
 * The lexical analyzer maps an input stream consisting of UTF-8 characters and unicode
 * escape sequences into a token sequence.
 */
final class JavaBasedTokenizer extends UnicodeReader {
    /**
     * If true then prints token information after each nextToken().
     */
    private static final boolean scannerDebug = false;

    /**
     * Sentinel for non-value.
     */
    private static final int NOT_FOUND = -1;

    /**
     * The log to be used for error reporting. Copied from scanner factory.
     */
    private final Log log;

    /**
     * The token factory. Copied from scanner factory.
     */
    private final Tokens tokens;

    /**
     * The token kind, set by nextToken().
     */
    Tokens.TokenKind tk;

    /**
     * The token's radix, set by nextToken().
     */
    int radix;

    /**
     * The token's name, set by nextToken().
     */
    String name;

    /**
     * The position where a lexical error occurred;
     */
    int errPos = Position.NOPOS;

    /**
     * true if contains escape sequences, set by nextToken().
     */
    boolean hasEscapeSequences;

    /**
     * Buffer for building literals, used by nextToken().
     */
    StringBuilder sb;

    /**
     * Construct a Java token scanner from the input character array.
     *
     * @param fac    the factory which created this Scanner
     * @param array  the input character array.
     * @param length The length of the meaningful content in the array.
     */
    JavaBasedTokenizer(Scanner.Factory fac, char[] array, int length) {
        super(fac, array, length);
        this.log = fac.log;
        this.tokens = fac.tokens;
        this.sb = new StringBuilder(256);
    }

    /**
     * Report an error at the given position using the provided arguments.
     *
     * @param pos position in input buffer.
     * @param key error key to report.
     */
    void lexError(int pos, Errors.Error key) {
        tk = Tokens.TokenKind.ERROR;
        errPos = pos;
        log.error(pos, key);
    }

    /**
     * Add a character to the literal buffer.
     *
     * @param ch character to add.
     */
    void put(char ch) {
        sb.append(ch);
    }

    /**
     * Add a codepoint to the literal buffer.
     *
     * @param codePoint codepoint to add.
     */
    void putCodePoint(int codePoint) {
        sb.appendCodePoint(codePoint);
    }

    /**
     * Add current character or codepoint to the literal buffer.
     */
    void put() {
        if (isSurrogate()) {
            putCodePoint(getCodepoint());
        } else {
            put(get());
        }
    }

    /**
     * Add a string to the literal buffer.
     */
    void put(String string) {
        sb.append(string);
    }

    /**
     * Add current character or codepoint to the literal buffer then return next character.
     */
    char putThenNext() {
        put();

        return next();
    }

    /**
     * If the specified character ch matches the current character then add current character
     * to the literal buffer and then advance.
     *
     * @param ch character to match.
     * @return true if ch matches current character.
     */
    boolean acceptThenPut(char ch) {
        if (is(ch)) {
            put(get());
            next();

            return true;
        }

        return false;
    }

    /**
     * If either ch1 or ch2 matches the current character then add current character
     * to the literal buffer and then advance.
     *
     * @param ch1 first character to match.
     * @param ch2 second character to match.
     * @return true if either ch1 or ch2 matches current character.
     */
    boolean acceptOneOfThenPut(char ch1, char ch2) {
        if (isOneOf(ch1, ch2)) {
            put(get());
            next();

            return true;
        }

        return false;
    }

    /**
     * Test if the current character is a line terminator.
     *
     * @return true if current character is a line terminator.
     */
    private boolean isEOLN() {
        return isOneOf('\n', '\r');
    }

    /**
     * Processes the current character and places in the literal buffer. If the current
     * character is a backslash then the next character is validated as a proper
     * escape character. Conversion of escape sequences takes place at end of nextToken().
     *
     * @param pos position of the first character in literal.
     */
    private void scanLitChar(int pos) {
        if (acceptThenPut('\\')) {
            hasEscapeSequences = true;

            switch (get()) {
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                    char leadch = get();
                    putThenNext();

                    if (inRange('0', '7')) {
                        putThenNext();

                        if (leadch <= '3' && inRange('0', '7')) {
                            putThenNext();
                        }
                    }
                    break;

                case 'b':
                case 't':
                case 'n':
                case 'f':
                case 'r':
                case '\'':
                case '\"':
                case '\\':
                    putThenNext();
                    break;

                case '\n':
                case '\r':
                    lexError(position(), Errors.IllegalEscChar);
                    break;

                default:
                    lexError(position(), Errors.IllegalEscChar);
                    break;
            }
        } else {
            putThenNext();
        }
    }

    /**
     * Scan a string literal.
     *
     * @param pos position of the first character in literal.
     */
    private void scanString(int pos) {
        // Assume the best.
        tk = Tokens.TokenKind.STRINGLITERAL;
        // Skip first quote.
        next();

        // While characters are available.
        while (isAvailable()) {
            if (accept('\"')) {
                return;
            }

            if (isEOLN()) {
                // Line terminator in string literal is an error.
                // Fall out to unclosed string literal error.
                break;
            } else {
                // Add character to string buffer.
                scanLitChar(pos);
            }
        }

        lexError(pos, Errors.UnclosedStrLit);
    }

    /**
     * Scan sequence of digits.
     *
     * @param pos        position of the first character in literal.
     * @param digitRadix radix of numeric literal.
     */
    private void scanDigits(int pos, int digitRadix) {
        int leadingUnderscorePos = is('_') ? position() : NOT_FOUND;
        int trailingUnderscorePos;

        do {
            if (!is('_')) {
                put();
                trailingUnderscorePos = NOT_FOUND;
            } else {
                trailingUnderscorePos = position();
            }

            next();
        } while (digit(pos, digitRadix) >= 0 || is('_'));

        if (leadingUnderscorePos != NOT_FOUND) {
            lexError(leadingUnderscorePos, Errors.IllegalUnderscore);
        } else if (trailingUnderscorePos != NOT_FOUND) {
            lexError(trailingUnderscorePos, Errors.IllegalUnderscore);
        }
    }

    /**
     * Read fractional part of hexadecimal floating point number.
     *
     * @param pos position of the first character in literal.
     */
    private void scanHexExponentAndSuffix(int pos) {
        if (acceptOneOfThenPut('p', 'P')) {
            skipIllegalUnderscores();
            acceptOneOfThenPut('+', '-');
            skipIllegalUnderscores();

            if (digit(pos, 10) >= 0) {
                scanDigits(pos, 10);
            } else {
                lexError(pos, Errors.MalformedFpLit);
            }
        } else {
            lexError(pos, Errors.MalformedFpLit);
        }

        if (acceptOneOfThenPut('f', 'F')) {
            tk = Tokens.TokenKind.FLOATLITERAL;
            radix = 16;
        } else {
            acceptOneOfThenPut('d', 'D');
            tk = Tokens.TokenKind.DOUBLELITERAL;
            radix = 16;
        }
    }

    /**
     * Read fractional part of floating point number.
     *
     * @param pos position of the first character in literal.
     */
    private void scanFraction(int pos) {
        skipIllegalUnderscores();

        if (digit(pos, 10) >= 0) {
            scanDigits(pos, 10);
        }

        int index = sb.length();

        if (acceptOneOfThenPut('e', 'E')) {
            skipIllegalUnderscores();
            acceptOneOfThenPut('+', '-');
            skipIllegalUnderscores();

            if (digit(pos, 10) >= 0) {
                scanDigits(pos, 10);
                return;
            }

            lexError(pos, Errors.MalformedFpLit);
            sb.setLength(index);
        }
    }

    /**
     * Read fractional part and 'd' or 'f' suffix of floating point number.
     *
     * @param pos position of the first character in literal.
     */
    private void scanFractionAndSuffix(int pos) {
        radix = 10;
        scanFraction(pos);

        if (acceptOneOfThenPut('f', 'F')) {
            tk = Tokens.TokenKind.FLOATLITERAL;
        } else {
            acceptOneOfThenPut('d', 'D');
            tk = Tokens.TokenKind.DOUBLELITERAL;
        }
    }

    /**
     * Read fractional part and 'd' or 'f' suffix of hexadecimal floating point number.
     *
     * @param pos position of the first character in literal.
     */
    private void scanHexFractionAndSuffix(int pos, boolean seendigit) {
        radix = 16;
        assert is('.');
        putThenNext();
        skipIllegalUnderscores();

        if (digit(pos, 16) >= 0) {
            seendigit = true;
            scanDigits(pos, 16);
        }

        if (!seendigit)
            lexError(pos, Errors.InvalidHexNumber);
        else
            scanHexExponentAndSuffix(pos);
    }

    /**
     * Skip over underscores and report as a error if found.
     */
    private void skipIllegalUnderscores() {
        if (is('_')) {
            lexError(position(), Errors.IllegalUnderscore);
            skip('_');
        }
    }

    /**
     * Read a number. (Spec. 3.10)
     *
     * @param pos   position of the first character in literal.
     * @param radix the radix of the number; one of 2, 8, 10, 16.
     */
    private void scanNumber(int pos, int radix) {
        // for octal, allow base-10 digit in case it's a float literal
        this.radix = radix;
        int digitRadix = (radix == 8 ? 10 : radix);
        int firstDigit = digit(pos, Math.max(10, digitRadix));
        boolean seendigit = firstDigit >= 0;
        boolean seenValidDigit = firstDigit >= 0 && firstDigit < digitRadix;

        if (seendigit) {
            scanDigits(pos, digitRadix);
        }

        if (radix == 16 && is('.')) {
            scanHexFractionAndSuffix(pos, seendigit);
        } else if (seendigit && radix == 16 && isOneOf('p', 'P')) {
            scanHexExponentAndSuffix(pos);
        } else if (digitRadix == 10 && is('.')) {
            putThenNext();
            scanFractionAndSuffix(pos);
        } else if (digitRadix == 10 && isOneOf('e', 'E', 'f', 'F', 'd', 'D')) {
            scanFractionAndSuffix(pos);
        } else {
            if (!seenValidDigit) {
                switch (radix) {
                    case 2:
                        lexError(pos, Errors.InvalidBinaryNumber);
                        break;
                    case 16:
                        lexError(pos, Errors.InvalidHexNumber);
                        break;
                }
            }
            // If it is not a floating point literal,
            // the octal number should be rescanned correctly.
            if (radix == 8) {
                sb.setLength(0);
                reset(pos);
                scanDigits(pos, 8);
            }

            if (acceptOneOf('l', 'L')) {
                tk = Tokens.TokenKind.LONGLITERAL;
            } else {
                tk = Tokens.TokenKind.INTLITERAL;
            }
        }
    }

    /**
     * Determines if the sequence in the literal buffer is a token (keyword, operator.)
     */
    private void checkIdent(Tokens.TokenKind identifier) {
        name = sb.toString();
        tk = tokens.lookupKind(name, identifier);
    }

    /**
     * Read an identifier. (Spec. 3.8)
     */
    private void scanIdent() {
        scanIdent(Tokens.TokenKind.IDENTIFIER);
    }

    /**
     * Read an identifier. (Spec. 3.8)
     */
    private void scanIdent(Tokens.TokenKind identifier) {
        putThenNext();

        do {
            switch (get()) {
                case 'A':
                case 'B':
                case 'C':
                case 'D':
                case 'E':
                case 'F':
                case 'G':
                case 'H':
                case 'I':
                case 'J':
                case 'K':
                case 'L':
                case 'M':
                case 'N':
                case 'O':
                case 'P':
                case 'Q':
                case 'R':
                case 'S':
                case 'T':
                case 'U':
                case 'V':
                case 'W':
                case 'X':
                case 'Y':
                case 'Z':
                case 'a':
                case 'b':
                case 'c':
                case 'd':
                case 'e':
                case 'f':
                case 'g':
                case 'h':
                case 'i':
                case 'j':
                case 'k':
                case 'l':
                case 'm':
                case 'n':
                case 'o':
                case 'p':
                case 'q':
                case 'r':
                case 's':
                case 't':
                case 'u':
                case 'v':
                case 'w':
                case 'x':
                case 'y':
                case 'z':
                case '$':
                case '_':
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    break;

                case '\u0000':
                case '\u0001':
                case '\u0002':
                case '\u0003':
                case '\u0004':
                case '\u0005':
                case '\u0006':
                case '\u0007':
                case '\u0008':
                case '\u000E':
                case '\u000F':
                case '\u0010':
                case '\u0011':
                case '\u0012':
                case '\u0013':
                case '\u0014':
                case '\u0015':
                case '\u0016':
                case '\u0017':
                case '\u0018':
                case '\u0019':
                case '\u001B':
                case '\u007F':
                    next();
                    continue;

                case '\u001A': // EOI is also a legal identifier part
                    if (isAvailable()) {
                        next();
                        continue;
                    }

                    checkIdent(identifier);
                    return;

                default:
                    boolean isJavaIdentifierPart;

                    if (isASCII()) {
                        // all ASCII range chars already handled, above
                        isJavaIdentifierPart = false;
                    } else {
                        if (Character.isIdentifierIgnorable(get())) {
                            next();
                            continue;
                        }

                        isJavaIdentifierPart = isSurrogate()
                                ? Character.isJavaIdentifierPart(getCodepoint())
                                : Character.isJavaIdentifierPart(get());
                    }

                    if (!isJavaIdentifierPart) {
                        checkIdent(identifier);
                        return;
                    }
            }

            putThenNext();
        } while (true);
    }

    /**
     * Read token (main entrypoint.)
     */
    public Tokens.Token readToken() {
        sb.setLength(0);
        name = null;
        radix = 0;
        hasEscapeSequences = false;

        int pos = 0;
        List<Tokens.Comment> comments = null;

        try {
            loop:
            while (true) {
                pos = position();

                switch (get()) {
                    case ' ':  // (Spec 3.6)
                    case '\t': // (Spec 3.6)
                    case '\f': // (Spec 3.6)
                        skipWhitespace();
                        processWhiteSpace(pos, position());
                        break;

                    case '\n': // (Spec 3.4)
                        next();
                        processLineTerminator(pos, position());
                        break;

                    case '\r': // (Spec 3.4)
                        next();
                        accept('\n');
                        processLineTerminator(pos, position());
                        break;

                    case 'A':
                    case 'B':
                    case 'C':
                    case 'D':
                    case 'E':
                    case 'F':
                    case 'G':
                    case 'H':
                    case 'I':
                    case 'J':
                    case 'K':
                    case 'L':
                    case 'M':
                    case 'N':
                    case 'O':
                    case 'P':
                    case 'Q':
                    case 'R':
                    case 'S':
                    case 'T':
                    case 'U':
                    case 'V':
                    case 'W':
                    case 'X':
                    case 'Y':
                    case 'Z':
                    case 'a':
                    case 'b':
                    case 'c':
                    case 'd':
                    case 'e':
                    case 'f':
                    case 'g':
                    case 'h':
                    case 'i':
                    case 'j':
                    case 'k':
                    case 'l':
                    case 'm':
                    case 'n':
                    case 'o':
                    case 'p':
                    case 'q':
                    case 'r':
                    case 's':
                    case 't':
                    case 'u':
                    case 'v':
                    case 'w':
                    case 'x':
                    case 'y':
                    case 'z':
                    case '$':
                    case '_': // (Spec. 3.8)
                        scanIdent();
                        break loop;

                    case '%':
                        scanIdent(Tokens.TokenKind.VALUE_IDENTIFIER);
                        break loop;

                    case '0': // (Spec. 3.10)
                        next();

                        if (acceptOneOf('x', 'X')) {
                            skipIllegalUnderscores();
                            scanNumber(pos, 16);
                        } else if (acceptOneOf('b', 'B')) {
                            skipIllegalUnderscores();
                            scanNumber(pos, 2);
                        } else {
                            put('0');

                            if (is('_')) {
                                int savePos = position();
                                skip('_');

                                if (digit(pos, 10) < 0) {
                                    lexError(savePos, Errors.IllegalUnderscore);
                                }
                            }

                            scanNumber(pos, 8);
                        }
                        break loop;

                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                    case '8':
                    case '9':  // (Spec. 3.10)
                        scanNumber(pos, 10);
                        break loop;

                    case '.': // (Spec. 3.12)
                        next();
                        int savePos = position();

                        if (accept('.')) {
                            lexError(savePos, Errors.IllegalDot);
                        } else if (digit(pos, 10) >= 0) {
                            put('.');
                            scanFractionAndSuffix(pos); // (Spec. 3.10)
                        } else {
                            tk = Tokens.TokenKind.DOT;
                        }
                        break loop;

                    case ',': // (Spec. 3.12)
                        next();
                        tk = Tokens.TokenKind.COMMA;
                        break loop;

                    case '(': // (Spec. 3.12)
                        next();
                        tk = Tokens.TokenKind.LPAREN;
                        break loop;

                    case ')': // (Spec. 3.12)
                        next();
                        tk = Tokens.TokenKind.RPAREN;
                        break loop;

                    case '[': // (Spec. 3.12)
                        next();
                        tk = Tokens.TokenKind.LBRACKET;
                        break loop;

                    case ']': // (Spec. 3.12)
                        next();
                        tk = Tokens.TokenKind.RBRACKET;
                        break loop;

                    case '{': // (Spec. 3.12)
                        next();
                        tk = Tokens.TokenKind.LBRACE;
                        break loop;

                    case '}': // (Spec. 3.12)
                        next();
                        tk = Tokens.TokenKind.RBRACE;
                        break loop;

                    case '?':
                        next();
                        tk = Tokens.TokenKind.QUES;
                        break loop;

                    case ';':
                        next();
                        tk = Tokens.TokenKind.SEMI;
                        break loop;

                    case ':':
                        next();
                        if (accept(':')) {
                            tk = Tokens.TokenKind.COLCOL;
                        } else {
                            tk = Tokens.TokenKind.COLON;
                        }
                        break loop;

                    case '&':
                        next();
                        tk = Tokens.TokenKind.AMP;
                        break loop;

                    case '@':
                        next();
                        tk = Tokens.TokenKind.MONKEYS_AT;
                        break loop;

                    case '^':
                        next();
                        tk = Tokens.TokenKind.CARET;
                        break loop;

                    case '=':
                        next();
                        tk = Tokens.TokenKind.EQ;
                        break loop;

                    case '<':
                        next();
                        tk = Tokens.TokenKind.LT;
                        break loop;

                    case '>':
                        next();
                        tk = Tokens.TokenKind.GT;
                        break loop;

                    case '+':
                        next();
                        tk = Tokens.TokenKind.PLUS;
                        break loop;

                    case '-':
                        next();
                        if (accept('>')) {
                            tk = Tokens.TokenKind.ARROW;
                        } else {
                            tk = Tokens.TokenKind.SUB;
                        }
                        break loop;

                    case '/':
                        next();

                        if (accept('/')) { // (Spec. 3.7)
                            skipToEOLN();

                            if (isAvailable()) {
                                comments = appendComment(comments, processComment(pos, position(), Tokens.Comment.CommentStyle.LINE));
                            }
                            break;
                        } else if (accept('*')) { // (Spec. 3.7)
                            while (isAvailable()) {
                                if (accept('*')) {
                                    if (is('/')) {
                                        break;
                                    }
                                } else {
                                    next();
                                }
                            }

                            if (accept('/')) {
                                comments = appendComment(comments, processComment(pos, position(), Tokens.Comment.CommentStyle.BLOCK));

                                break;
                            } else {
                                lexError(pos, Errors.UnclosedComment);

                                break loop;
                            }
                        } else {
                            lexError(pos, Errors.UnclosedComment);
                        }
                        break loop;

                    case '\'': // (Spec. 3.10)
                        next();

                        if (accept('\'')) {
                            lexError(pos, Errors.EmptyCharLit);
                        } else {
                            if (isEOLN()) {
                                lexError(pos, Errors.IllegalLineEndInCharLit);
                            }

                            scanLitChar(pos);

                            if (accept('\'')) {
                                tk = Tokens.TokenKind.CHARLITERAL;
                            } else {
                                lexError(pos, Errors.UnclosedCharLit);
                            }
                        }
                        break loop;

                    case '\"': // (Spec. 3.10)
                        scanString(pos);
                        break loop;

                    default:
                        boolean isJavaIdentifierStart;

                        if (isASCII()) {
                            // all ASCII range chars already handled, above
                            isJavaIdentifierStart = false;
                        } else {
                            isJavaIdentifierStart = isSurrogate()
                                    ? Character.isJavaIdentifierStart(getCodepoint())
                                    : Character.isJavaIdentifierStart(get());
                        }

                        if (isJavaIdentifierStart) {
                            scanIdent();
                        } else if (digit(pos, 10) >= 0) {
                            scanNumber(pos, 10);
                        } else if (is((char) EOI) || !isAvailable()) {
                            tk = Tokens.TokenKind.EOF;
                            pos = position();
                        } else {
                            String arg;

                            if (isSurrogate()) {
                                int codePoint = getCodepoint();
                                char hi = Character.highSurrogate(codePoint);
                                char lo = Character.lowSurrogate(codePoint);
                                arg = String.format("\\u%04x\\u%04x", (int) hi, (int) lo);
                            } else {
                                char ch = get();
                                arg = (32 < ch && ch < 127) ? String.format("%s", ch) :
                                        String.format("\\u%04x", (int) ch);
                            }

                            lexError(pos, Errors.IllegalChar(arg));
                            next();
                        }
                        break loop;
                }
            }

            int endPos = position();

            if (tk.tag == Tokens.Token.Tag.DEFAULT) {
                return new Tokens.Token(tk, pos, endPos, comments);
            } else if (tk.tag == Tokens.Token.Tag.NAMED) {
                return new Tokens.NamedToken(tk, pos, endPos, name, comments);
            } else {
                // Get characters from string buffer.
                String string = sb.toString();

                // Translate escape sequences if present.
                if (hasEscapeSequences) {
                    try {
                        string = string.translateEscapes();
                    } catch (Exception ex) {
                        // Error already reported, just use untranslated string.
                    }
                }

                if (tk.tag == Tokens.Token.Tag.STRING) {
                    // Build string token.
                    return new Tokens.StringToken(tk, pos, endPos, string, comments);
                } else {
                    // Build numeric token.
                    return new Tokens.NumericToken(tk, pos, endPos, string, radix, comments);
                }
            }
        } finally {
            int endPos = position();

            if (scannerDebug) {
                System.out.println("nextToken(" + pos
                        + "," + endPos + ")=|" +
                        new String(getRawCharacters(pos, endPos))
                        + "| " + tk.name());
            }
        }
    }

    /**
     * Appends a comment to the list of comments preceding the current token.
     *
     * @param comments existing list of comments.
     * @param comment  comment to append.
     * @return new list with comment prepended to the existing list.
     */
    List<Tokens.Comment> appendComment(List<Tokens.Comment> comments, Tokens.Comment comment) {
        if (comments == null) {
            comments = new ArrayList<>();
        }
        // prepend
        comments.add(0, comment);
        return comments;
    }

    /**
     * Return the position where a lexical error occurred.
     *
     * @return position in the input buffer of where the error occurred.
     */
    public int errPos() {
        return errPos;
    }

    /**
     * Set the position where a lexical error occurred.
     *
     * @param pos position in the input buffer of where the error occurred.
     */
    public void errPos(int pos) {
        errPos = pos;
    }

    /**
     * Called when a complete comment has been scanned. pos and endPos
     * will mark the comment boundary.
     *
     * @param pos    position of the opening / in the input buffer.
     * @param endPos position + 1 of the closing / in the input buffer.
     * @param style  style of comment.
     * @return the constructed BasicComment.
     */
    Tokens.Comment processComment(int pos, int endPos, Tokens.Comment.CommentStyle style) {
        if (scannerDebug) {
            System.out.println("processComment(" + pos
                    + "," + endPos + "," + style + ")=|"
                    + new String(getRawCharacters(pos, endPos))
                    + "|");
        }

        char[] buf = getRawCharacters(pos, endPos);
        return new BasicComment(style, new String(buf));
    }

    /**
     * Called when a complete whitespace run has been scanned. pos and endPos
     * will mark the whitespace boundary.
     * <p>
     * (Spec 3.6)
     *
     * @param pos    position in input buffer of first whitespace character.
     * @param endPos position + 1 in input buffer of last whitespace character.
     */
    void processWhiteSpace(int pos, int endPos) {
        if (scannerDebug) {
            System.out.println("processWhitespace(" + pos
                    + "," + endPos + ")=|" +
                    new String(getRawCharacters(pos, endPos))
                    + "|");
        }
    }

    /**
     * Called when a line terminator has been processed.
     *
     * @param pos    position in input buffer of first character in sequence.
     * @param endPos position + 1 in input buffer of last character in sequence.
     */
    void processLineTerminator(int pos, int endPos) {
        if (scannerDebug) {
            System.out.println("processTerminator(" + pos
                    + "," + endPos + ")=|" +
                    new String(getRawCharacters(pos, endPos))
                    + "|");
        }
    }

    /**
     * Build a map for translating between line numbers and positions in the input.
     *
     * @return a LineMap
     */
    public Position.LineMap getLineMap() {
        return Position.makeLineMap(getRawCharacters(), length(), false);
    }

    /**
     * Scan a documentation comment; determine if a deprecated tag is present.
     * Called once the initial /, * have been skipped, positioned at the second *
     * (which is treated as the beginning of the first line).
     * Stops positioned at the closing '/'.
     *
     * @param style Style of comment
     *              LINE starting with //
     *              BLOCK starting with /*
     */
    record BasicComment(Tokens.Comment.CommentStyle style, String text) implements Tokens.Comment {
    }
}
