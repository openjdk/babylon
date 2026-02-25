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
package shade.parsers;

import optkl.textmodel.terminal.ANSI;
import shade.shaders.WavesShader;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ShaderTokenizer {
    interface Token {
        TOKEN_TYPE tokenType();

        String value();
    }

    interface ParentToken extends Token {
        List<Token> children();
    }

    interface LeafToken extends Token {
    }

    interface TypeToken extends LeafToken {
    }

    interface ConstToken extends Token {
    }

    interface SymbolToken extends Token {
    }

    interface SeparatorToken extends SymbolToken {
    }

    record CreateToken(TOKEN_TYPE tokenType, String value) implements ConstToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record MathLibCallToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record CallToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record VecTypeToken(TOKEN_TYPE tokenType, String value) implements TypeToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record MatTypeToken(TOKEN_TYPE tokenType, String value) implements TypeToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record PrimitiveTypeToken(TOKEN_TYPE tokenType, String value) implements TypeToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record FloatConstToken(TOKEN_TYPE tokenType, String value, float f32) implements ConstToken {
        public String toString() {
            return "F32:" + f32;
        }
    }

    record WSAndLineCommentToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return "WS:" + value.replace("\n", "\\n").replace("\t", "\\t").replace(" ", ".");
        }
    }

    record IntConstToken(TOKEN_TYPE tokenType, String value, int i32) implements ConstToken {
        public String toString() {
            return "S32:" + i32;
        }
    }

    record ReservedToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record UniformToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record IdentifierToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record CommaToken(TOKEN_TYPE tokenType, String value) implements SeparatorToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record SemicolonToken(TOKEN_TYPE tokenType, String value) implements SeparatorToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record OToken(TOKEN_TYPE tokenType, String value) implements SymbolToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record CToken(TOKEN_TYPE tokenType, String value) implements SymbolToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record DotToken(TOKEN_TYPE tokenType, String value) implements SymbolToken {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record PreprocessorToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record AssignToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {
            return tokenType + ":" + value;
        }
    }

    record ArithmeticOperator(TOKEN_TYPE tokenType, String value) implements Token {
        // https://www.codingeek.com/tutorials/c-programming/precedence-and-associativity-of-operators-in-c/
        enum Precedence {
            Multiplicative, Additive
        }

        public Precedence precedence() {
            return switch (value) {
                case "*", "/" -> Precedence.Multiplicative; // multiplacative
                case "-", "+" -> Precedence.Additive; // additive
                default -> throw new IllegalStateException("Unexpected value: " + value);
            };
        }

        public String toString() {
            return tokenType + ":" + value + " " + precedence();
        }
    }


    enum TOKEN_TYPE {
        NONE(null),// NONE looks useless, but the ordinals for these enums define the groups # from regex.  So dont remove ;)
        WS_AND_LINE_COMMENT("([ \n\t]+|//[^\n]*\n)"),
        RESERVED("(in|out|mainImage|return)(?![a-zA-Z0-9_])"),
        UNIFORM("(iTime|iResolution|iMouse)"),
        CREATE("(ivec[234]|vec[234]|imat[234]|mat[234])(?=[({])"),
        MATH_LIB_CALL("(abs|clamp|cos|dot|exp|length|normalize|normal|max|min|mix|pow|reflect|sin|sqrt|tan)(?=[({])"),
        CALL("([a-zA-Z_][a-zA-Z0-9_]*)(?=[({])"),
        PRIMITIVE_TYPE("(float|int|void)(?![a-zA-Z0-9_])"),
        VEC_TYPE("(ivec[234]|vec[234])(?![a-zA-Z0-9_])"),
        MAT_TYPE("(imat[234]|mat[234])(?![a-zA-Z0-9_])"),
        PRE_PREPROCESSOR("(#define|#include)"),
        IDENTIFIER("([a-zA-Z_][a-zA-Z0-9_]*)"),
        CONST("(\\d+\\.?\\d*)"),
        OSYMBOL("([{(])"),
        CSYMBOL("([})])"),
        ASSIGN("(=|\\+=|\\-=|\\*=|/=)"),
        ARITHMETIC_OPERATOR("([+/\\-*])"),
        SEMICOLON("(;)"),
        COMMA("(,)"),
        DOT("(.)");
        String regex;

        TOKEN_TYPE(String regex) {
            this.regex = regex;
        }
    }

    private static List<Token> tokenize(String source) {
        List<Token> tokens = new ArrayList<>();
        // We walk the enum in order and create a single regex for all token types
        // order is important ;)
        StringBuilder regexBuilder = new StringBuilder();
        for (var token : TOKEN_TYPE.values()) {
            if (!token.equals(TOKEN_TYPE.NONE)) {
                if (!regexBuilder.isEmpty()) {
                    regexBuilder.append("|");
                }
                regexBuilder.append(token.regex);
            }
        }

        // Now try to compile the resulting regex
        Pattern pattern = Pattern.compile(regexBuilder.toString());
        Matcher matcher = pattern.matcher(source);
        while (matcher.find()) {
            for (var tokenType : TOKEN_TYPE.values()) {
                if (!tokenType.equals(TOKEN_TYPE.NONE)) {
                    String val = matcher.group(tokenType.ordinal());
                    if (val != null && !val.isEmpty()) {
                        tokens.add(switch (tokenType) {
                            case WS_AND_LINE_COMMENT -> new WSAndLineCommentToken(tokenType, val);
                            case MAT_TYPE -> new MatTypeToken(tokenType, val);
                            case VEC_TYPE -> new VecTypeToken(tokenType, val);
                            case PRIMITIVE_TYPE -> new PrimitiveTypeToken(tokenType, val);
                            case PRE_PREPROCESSOR -> new PreprocessorToken(tokenType, val);
                            case MATH_LIB_CALL -> new MathLibCallToken(tokenType, val);
                            case CALL -> new CallToken(tokenType, val);
                            case CREATE -> new CreateToken(tokenType, val);
                            case UNIFORM -> new UniformToken(tokenType, val);
                            case RESERVED -> new ReservedToken(tokenType, val);
                            case IDENTIFIER -> new IdentifierToken(tokenType, val);
                            case CONST -> val.contains(".")
                                    ? new FloatConstToken(tokenType, val, Float.parseFloat(val))
                                    : new IntConstToken(tokenType, val, Integer.parseInt(val));
                            case OSYMBOL -> new OToken(tokenType, val);
                            case CSYMBOL -> new CToken(tokenType, val);
                            case ARITHMETIC_OPERATOR -> new ArithmeticOperator(tokenType, val);
                            case SEMICOLON -> new SemicolonToken(tokenType, val);
                            case COMMA -> new CommaToken(tokenType, val);
                            case DOT -> new DotToken(tokenType, val);
                            case ASSIGN -> new AssignToken(tokenType, val);
                            default -> throw new IllegalStateException("We should never get here");
                        });
                    }
                }
            }
        }
        return tokens;
    }

    static class Cursor {
        List<Token> tokens;
        int idx;

        Cursor(List<Token> tokens, int idx) {
            this.tokens = tokens;
            this.idx = idx;
        }

        Token get() {
            return tokens.get(idx);
        }

        Token next() {
            idx++;
            while (get() instanceof WSAndLineCommentToken) {
                idx++;
            }
            return get();
        }

        static Cursor of(List<Token> tokens, int idx) {
            return idx < tokens.size() ? new Cursor(tokens, idx) : null;
        }
    }

    // Example parse: print declarations and functions
    private static void parse(List<Token> tokens) {
        for (int i = 0; i < tokens.size(); i++) {
            var c = Cursor.of(tokens, i);
            if (c.get() instanceof TypeToken typeToken && c.next() instanceof CallToken callToken && c.next() instanceof OToken oToken) {
                System.out.println("Function found: returnType=" + typeToken + ", name=" + callToken);
            }
            if (c.get() instanceof ArithmeticOperator arithmeticOperator) {
                System.out.println("Arithmetic " + arithmeticOperator);
            }

        }
    }


    // Example usage
    public static void main(String[] args) {

        List<Token> tokens = tokenize(WavesShader.glslSource);
        var ansi = ANSI.of(System.out);

        tokens.forEach(token -> {
            switch (token) {
                case WSAndLineCommentToken _ -> ansi.color(ANSI.GREEN, a -> a.apply(token.value()));
                case ConstToken _ -> ansi.color(ANSI.YELLOW, a -> a.apply(token.value()));
                case TypeToken _ -> ansi.color(ANSI.BLUE, a -> a.apply(token.value()));
                case ReservedToken _ -> ansi.color(ANSI.CYAN, a -> a.apply(token.value()));
                case PreprocessorToken _ -> ansi.color(ANSI.CYAN, a -> a.apply(token.value()));
                case ArithmeticOperator _ -> ansi.color(ANSI.RED, a -> a.apply(token.value()));
                case AssignToken _ -> ansi.color(ANSI.RED, a -> a.apply(token.value()));
                case UniformToken _ -> ansi.color(ANSI.CYAN, a -> a.apply(token.value()));
                case CallToken _ -> ansi.color(ANSI.PURPLE, a -> a.apply(token.value()));
                case MathLibCallToken _ -> ansi.color(ANSI.BLUE, a -> a.apply(token.value()));
                case SeparatorToken _ -> ansi.color(ANSI.GREEN, a -> a.apply(token.value()));
                default -> ansi.apply(token.value());
            }
        });
        // parse(tokens);
    }

}
