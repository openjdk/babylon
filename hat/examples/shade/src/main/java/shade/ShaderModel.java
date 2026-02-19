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
package shade;

import optkl.textmodel.TextModel;
import optkl.textmodel.terminal.ANSI;
import optkl.textmodel.tokens.AbstractParentToken;
import optkl.textmodel.tokens.Close;
import optkl.textmodel.tokens.Comment;
import optkl.textmodel.tokens.FloatConst;
import optkl.textmodel.tokens.IntConst;
import optkl.textmodel.tokens.LeafReplacementToken;
import optkl.textmodel.tokens.Open;
import optkl.textmodel.tokens.Parent;
import optkl.textmodel.tokens.Parenthesis;
import optkl.textmodel.tokens.Seq;
import optkl.textmodel.tokens.Char;
import optkl.textmodel.tokens.Token;
import optkl.textmodel.tokens.Ws;
import optkl.util.Regex;
import shade.shaders.WavesShader;

import java.util.ArrayList;
import java.util.List;

import static optkl.textmodel.terminal.ANSI.BLACK;
import static optkl.textmodel.terminal.ANSI.BLUE;
import static optkl.textmodel.terminal.ANSI.CYAN;
import static optkl.textmodel.terminal.ANSI.GREEN;
import static optkl.textmodel.terminal.ANSI.PURPLE;
import static optkl.textmodel.terminal.ANSI.RED;
import static optkl.textmodel.terminal.ANSI.WHITE;
import static optkl.textmodel.terminal.ANSI.YELLOW;

public class ShaderModel extends TextModel {
    public static class Type extends LeafReplacementToken {
        public static final Regex regex = Regex.of("void|int|float|vec[234]|ivec[234]|mat[234]|imat[234]");
        public Type(Token t) {
            super(t);
        }
    }

    public static class Identifier extends LeafReplacementToken {
        public static final Regex regex = Regex.of("[a-zA-Z][a-zA-Z_0-9]*");

        public Identifier(Token t) {
            super(t);
        }
    }

    public static class ReservedWord extends LeafReplacementToken {
        public static final Regex regex = Regex.of("in|out|if|while|case|switch|break|for|new|return");

        public ReservedWord(Token t) {
            super(t);
        }
    }

    public static class Uniform extends LeafReplacementToken {
        public static final Regex regex = Regex.of("iMouse|iResolution|iTime|fragColor|fragCoord");

        public Uniform(Token t) {
            super(t);
        }
    }

    public static class MathFunc extends LeafReplacementToken {
        public static final Regex regex = Regex.of("cross|mul|add|div|sub|cos|sin|tan|atan|exp|abs|dot|sqrt|pow|clamp|min|max|mix|normalize|reflect|normal");

        public MathFunc(Token t) {
            super(t);
        }
    }

    public static class ArithmeticOperator extends LeafReplacementToken {
        public static final Regex regex = Regex.of("[+-/\\*]");

        public ArithmeticOperator(Token t) {
            super(t);
        }
    }
    public static class HashDefine extends LeafReplacementToken {
        public HashDefine(Token hash, Token define) {
            super(hash,define);
        }
    }
    public static class Declaration extends AbstractParentToken {
        static TknPredicate3<Token> predicate = (l, ws, r) -> l instanceof Type && ws instanceof Ws && r instanceof Identifier;
        Type type;
        Identifier identifier;
        public Declaration(Token t, Token ws, Token i ) {
            super(List.of(t,ws,i));
            this.type = (Type) t;
            this.identifier = (Identifier) i;
        }
        @Override
        public String toString() {
            return (type.pos() + ":" + type.asString() + " " + identifier.asString());
        }
    }
    public static class MethodDeclaration extends AbstractParentToken {
        static TknPredicate2<Token> predicate = (l,r) -> l instanceof Declaration && r instanceof Parenthesis;
        Type type;
        Identifier identifier;
        Parenthesis parenthesis;
        public MethodDeclaration(Token declaration, Token parenthesis ) {
            super(List.of(declaration,parenthesis));
            this.type = ((Declaration) declaration).type;
            this.identifier = ((Declaration) declaration).identifier;
            this.parenthesis = (Parenthesis) parenthesis;
        }
        @Override
        public String toString() {
            return (type.pos() + ":" + type.asString() + " " + identifier.asString()+" (...)");
        }
    }




    // Example usage
    public static void main(String[] args) {
        ShaderModel shaderModel = new ShaderModel();
        shaderModel.parse(WavesShader.glslSource);
        shaderModel.replace( (lhs, rhs) ->Char.isA(lhs, $ -> $.is("#")) && Seq.isA(rhs, $->$.is("define")), HashDefine::new);

        shaderModel.replace(t -> Seq.isA(t, $ -> $.matches(FloatConst.regex)), FloatConst::new);  // "[0-9][0-9]*" ->FloatConst
        shaderModel.replace(t -> Seq.isA(t, $ -> $.matches(IntConst.regex)), IntConst::new); // "[0-9][0-9]*" ->IntConst
        shaderModel.replace(t -> Seq.isA(t, $ -> $.matches(Type.regex)), Type::new);
        shaderModel.replace(t -> Seq.isA(t, $ -> $.matches(ReservedWord.regex)), ReservedWord::new);
        shaderModel.replace(t -> Seq.isA(t, $ -> $.matches(Uniform.regex)), Uniform::new);
        shaderModel.replace(t -> Seq.isA(t, $ -> $.matches(MathFunc.regex)), MathFunc::new);
        shaderModel.replace(t -> Char.isA(t, $ -> $.matches(ArithmeticOperator.regex)), ArithmeticOperator::new);
        shaderModel.replace(t -> Seq.isA(t, $ -> $.matches(Identifier.regex)), Identifier::new);

        shaderModel.replace( (t, w, i)-> Declaration.predicate.test(t,w,i), Declaration::new);
        shaderModel.replace( (d, p)-> MethodDeclaration.predicate.test(d,p), MethodDeclaration::new);

      //  List<MethodDeclaration> declarations = new ArrayList<>();
       // shaderModel.find(c->c instanceof MethodDeclaration, (c)->{
         //   declarations.add((MethodDeclaration) c);
        //});
      //  declarations.forEach(System.out::println);

        var c = ANSI.of(System.out);
        shaderModel.visitPostOrder(token -> {
                    if (token instanceof Parent) {
                        switch (token) {
                            case MethodDeclaration _ -> c.apply("/* DECL */");
                            default ->{}
                        }
                    } else {
                        switch (token) {

                            case FloatConst _, IntConst _, Uniform _, ReservedWord _, HashDefine _ -> c.fg(PURPLE, token);
                            case Comment _ -> c.fg(GREEN, _ -> c.bg(BLACK, token));
                            case MathFunc _ -> c.fg(CYAN, token);
                            case Open _, Close _ -> c.fg(BLUE, token);
                            case Type _ -> c.fg(BLUE, token);
                            case ArithmeticOperator _ -> c.fg(RED, token);
                            case Identifier _ -> c.fg(YELLOW, token);
                            default -> c.fg(WHITE, token);
                        }
                    }
                }
        );
    }
}


