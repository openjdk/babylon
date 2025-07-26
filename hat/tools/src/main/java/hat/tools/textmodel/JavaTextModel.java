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
package hat.tools.textmodel;

import hat.tools.textmodel.tokens.At;
import hat.tools.textmodel.tokens.Ch;
import hat.tools.textmodel.tokens.DottedName;
import hat.tools.textmodel.tokens.FloatConst;
import hat.tools.textmodel.tokens.IntConst;
import hat.tools.textmodel.tokens.LeafReplacementToken;
import hat.tools.textmodel.tokens.ReservedWord;
import hat.tools.textmodel.tokens.Seq;
import hat.tools.textmodel.tokens.Token;

import java.util.regex.Pattern;

public class JavaTextModel extends TextModel {


    public static class JavaModifier extends LeafReplacementToken {
        public static final Pattern regex = Pattern.compile("static|abstract|public|private|protected|final");
        public JavaModifier(Token t) {
            super(t);
        }
    }
    public static class JavaType extends LeafReplacementToken  {
       public static final Pattern regex = Pattern.compile(
                "var|if|while|case|switch|break|for|new|import|instanceof|default|return|super|package"
       );

        public JavaType(Token t) {
            super(t);
        }
    }
    public static class JavaAnnotation extends LeafReplacementToken{
        public JavaAnnotation(Token at, Token identifier) {
            super(at, identifier);
        }
    }

     public void transform(){
         // "[0-9][0-9]*" ->IntConst
         replace(true, t -> Seq.isA(t, $->$.matches(IntConst.regex)), IntConst::new);

         // IntConst '.' IntConst ->FloatConst   (yeah we are missing '.' IntConst  and the exponent stuff)
         replace(true, (t1,t2,t3) -> IntConst.isA(t1) && Ch.isADot(t2) && IntConst.isA(t3), FloatConst::new);

         // @ (char) -> At
         replace(true, Ch::isAnAt, At::new);

         Pattern javaTypes = Pattern.compile("void|int|float|double|boolean|char|short|long|class|record|interface|String");

         replace(true,t -> Seq.isA(t, $->$.matches(javaTypes)), ReservedWord::new);

         replace(true,t -> Seq.isA(t,$->$.matches(JavaType.regex)), JavaType::new);

         replace(true,t -> Seq.isA(t,$->$.matches(JavaModifier.regex)),JavaModifier::new);

         // (Seq|Dname) '.' Seq -> Dname
         replace(true,(t1,t2,t3) -> (Seq.isA(t1) || DottedName.isA(t1)) && Ch.isADot(t2) && Seq.isA(t3),DottedName::new);

         // map all seqs to DottedName
         replace(true, t -> Seq.isA(t,$->$.matches(DottedName.regex)), DottedName::new);
     }
    static public JavaTextModel of(String text) {
        JavaTextModel doc = new JavaTextModel();
        doc.parse(text);
        doc.transform();
        return doc;
    }
}
