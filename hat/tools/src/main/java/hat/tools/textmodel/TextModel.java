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

import hat.tools.textmodel.tokens.Parenthesis;
import hat.tools.textmodel.tokens.Root;
import hat.tools.textmodel.tokens.Token;

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public  class TextModel implements Root {
    List<Token> children = new ArrayList<>();

    @Override
    public List<Token> children() {
        return children;
    }

    final List<Parenthesis.OpenClose> openCloses;
    final Map<Character, Parenthesis.OpenClose> openCloseMap= new HashMap<>();
    final Map<Character, Parenthesis.OpenClose> closeOpenMap= new HashMap<>();
    @Override
    public Parenthesis.OpenClose opensWith(Character ch) {
        return (openCloseMap.containsKey(ch) && openCloseMap.get(ch) instanceof Parenthesis.OpenClose openClose) ? openClose : null;
    }

    @Override
    public Parenthesis.OpenClose closedBy(Character ch) {
        return (closeOpenMap.containsKey(ch) && closeOpenMap.get(ch) instanceof Parenthesis.OpenClose openClose) ? openClose : null;
    }

   protected  TextModel(List<Parenthesis.OpenClose> openCloses) {
        this.openCloses = openCloses;
        for (Parenthesis.OpenClose openClose : openCloses) {
            this.openCloseMap.put(openClose.open(), openClose);
            this.closeOpenMap.put(openClose.close(), openClose);
        }

    }
    protected TextModel() {
        this(List.of(
                Parenthesis.OpenClose.of('(', ')'),
                Parenthesis.OpenClose.of('{', '}'),
                Parenthesis.OpenClose.of('[', ']')
        ));
    }

    public static TextModel of(String text) {
        Cursor cursor = new Cursor(text);
        TextModel doc = new TextModel();
        doc.parse(cursor);
        String plainText = doc.plainText();
        if (!plainText.equals(text)) {
            try {
                var orig =Files.createTempFile("orig", ".txt");
                Files.writeString(orig, text);
                var generated =Files.createTempFile("generated", ".txt");
                Files.writeString(generated, text);
                throw new RuntimeException("Generated Text != original. Check " + orig.toAbsolutePath() + " and " + generated.toAbsolutePath());
            } catch (IOException e) {
                throw new RuntimeException("text != original but unable to generate files" + e);
            }
        }
        return doc;
    }

    public String plainText(){
            var docModelRenderer = new PlainTextRenderer();
            visit(docModelRenderer);
            return docModelRenderer.toString();
    }

    private  static class PlainTextRenderer implements Consumer<Token> {
        StringBuilder stringBuilder = new StringBuilder();
        @Override
        public void accept(Token t) {
            stringBuilder.append(t.asString());
        }
        @Override
        public String toString() {
            return stringBuilder.toString();
        }
    }

}
