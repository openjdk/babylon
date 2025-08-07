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
package hat.tools.textmodel.ui;

import hat.tools.textmodel.BabylonTextModel;
import hat.tools.textmodel.JavaTextModel;
import hat.tools.textmodel.TextModel;
import hat.tools.textmodel.tokens.Arrow;
import hat.tools.textmodel.tokens.At;
import hat.tools.textmodel.tokens.Ch;
import hat.tools.textmodel.tokens.Comment;
import hat.tools.textmodel.tokens.DottedName;
import hat.tools.textmodel.tokens.FloatConst;
import hat.tools.textmodel.tokens.IntConst;
import hat.tools.textmodel.tokens.Nl;
import hat.tools.textmodel.tokens.ReservedWord;
import hat.tools.textmodel.tokens.Seq;
import hat.tools.textmodel.tokens.StringLiteral;
import hat.tools.textmodel.tokens.Ws;

import javax.swing.JTextPane;
import javax.swing.text.Style;
import java.awt.Color;

public abstract class BabylonTextModelViewer extends TextViewer {
    final protected Style comment;
    final protected Style stringLiteral;
    final protected Style number;
    final protected Style operator;
    final protected Style babylonSSARef;
    final protected Style babylonSSADef;
    final protected Style arrow;
    final protected Style modifier;
    final protected Style javaAnnotation;
    final protected Style babylonOp;
    final protected Style dottedName;
    final protected Style type;
    final protected Style babylonBlockOrBody;
    final protected Style babylonAnonymousAttribute;
    final protected Style babylonNamedAttribute;
    final protected Style babylonLocationAttribute;
    final protected Style babylonFileLocationAttribute;
    final protected Style reservedWord;

    BabylonTextModelViewer(TextModel textModel, JTextPane jtextPane, boolean dark) {
        super(textModel,jtextPane,
                dark?new Color(80, 80, 80):new Color(200, 200, 0),
                dark?new Color(255, 255, 255):new Color(0, 0, 0)
        );
        this.textModel = textModel;
        if (dark) {
            jtextPane.setBackground(Color.BLACK);
            this.comment = style( "Comment", new Color(0, 255, 150), false, true, false);
            this.stringLiteral = style( "StringLiteral", new Color(255, 42, 42), false, false, false);
            this.number = style( "Number", new Color(255, 42, 42), false, false, false);
            this.operator = style( "Operator", new Color(120, 120, 120), false, false, false);
            this.babylonSSADef = style( "SSADef", new Color(8, 160, 255), true, true, true);
            this.babylonSSARef = style( "SSARef", new Color(8, 160, 255), false, false, false);
            this.arrow = style( "Arrow", new Color(255, 120, 120), true, false, false);
            this.modifier = style( "Modifier", new Color(200, 125, 0), true, true, false);
            this.javaAnnotation = style( "Annotation", new Color(255, 255, 180), false, true, false);
            this.babylonOp = style( "Op", new Color(2, 210, 10), false, true, false);
            this.dottedName = style( "DottedName", new Color(120, 120, 10), false, true, false);
            this.type = style( "Type", new Color(12, 255, 170), false, true, false);
            this.babylonBlockOrBody = style( "Body", new Color(180, 133, 130), false, true, false);
            this.babylonAnonymousAttribute = style( "AnonymousAttribute", new Color(255, 255, 18), false, true, false);
            this.babylonNamedAttribute = style( "NamedAttribute", new Color(255, 25, 180), false, true, false);
            this.babylonLocationAttribute = style( "LocationAttribute", new Color(200, 200, 200), false, true, false);
            this.babylonFileLocationAttribute = style( "FileLocationAttribute", new Color(255, 255, 255), false, true, false);
            this.reservedWord = style( "ReservedWord", new Color(255, 200, 100), false, true, false);
           } else {
            this.comment = style( "Comment", new Color(0, 100, 0), false, true, false);
            this.stringLiteral = style( "StringLiteral", new Color(100, 10, 100), false, false, false);
            this.number = style( "Number", new Color(255, 42, 42), false, false, false);
            this.babylonOp = style( "Operator", new Color(120, 120, 120), false, false, false);
            this.babylonSSADef = style( "SSADef", new Color(150, 60, 55), true, true, true);
            this.babylonSSARef = style( "SSARef", new Color(150, 60, 255), false, false, false);
            this.arrow = style( "Arrow", new Color(0, 0, 0), true, false, false);
            this.modifier = style( "Modifier", new Color(100, 65, 0), true, true, false);
            this.javaAnnotation = style( "Annotation", new Color(25, 25, 180), false, true, false);
            this.operator = style( "Op", new Color(0, 100, 0), false, true, false);
            this.dottedName = style( "DottedName", new Color(120, 120, 10), false, true, false);
            this.type = style( "Type", new Color(120, 55, 70), true, true, false);
            this.babylonBlockOrBody = style( "Body", new Color(180, 33, 30), false, true, false);
            this.babylonAnonymousAttribute = style( "AnonymousAttribute", new Color(100, 2, 2), false, true, false);
            this.babylonNamedAttribute = style( "NamedAttribute", new Color(100, 100, 2), false, true, false);
            this.babylonLocationAttribute = style( "LocationAttribute", new Color(100, 100, 2), false, true, true);
            this.babylonFileLocationAttribute = style( "FileLocationAttribute", new Color(25, 25, 25), false, true, false);
            this.reservedWord = style( "ReservedWord", new Color(1, 1, 1), false, true, false);
            }
        setTextFromDocModel();
    }

    @Override
    public void applyStyles() {
        textModel.visit(t -> {
                    Style currentStyle = switch (t) {
                        case Ch _, At _ -> operator;
                        case StringLiteral _ -> stringLiteral;
                        case JavaTextModel.JavaAnnotation _ -> javaAnnotation;
                        case ReservedWord _ -> reservedWord;
                        case JavaTextModel.JavaModifier _ -> modifier;
                        case Comment _ -> comment;
                        case Arrow _ -> arrow;
                        case BabylonTextModel.BabylonAnonymousAttribute _ -> babylonAnonymousAttribute;
                        case BabylonTextModel.BabylonFileLocationAttribute _ -> babylonFileLocationAttribute;
                        case BabylonTextModel.BabylonLocationAttribute _ -> babylonLocationAttribute;
                        case BabylonTextModel.BabylonNamedAttribute _ -> babylonNamedAttribute;
                        case JavaTextModel.JavaType _, BabylonTextModel.BabylonTypeAttribute _ -> type;
                        case BabylonTextModel.BabylonSSADef _ -> babylonSSADef;
                        case BabylonTextModel.BabylonSSARef _ -> babylonSSARef;
                        case BabylonTextModel.BabylonOp _ -> babylonOp;
                        case DottedName _ -> dottedName;
                        case BabylonTextModel.BabylonBlockOrBody _ -> babylonBlockOrBody;
                        case Ws _, Nl _, Seq _ -> defaultStyle;
                        case FloatConst _, IntConst _ -> number;
                        default -> defaultStyle;
                    };
                    jTextPane.getStyledDocument().setCharacterAttributes(t.pos().textOffset(), t.len(), currentStyle, true);
                }
        );
    }

    @Override
    public String plainText() {
        return     textModel.plainText();
    }

}
