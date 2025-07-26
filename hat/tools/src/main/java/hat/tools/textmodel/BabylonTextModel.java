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

import jdk.incubator.code.dialect.core.CoreOp;
import hat.tools.textmodel.tokens.Arrow;
import hat.tools.textmodel.tokens.At;
import hat.tools.textmodel.tokens.Ch;
import hat.tools.textmodel.tokens.DottedName;
import hat.tools.textmodel.tokens.FloatConst;
import hat.tools.textmodel.tokens.IntConst;
import hat.tools.textmodel.tokens.LeafReplacementToken;
import hat.tools.textmodel.tokens.LineCol;
import hat.tools.textmodel.tokens.Parenthesis;
import hat.tools.textmodel.tokens.ReservedWord;
import hat.tools.textmodel.tokens.Seq;
import hat.tools.textmodel.tokens.StringLiteral;
import hat.tools.textmodel.tokens.Token;
import hat.tools.textmodel.tokens.Ws;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class BabylonTextModel extends TextModel {

    public static class BabylonTypeAttribute extends LeafReplacementToken {
        public BabylonTypeAttribute(Token l, Token m, Token r) {
            super(l, m, r);
        }
    }

    public static class BabylonRefAttribute extends LeafReplacementToken {
        public BabylonRefAttribute(Token l, Token m, Token r) {
            super(l, m, r);
        }
    }

    public static class BabylonNamedAttribute extends LeafReplacementToken {
        public final String name;

        public BabylonNamedAttribute(Token l, Token lm, Token rm, Token r) {
            super(l, lm, rm, r);
            this.name = l.asString();
        }
    }

    public static class BabylonLocationAttribute extends BabylonNamedAttribute implements LineCol {
        static Pattern locPattern = Pattern.compile("\"([0-9]+):([0-9]+)[^\"]*\"");
        public final int line;
        public final int col;

        public BabylonLocationAttribute(Token l, Token lm, Token rm, Token r) {
            super(l, lm, rm, r);
            if (locPattern.matcher(r.asString()) instanceof Matcher m && m.matches() && m.groupCount() > 1) {
                line = Integer.parseInt(m.group(1));
                col = Integer.parseInt(m.group(2));
            } else {
                throw new IllegalArgumentException("invalid location attribute no line/col");
            }
        }

        @Override
        public int line() {
            return line;
        }

        @Override
        public int col() {
            return col;
        }
    }

    public static class BabylonFileLocationAttribute extends BabylonLocationAttribute {
        static Pattern locFilePattern = Pattern.compile("\"([0-9]+):([0-9]+):file:([^\"]*)\"");
        final Path path;

        static Path getPathFromFileLocString(String fileLocString) {
            return locFilePattern.matcher(fileLocString) instanceof Matcher m
                    && m.matches()
                    && m.groupCount() > 2
                    && m.group(3) instanceof String filename
                    && Path.of(filename) instanceof Path javaSource
                    ? javaSource : null;
        }

        public BabylonFileLocationAttribute(Token l, Token lm, Token rm, Token r) {
            super(l, lm, rm, r);
            this.path = getPathFromFileLocString(r.asString());
        }
    }

    public static class BabylonAnonymousAttribute extends LeafReplacementToken {
        public BabylonAnonymousAttribute(Token l, Token r) {
            super(l, r);
        }
    }

    public static class BabylonSSARef extends LeafReplacementToken {
        public final int id;

        public BabylonSSARef(Token t1, Token intConst) {
            super(t1, intConst);
            this.id = ((IntConst) intConst).i;
        }

        public static boolean isA(Token t, Predicate<BabylonSSARef> predicate) {
            return t instanceof BabylonSSARef ssaRef && predicate.test(ssaRef);
        }

        public static boolean isA(Token t) {
            return isA(t, _ -> true);
        }
    }

    public static class BabylonBlock extends LeafReplacementToken {
        static final public Pattern regex = Pattern.compile("block_([0-9]+)");

        public final int id;

        public BabylonBlock(Token t1, Token t2) {
            super(t1, t2);
            if (regex.matcher(t2.asString()) instanceof Matcher m && m.matches() && m.groupCount() == 1) {
                id = Integer.parseInt(m.group(1));
            } else {
                throw new IllegalArgumentException("invalid block attribute no id");
            }
        }

        public static boolean isA(Token t, Predicate<Token> predicate) {
            return t instanceof BabylonBlock && predicate.test(t);
        }

        public static boolean isA(Token t) {
            return isA(t, _ -> true);
        }
    }

    public static class BabylonBlockDef extends LeafReplacementToken {
        public final int id;

        public BabylonBlockDef(Token ref) {
            super(ref);
            this.id = ((BabylonBlock) ref).id;
        }
    }

    public static class BabylonSSADef extends LeafReplacementToken {
        public final int id;

        public BabylonSSADef(Token ssaRef) {
            super(ssaRef);
            this.id = ((BabylonSSARef) ssaRef).id;
        }
    }

    public static class BabylonOp extends LeafReplacementToken {
        public static final Pattern regex = Pattern.compile(
                "(field|var)\\.(store|load)|var|return|yield|continue|invoke|conv|mul|div|add|sub|constant|mod|lt"
        );

        public BabylonOp(Token t) {
            super(t);
        }
    }

    public static class BabylonBlockOrBody extends LeafReplacementToken {
        public static final Pattern regex = Pattern.compile("java\\.(if|while)");

        public BabylonBlockOrBody(Token t) {
            super(t);
        }
    }

    public Path javaSource;
    public JavaTextModel javaTextModel;

    public record SSAEdge(BabylonSSARef ssaRef, BabylonSSADef ssaDef) {
    }

    public record BlockEdge(BabylonBlock ref, BabylonBlockDef def) {
    }

    public List<SSAEdge> ssaEdgeList = new ArrayList<>();
    public List<BlockEdge> blockEdgeList = new ArrayList<>();
    public Map<Integer, BabylonSSADef> idToSSADefMap = new HashMap<>();
    public Map<Integer, BabylonBlockDef> idToBlockDefMap = new HashMap<>();
    public List<BabylonLocationAttribute> babylonLocationAttributes = new ArrayList<>();

    private BabylonTextModel transform() {
        // "[0-9][0-9]*" ->IntConst
        replace(true, t -> Seq.isA(t, $ -> $.matches(IntConst.regex)), IntConst::new);

        // IntConst '.' IntConst ->FloatConst   (yeah we are missing '.' IntConst  and the exponent stuff)
        replace(true, (t1, t2, t3) -> IntConst.isA(t1) && Ch.isADot(t2) && Seq.isA(t3), FloatConst::new);

        // (Seq|Dname) '.' Seq -> Dname
        replace(true, (t1, t2, t3) -> (Seq.isA(t1) || DottedName.isA(t1)) && Ch.isADot(t2) && Seq.isA(t3), DottedName::new);

        // map all seqs to DottedName
        replace(true, t -> Seq.isA(t, $ -> $.matches(DottedName.regex)), DottedName::new);

        Pattern reservedWords = Pattern.compile("(func|Var)");
        // reserved word -> ReservedWord
        replace(true, t -> DottedName.isA(t, $ -> $.matches(reservedWords)), ReservedWord::new);


        // ^block_[0-9]+ -> Block
        replace(true, (t1, t2) -> Ch.isAHat(t1) && DottedName.isA(t2, $ -> $.matches(BabylonBlock.regex)), BabylonBlock::new);

        // ^block_[0-9]+: -> BlockDef
        replace(true, t -> BabylonBlock.isA(t, $ -> $.next(Ch::isAColon)), BabylonBlockDef::new);

        // ^block_[0-9]+() -> Block
        // This is broken just because we have a '(' does not make it a def we also need to check for the colon
        replace(true, t -> BabylonBlock.isA(t, $ -> $.next2((t2,t3)->t2 instanceof Parenthesis && Ch.isAColon(t3))), BabylonBlockDef::new);


        // various opnames -> Op  (I am sure I have missed some)
        replace(true, t -> DottedName.isA(t, $ -> $.matches(BabylonOp.regex)), BabylonOp::new);

        // java.while || java.if -> Body
        replace(true, t -> DottedName.isA(t, $ -> $.matches(BabylonBlockOrBody.regex)), BabylonBlockOrBody::new);

        // '-' + '>' -> ->
        replace(true, (t1, t2) -> Ch.isAHyphen(t1) && Ch.isAGt(t2), Arrow::new);


        // java.type:"MyTypename" -> Type
        replace(true, (t1, t2, t3) ->
                        DottedName.isA(t1, $ -> $.is("java.type")) && Ch.isAColon(t2) && StringLiteral.isA(t3)
                , BabylonTypeAttribute::new
        );

        // java.ref:"MyTypename" -> Type
        replace(true, (t1, t2, t3) ->
                        DottedName.isA(t1, $ -> $.is("java.ref")) && Ch.isAColon(t2) && StringLiteral.isA(t3)
                , BabylonRefAttribute::new
        );

        // %[0-9]+ -> BabylonSSARef
        replace(true, (t1, t2) -> Ch.isAPercent(t1) && IntConst.isA(t2), BabylonSSARef::new);

        // We separate SSARefs and SSADefs
        // SSARef : -> SSADef
        // otherwise we leave as a SSARef
        replace(true, t -> BabylonSSARef.isA(t,
                        $ -> $.next2((t2, t3) -> Ws.isA(t2) && Ch.isAColon(t3)) // this is a lookahead.. t2 and t3 are not replaced
                )
                , BabylonSSADef::new
        );

        // @ (char) -> At
        replace(true, Ch::isAnAt, At::new);

        //  @"foo" -> AnonymousAttribute
        replace(true, (t1, t2) -> At.isA(t1) && StringLiteral.isA(t2), BabylonAnonymousAttribute::new);

        //  @loc="line:col:file.*" -> FileLocationAttribute
        replace(true, (t1, t2, t3, t4) ->
                        At.isA(t1)
                                && DottedName.isA(t2, $ -> $.is("loc"))
                                && Ch.isAnEquals(t3)
                                && StringLiteral.isA(t4, $ -> $.matches(BabylonFileLocationAttribute.locFilePattern))
                , BabylonFileLocationAttribute::new
        );

        //  @loc="line:col:.*" -> LocationAttribute
        replace(true, (t1, t2, t3, t4) ->
                        At.isA(t1)
                                && DottedName.isA(t2, $ -> $.is("loc"))
                                && Ch.isAnEquals(t3)
                                && StringLiteral.isA(t4, $ -> $.matches(BabylonLocationAttribute.locPattern))
                , BabylonLocationAttribute::new
        );
        //  @name=".*" -> LocationAttribute
        replace(true, (t1, t2, t3, t4) ->
                        At.isA(t1) && DottedName.isA(t2) && Ch.isAnEquals(t3) && StringLiteral.isA(t4)
                , BabylonNamedAttribute::new
        );

        visit(t -> {
            if (t instanceof BabylonSSADef def) {
                idToSSADefMap.put(def.id, def);
            } else if (t instanceof BabylonSSARef ref) {
                if (!idToSSADefMap.containsKey(ref.id)) {
                    throw new IllegalArgumentException("Unknown possibly forward BabylonSSARef id " + ref.id);
                }
                var def = idToSSADefMap.get(ref.id);
                ssaEdgeList.add(new SSAEdge(ref, def));
            } else if (t instanceof BabylonLocationAttribute loc) {
                babylonLocationAttributes.add(loc);
            } else if (t instanceof BabylonBlockDef def) {
                idToBlockDefMap.put(def.id, def);
            }
        });
        visit(t -> {
                    if (t instanceof BabylonBlock ref) {
                        if (!idToBlockDefMap.containsKey(ref.id)) {
                            throw new IllegalArgumentException("Unknown possibly forward BabylonBlock id " + ref.id);
                        }
                        var def = idToBlockDefMap.get(ref.id);
                        blockEdgeList.add(new BlockEdge(ref, def));
                    }
                }
        );

        babylonLocationAttributes = babylonLocationAttributes.stream().sorted().collect(Collectors.toList());
        return this;
    }

    static public BabylonTextModel of(String text) {
        BabylonTextModel doc = new BabylonTextModel();
        doc.parse(text);
        doc.find(true, (t) -> t instanceof StringLiteral, (t) -> {
            if (t instanceof StringLiteral sl
                    && sl.matcher(BabylonFileLocationAttribute.locFilePattern) instanceof Matcher m
                    && Path.of(m.group(3)) instanceof Path javaSource && Files.exists(javaSource)
            ) {
                doc.javaSource = javaSource;
                try {
                    doc.javaTextModel = JavaTextModel.of(Files.readString(javaSource));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        if (doc.javaSource == null) {
            throw new IllegalStateException("No source!");
        }
        doc.transform();
        return doc;
    }

    static public BabylonTextModel of(CoreOp.FuncOp javaFunc) {
        var crDoc = of(javaFunc.toText());
        return crDoc;
    }
}
