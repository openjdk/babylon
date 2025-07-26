package hat.tools.textmodel.terminal;

import hat.KernelContext;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface;
import hat.tools.textmodel.BabylonTextModel;
import hat.tools.textmodel.JavaTextModel;
import hat.tools.textmodel.tokens.At;
import hat.tools.textmodel.tokens.Close;
import hat.tools.textmodel.tokens.DottedName;
import hat.tools.textmodel.tokens.Nl;
import hat.tools.textmodel.tokens.Parenthesis;
import hat.tools.textmodel.tokens.ReservedWord;
import hat.tools.textmodel.tokens.Root;
import hat.tools.textmodel.tokens.Seq;
import hat.tools.textmodel.tokens.StringLiteral;
import hat.tools.textmodel.tokens.Ws;
import jdk.incubator.code.Op;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static hat.tools.textmodel.terminal.ANSI.*;

public class CodeModelFormatter {
    static void main(String[] argArr) throws NoSuchMethodException, IOException {
        var args = new ArrayList<>(List.of(argArr));
        var doc = BabylonTextModel.of(Files.readString(Path.of(args.getFirst())));
        // We expect the babylonDocModel to have have a javaTextModel embedded
        // We use this to map from babylon lines containing java loc info  -> java lines #

        Map<Integer, Integer> babylonLineToJavaLine = new HashMap<>();
        doc.babylonLocationAttributes.stream().forEach(loc -> {
            babylonLineToJavaLine.put(loc.pos().line(), loc.line());
        });
        var ansi = ANSI.of(System.out);
        ansi.apply(String.format("%4d :", 1));
        int[] online = new int[]{-1};

        doc.visit(t -> {
                    switch (t) {
                        case Nl nl -> {
                            int babLine = nl.pos().line() + 1;
                            if (babylonLineToJavaLine.containsKey(babLine)
                                    && babylonLineToJavaLine.get(babLine) instanceof Integer javaLine
                                    && javaLine != online[0]
                            ) {
                                ansi.apply("\n\n     // #" + javaLine + "  >");
                                doc.javaTextModel.visit(j -> {
                                    if (j.pos().line() == javaLine) {
                                        switch (j) {
                                            case JavaTextModel.JavaType _ -> ansi.fg(YELLOW, _ -> ansi.apply(j));
                                            case At _ -> ansi.fg(YELLOW, _ -> ansi.apply(j));
                                            case DottedName _ -> ansi.fg(GREEN, _ -> ansi.apply(j));
                                            case Nl _ -> ansi.skip();
                                            case Ws _ -> ansi.apply(j);
                                            case ReservedWord _ -> ansi.fg(YELLOW, _ -> ansi.apply(j));
                                            default -> ansi.apply(j);
                                        }
                                    }
                                });
                                online[0] = javaLine;
                            }
                            ansi.apply(t).apply(String.format("%4d :", nl.pos().line() + 1));
                        }
                        case Seq s when s.len() == 1 -> ansi.fg(RED, _ -> ansi.apply(t));
                        case Seq _ -> ansi.fg(PURPLE, _ -> ansi.apply(t));
                        case StringLiteral _, ReservedWord _, BabylonTextModel.BabylonAnonymousAttribute _ ->
                                ansi.fg(YELLOW, _ -> ansi.apply(t));
                        case BabylonTextModel.BabylonLocationAttribute _ ->
                                ansi.skip();//  ansi.bg(WHITE, _ -> ansi.fg(BLUE, _ -> ansi.apply(t)));
                        case BabylonTextModel.BabylonTypeAttribute _ -> ansi.fg(CYAN, _ -> ansi.apply(t));
                        case BabylonTextModel.BabylonBlockDef _ -> ansi.fg(RED, _ -> ansi.apply(t));
                        case BabylonTextModel.BabylonBlock _ -> ansi.fg(BLUE, _ -> ansi.apply(t));
                        case BabylonTextModel.BabylonSSADef _ -> {
                            var parent = t.parent();
                            if (parent instanceof Parenthesis p && p.openClose().open() == '(') {
                                ansi.apply("\n              ");
                            }
                            ansi.fg(YELLOW, _ -> ansi.apply(t));
                        }
                        case BabylonTextModel.BabylonSSARef _ -> ansi.fg(GREEN, _ -> ansi.apply(t));
                        case BabylonTextModel.BabylonBlockOrBody _ -> ansi.fg(PURPLE, _ -> ansi.apply(t));
                        case Op _ -> ansi.fg(CYAN, _ -> ansi.apply(t));
                        case Close c -> {
                            if (c.parent().parent() instanceof Root && c.ch() == ')') {
                                ansi.fg(WHITE, _ -> ansi.apply("\n      " + c.ch()));
                            } else {
                                ansi.fg(WHITE, _ -> ansi.apply("" + c.ch()));
                            }
                        }
                        default -> ansi.fg(WHITE, _ -> ansi.apply(t));
                    }
                }
        );
    }
}


