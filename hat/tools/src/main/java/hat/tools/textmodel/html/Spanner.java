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
package hat.tools.textmodel.html;


import hat.tools.textmodel.tokens.Token;

import java.io.PrintStream;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public interface Spanner<T extends Spanner<T>> extends  Function<String, T> {
    T apply(String o);

    default T apply(Token t) {
        return apply(t.asString());
    }

    default T self() {
        return (T) this;
    }
    default T esc() {
        return apply("\033");
    }

    default T color(Style style, Consumer<T> consumer) {
        apply("<span class=").apply(style.name).apply(">");
        consumer.accept(self());
        apply("</span>");
        return self();
    }

    default T div(Style style, Consumer<T> consumer) {
        apply("<div class=").apply(style.name).apply(">");
        consumer.accept(self());
        apply("</div>");
        return self();
    }

    default T pre(Style style, Consumer<T> consumer) {
        apply("<pre class=").apply(style.name).apply(">");
        consumer.accept(self());
        apply("</pre>");
        return self();
    }

    record Style(Style base, String type, String name, List<String> s) {
        static Style div(String name) {
            return new Style(null, "pre", name, List.of());
        }

        static Style pre(String name) {
            return new Style(null, "div", name, List.of());
        }

        static Style span(String name) {
            return new Style(null, "span", name, List.of());
        }

        Style with(String name, List<String> s) {
            return new Style(this, this.type, name, s);
        }

        Style with(String name, String... s) {
            return with(name, List.of(s));
        }

        Style bold() {
            return with(this.name() + "_BOLD", "font-weight: bold;");
        }

        Style italic() {
            return with(this.name() + "_ITALIC", "font-style: italic;");
        }

        Style bright() {
            return with(this.name() + "_BRIGHT", "font-style: lighter;");
        }

        Style fg() {
            return with(this.name() + "_FG", "color: " + name().toLowerCase());
        }

        Style bg(String color) {
            return with(this.name(), "background: " + color);
        }

        public Style bg() {
            return with(this.name() + "_BG", "background: " + name().toLowerCase());
        }

        String styleName() {
            return this.type + "." + this.name() + "{\n      " + String.join(";\n       ", this.s) + "\n}";
        }
    }

    Style BLACK = Style.span("black");
    Style RED = Style.span("red");
    Style GREEN = Style.span("green");
    Style YELLOW = Style.span("yellow");
    Style BLUE = Style.span("blue");
    Style PURPLE = Style.span("purple");
    Style CYAN = Style.span("cyan");
    Style MAGENTA = Style.span("magenta");
    Style WHITE = Style.span("white");

    Style DIV = Style.div("code").bg("black");
    Style PRE = Style.pre("code").bg("black");
    List<Style> styles = List.of(
            BLACK.fg(), RED.fg(), GREEN.fg(), YELLOW.fg(), BLUE.fg(), PURPLE.fg(), CYAN.fg(), MAGENTA.fg(), WHITE.fg()
            , DIV, PRE
    );

    default String styleSheet() {
        StringBuilder sb = new StringBuilder("<style>\n");
        for (Style style : styles) {
            sb.append(style.styleName());
        }
        sb.append("</style>");
        return sb.toString();
    }


    default T line(int n) {
        apply("<span style=\"color:black; background:grey; text-align: right; width:50px; margin-left:2px; margin-right:20px; \">" + n + "</span>");
        return (T) this;
    }

    default T black(Consumer<T> cc) {
        return color(BLACK.fg(), cc);
    }

    default T red(Consumer<T> cc) {
        return color(RED.fg(), cc);
    }

    default T green(Consumer<T> cc) {
        return color(GREEN.fg(), cc);
    }

    default T yellow(Consumer<T> cc) {
        return color(YELLOW.fg(), cc);
    }

    default T blue(Consumer<T> cc) {
        return color(BLUE.fg(), cc);
    }

    default T magenta(Consumer<T> cc) {
        return color(PURPLE.fg(), cc);
    }

    default T cyan(Consumer<T> cc) {
        return color(CYAN.fg(), cc);
    }

    default T white(Consumer<T> cc) {
        return color(WHITE.fg(), cc);
    }

    default T blackBold(Consumer<T> cc) {
        return color(BLACK.fg().bold(), cc);
    }

    default T redBold(Consumer<T> cc) {
        return color(RED.fg().bold(), cc);
    }

    default T greenBold(Consumer<T> cc) {
        return color(GREEN.fg().bold(), cc);
    }

    default T yellowBold(Consumer<T> cc) {
        return color(YELLOW.fg().bold(), cc);
    }

    default T blueBold(Consumer<T> cc) {
        return color(BLUE.fg().bold(), cc);
    }

    default T magentaBold(Consumer<T> cc) {
        return color(PURPLE.fg().bold(), cc);
    }

    default T cyanBold(Consumer<T> cc) {
        return color(CYAN.fg().bold(), cc);
    }

    default T whiteBold(Consumer<T> cc) {
        return color(WHITE.fg().bold(), cc);
    }

    default T blackBoldAndBright(Consumer<T> cc) {
        return color(BLACK.fg().bold().bright(), cc);
    }

    default T redBoldAndBright(Consumer<T> cc) {
        return color(RED.fg().bold().bright(), cc);
    }

    default T greenBoldAndBright(Consumer<T> cc) {
        return color(GREEN.fg().bold().bright(), cc);
    }

    default T yellowBoldAndBright(Consumer<T> cc) {
        return color(YELLOW.fg().bold().bright(), cc);
    }

    default T blueBoldAndBright(Consumer<T> cc) {
        return color(BLUE.fg().bold().bright(), cc);
    }

    default T magentaAndBright(Consumer<T> cc) {
        return color(PURPLE.fg().bold().bright(), cc);
    }

    default T cyanBoldAndBright(Consumer<T> cc) {
        return color(CYAN.fg().bold().bright(), cc);
    }

    default T whiteBoldAndBright(Consumer<T> cc) {
        return color(WHITE.fg().bold().bright(), cc);
    }

    default T blackBright(Consumer<T> cc) {
        return color(BLACK.fg().bright(), cc);
    }

    default T redBright(Consumer<T> cc) {
        return color(RED.fg().bright(), cc);
    }

    default T greenBright(Consumer<T> cc) {
        return color(GREEN.fg().bright(), cc);
    }

    default T yellowBright(Consumer<T> cc) {
        return color(YELLOW.fg().bright(), cc);
    }

    default T blueBright(Consumer<T> cc) {
        return color(BLUE.fg().bright(), cc);
    }

    default T magentaBright(Consumer<T> cc) {
        return color(PURPLE.fg().bright(), cc);
    }

    default T cyanBright(Consumer<T> cc) {
        return color(CYAN.fg().bright(), cc);
    }

    default T whiteBright(Consumer<T> cc) {
        return color(WHITE.fg().bright(), cc);
    }


    default T blackBack(Consumer<T> cc) {
        return color(BLACK.bg(), cc);
    }

    default T redBack(Consumer<T> cc) {
        return color(RED.bg(), cc);
    }

    default T greenBack(Consumer<T> cc) {
        return color(GREEN.bg(), cc);
    }

    default T yellowBack(Consumer<T> cc) {
        return color(YELLOW.bg(), cc);
    }

    default T blueBack(Consumer<T> cc) {
        return color(BLUE.bg(), cc);
    }

    default T magentaBack(Consumer<T> cc) {
        return color(PURPLE.bg(), cc);
    }

    default T cyanBack(Consumer<T> cc) {
        return color(CYAN.bg(), cc);
    }

    default T whiteBack(Consumer<T> cc) {
        return color(WHITE.bg(), cc);
    }

    record Adaptor(Consumer<String> consumer) implements Spanner<Adaptor> {
        @Override
        public Adaptor apply(String s) {
            consumer.accept(s);
            return self();
        }
    }

    ;

    static Spanner<Adaptor> adapt(Consumer<String> consumer) {
        return new Adaptor(consumer);
    }

    default T open() {
        return apply("""
                <html>
                   <head>
                """ + styleSheet() + """
                   </head>
                   <body>
                """);
    }

    default T close() {
        return apply("""
                    </body>
                </html>
                """);
    }

    class IMPL implements Spanner<Spanner.IMPL> {
        private final PrintStream printStream;

        @Override
        public IMPL apply(String s) {
            printStream.append(s);
            return self();
        }

        IMPL(PrintStream printStream) {
            this.printStream = printStream;
        }
    }

    static IMPL of(PrintStream printStream) {
        return new IMPL(printStream);
    }

}
