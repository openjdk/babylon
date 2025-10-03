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
package hat;

import hat.codebuilders.CodeBuilder;
import hat.codebuilders.C99HATConfigBuilder;
import hat.util.StreamMutable;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.Date;

public class FFIConfigCreator {
    public static void main(String[] args) throws IOException {

        Path ffiInclude = Path.of("backends/ffi/shared/src/main/native/include");
        if (!Files.isDirectory(ffiInclude)) {
            System.out.println("No dir at " + ffiInclude);
            System.exit(1);
        }
        Path configDotH = ffiInclude.resolve("config.h");
        if (!Files.isRegularFile(configDotH)) {
            System.out.println("Expected to replace " + configDotH + " but no file exists");
            System.exit(1);
        }

        C99HATConfigBuilder cb = new C99HATConfigBuilder();
        cb.oracleCopyright();
        cb.blockComment("""
                You probably should not edit this this file!!!
                It was auto generated""" + " " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date()) + " by " + FFIConfigCreator.class.getName()
        );
        cb.pragma("once").nl();
        cb.includeSys("iostream").nl();
        final int START_BIT_INDEX = Config.bitList.stream().filter(bit -> bit.size() == 1).findFirst().get().index();

        cb.structKeyword().space().className().braceNlIndented((_) -> {
            var i = StreamMutable.of(START_BIT_INDEX);
            Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit -> {
                cb.staticConstIntShiftedOne(bit.name() + "_BIT", 32, i.get());
                i.set(i.get() + 1);
            });
            cb.constKeyword().space().staticKeyword().space().charType().space().asterisk().bitNamesVar().osbrace().csbrace().semicolon().space().lineComment("See below for initialization");
            cb.constKeyword().space().staticKeyword().space().charType().space().asterisk().bitDescriptionsVar().osbrace().csbrace().semicolon().space().lineComment("See below for initialization");

            cb.intType().space().identifier("configBits").semicolon().nl();

            Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                    cb.identifier("bool").space().camelExceptFirst(bit.name()).semicolon().nl()
            );

            cb.intType().space().identifier("platform").semicolon().nl();
            cb.intType().space().identifier("device").semicolon().nl();
            cb.identifier("bool").space().identifier("alwaysCopy").semicolon().nl();
            //Constructor
            cb.explicitKeyword().space().className().paren((_) -> cb.intType().space().configBitsVar()).colon().nl().indent((_) -> {
                cb.configBitsVar().paren((_) -> cb.configBitsVar()).comma().nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.camelExceptFirst(bit.name()).paren((_) -> cb.paren((_) -> cb.configBitsAndBitName(bit.name())).eq().identifier(bit.name() + "_BIT")).comma().nl()
                );
                cb.identifier("platform").paren((_) -> cb.configBitsAnd().intHexValue(0xf)).comma().nl();
                cb.identifier("alwaysCopy").paren(_ -> cb.pling().camelExceptFirst("MINIMIZE_COPIES")).comma().nl();
                cb.identifier("device").paren(_ ->
                        cb.paren(_ -> cb.configBitsAnd().intHexValue(0xf0)).space().rightShift().space().intValue(4)).braceNlIndented(_ ->
                        cb.ifKeyword().paren(_ -> cb.identifier("info")).braceNlIndented(_ -> {
                            cb.separated(Config.bitList.stream().filter(bit -> bit.size() == 1), CodeBuilder::nl, bit ->
                                    cb.stdCout("native " + C99HATConfigBuilder.toCamelExceptFirst(bit.name()) + " ").space().leftShift().space().camelExceptFirst(bit.name()).space().leftShift().space().stdEndl().semicolon()
                            );
                            cb.nl().stdCout("native platform ").space().leftShift().space().identifier("platform").space().leftShift().space().stdEndl().semicolon();
                            cb.nl().stdCout("native device ").space().leftShift().space().identifier("device").space().leftShift().space().stdEndl().semicolon();
                        })
                );
            }).nl();

            cb.virtualKeyword().space().tilde().className().ocparen().equals().space().defaultKeyword().semicolon();
        }).semicolon().nl().nl();


        cb.hashIfdef("shared_cpp", (_) -> {
            cb.constKeyword().space().charType().space().asterisk().className().colon().colon().bitNamesVar().ocsbrace().equals().brace((_) -> {
                cb.nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.dquote().identifier(bit.name() + "_BIT").dquote().comma().nl()
                );
            }).semicolon().nl();
            cb.constKeyword().space().charType().space().asterisk().className().colon().colon().bitDescriptionsVar().ocsbrace().equals().brace((_) -> {
                cb.nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.dquote().identifier(bit.description()).dquote().comma().nl()
                );
            }).semicolon().nl();
        });

        Files.writeString(configDotH, cb.toString());
    }
}
