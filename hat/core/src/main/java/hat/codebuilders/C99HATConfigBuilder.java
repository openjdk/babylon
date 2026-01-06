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
package hat.codebuilders;

import hat.Config;
import hat.FFIConfigCreator;
import optkl.codebuilders.C99CodeBuilder;
import optkl.util.StreamMutable;

import java.text.SimpleDateFormat;
import java.util.Date;

public  class C99HATConfigBuilder extends C99CodeBuilder<C99HATConfigBuilder> {

   public final  C99HATConfigBuilder staticConstInt(String name, int padWidth, int value) {
        staticKeyword().space().constexprKeyword().space().s32Type().space().identifier(name, padWidth).space().equals().space().intHexValue(value).semicolon().nl();
        return this;
    }

    public final C99HATConfigBuilder staticConstIntShiftedOne(String name, int padWidth, int shift) {
        staticKeyword().space().constexprKeyword().space().s32Type().space().identifier(name, padWidth).space().equals().space().intValue(1).leftShift().intHexValue(shift).semicolon().nl();
        return this;
    }

    public final C99HATConfigBuilder className() {
        return identifier("BasicConfig");
    }

    public final C99HATConfigBuilder bitNamesVar() {
        return identifier("bitNames");
    }

    public final C99HATConfigBuilder bitDescriptionsVar() {
        return identifier("bitDescriptions");
    }

    public final C99HATConfigBuilder configBitsVar() {
        return identifier("configBits");
    }

    public final C99HATConfigBuilder configBitsAnd() {
        return configBitsVar().space().ampersand().space();
    }

    public final C99HATConfigBuilder configBitsAndBitName(String bitName) {
        return configBitsAnd().identifier(bitName + "_BIT");
    }

    public final C99HATConfigBuilder camelExceptFirst(String s) {
        return identifier(toCamelExceptFirst(s));
    }

    public final C99HATConfigBuilder std(String s) {
        return identifier("std").colon().colon().identifier(s);
    }

    public final C99HATConfigBuilder stdEndl() {
        return std("endl");
    }

    public final C99HATConfigBuilder stdCout(String s) {
        return std("cout").space().leftShift().space().dquote().literal(s).dquote();
    }

    public static String create(){

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
            cb.constKeyword().space().staticKeyword().space().s08Type().space().asterisk().bitNamesVar().osbrace().csbrace().semicolon().space().lineComment("See below for initialization");
            cb.constKeyword().space().staticKeyword().space().s08Type().space().asterisk().bitDescriptionsVar().osbrace().csbrace().semicolon().space().lineComment("See below for initialization");

            cb.s32Type().space().identifier("configBits").semicolon().nl();

            Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                    cb.identifier("bool").space().camelExceptFirst(bit.name()).semicolon().nl()
            );

            cb.s32Type().space().identifier("platform").semicolon().nl();
            cb.s32Type().space().identifier("device").semicolon().nl();
            cb.identifier("bool").space().identifier("alwaysCopy").semicolon().nl();
            //Constructor
            cb.explicitKeyword().space().className().paren((_) -> cb.s32Type().space().configBitsVar()).colon().nl().indent((_) -> {
                cb.configBitsVar().paren((_) -> cb.configBitsVar()).comma().nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.camelExceptFirst(bit.name()).paren((_) -> cb.paren((_) -> cb.configBitsAndBitName(bit.name())).eq().identifier(bit.name() + "_BIT")).comma().nl()
                );
                cb.identifier("platform").paren((_) -> cb.configBitsAnd().intHexValue(0xf)).comma().nl();
                cb.identifier("alwaysCopy").paren(_ -> cb.pling().camelExceptFirst("MINIMIZE_COPIES")).comma().nl();
                cb.identifier("device").paren(_ ->
                        cb.paren(_ -> cb.configBitsAnd().intHexValue(0xf0)).space().rightShift().space().intValue(4)).braceNlIndented(_ ->
                        cb.ifKeyword().paren(_ -> cb.identifier("showDeviceInfo")).braceNlIndented(_ -> {
                            cb.nlSeparated(
                                    Config.bitList.stream().filter(bit -> bit.size() == 1),
                                    bit -> cb.stdCout("native " + cb.toCamelExceptFirst(bit.name()) + " ").space().leftShift().space().camelExceptFirst(bit.name()).space().leftShift().space().stdEndl().semicolon()
                            );
                            cb.nl().stdCout("native platform ").space().leftShift().space().identifier("platform").space().leftShift().space().stdEndl().semicolon();
                            cb.nl().stdCout("native device ").space().leftShift().space().identifier("device").space().leftShift().space().stdEndl().semicolon();
                        })
                );
            }).nl();

            cb.virtualKeyword().space().tilde().className().ocparen().equals().space().defaultKeyword().semicolon();
        }).semicolon().nl().nl();


        cb.hashIfdef("shared_cpp", (_) -> {
            cb.constKeyword().space().s08Type().space().asterisk().className().colon().colon().bitNamesVar().ocsbrace().equals().brace((_) -> {
                cb.nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.dquote().identifier(bit.name() + "_BIT").dquote().comma().nl()
                );
            }).semicolon().nl();
            cb.constKeyword().space().s08Type().space().asterisk().className().colon().colon().bitDescriptionsVar().ocsbrace().equals().brace((_) -> {
                cb.nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.dquote().identifier(bit.description()).dquote().comma().nl()
                );
            }).semicolon().nl();
        });
        return cb.toString();

    }

    static public void main(){
        var c = Config.fromSpec("INFO,SHOW_CODE,HEADLESS,SHOW_KERNEL_MODEL,SHOW_COMPUTE_MODEL,PLATFORM:0,DEVICE:0");
        System.out.println(c);
        System.out.println(create());
        System.exit(1);
    }
}
