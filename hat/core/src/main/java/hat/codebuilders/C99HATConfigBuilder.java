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
import optkl.codebuilders.ScopedCodeBuilderContext;
import optkl.util.Mutable;

import java.lang.invoke.MethodHandles;
import java.text.SimpleDateFormat;
import java.util.Date;

public  class C99HATConfigBuilder extends C99CodeBuilder<C99HATConfigBuilder> {

    public C99HATConfigBuilder(ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(scopedCodeBuilderContext);
    }

    public final  C99HATConfigBuilder staticConstInt(String name, int padWidth, int value) {
        staticKeyword().sp().constexprKeyword().sp().s32Type().sp().identifier(name, padWidth).sp().equals().sp().intHexValue(value).semicolon().nl();
        return this;
    }

    public final C99HATConfigBuilder staticConstIntShiftedOne(String name, int padWidth, int shift) {
        staticKeyword().sp().constexprKeyword().sp().s32Type().sp().identifier(name, padWidth).sp().equals().sp().intValue(1).leftShift().intHexValue(shift).semicolon().nl();
        return this;
    }

    public final C99HATConfigBuilder className() {
        return id("BasicConfig");
    }

    public final C99HATConfigBuilder bitNamesVar() {
        return id("bitNames");
    }

    public final C99HATConfigBuilder bitDescriptionsVar() {
        return id("bitDescriptions");
    }

    public final C99HATConfigBuilder configBitsVar() {
        return id("configBits");
    }

    public final C99HATConfigBuilder configBitsAnd() {
        return configBitsVar().sp().ampersand().sp();
    }

    public final C99HATConfigBuilder configBitsAndBitName(String bitName) {
        return configBitsAnd().id(bitName + "_BIT");
    }

    public final C99HATConfigBuilder camelExceptFirst(String s) {
        return id(toCamelExceptFirst(s));
    }

    public final C99HATConfigBuilder std(String s) {
        return id("std").colon().colon().id(s);
    }

    public final C99HATConfigBuilder stdEndl() {
        return std("endl");
    }

    public final C99HATConfigBuilder stdCout(String s) {
        return std("cout").sp().leftShift().sp().dquote().literal(s).dquote();
    }

    public static String create(){
        C99HATConfigBuilder cb = new C99HATConfigBuilder(new ScopedCodeBuilderContext(MethodHandles.lookup(),null));
        cb.oracleCopyright();
        cb.blockComment("""
                You probably should not edit this this file!!!
                It was auto generated""" + " " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date()) + " by " + FFIConfigCreator.class.getName()
        );
        cb.pragma("once").nl();
        cb.includeSys("iostream").nl();
        final int START_BIT_INDEX = Config.bitList.stream().filter(bit -> bit.size() == 1).findFirst().get().index();

        cb.structKeyword().sp().className().braceNlIndented((_) -> {
            var i = Mutable.of(START_BIT_INDEX);
            Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit -> {
                cb.staticConstIntShiftedOne(bit.name() + "_BIT", 32, i.get());
                i.set(i.get() + 1);
            });
            cb.constKeyword().sp().staticKeyword().sp().s08Type().sp().asterisk().bitNamesVar().osbrace().csbrace().semicolon().sp().lineComment("See below for initialization");
            cb.constKeyword().sp().staticKeyword().sp().s08Type().sp().asterisk().bitDescriptionsVar().osbrace().csbrace().semicolon().sp().lineComment("See below for initialization");

            cb.s32Type().sp().id("configBits").semicolon().nl();

            Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                    cb.id("bool").sp().camelExceptFirst(bit.name()).semicolon().nl()
            );

            cb.s32Type().sp().id("platform").semicolon().nl();
            cb.s32Type().sp().id("device").semicolon().nl();
            cb.id("bool").sp().id("alwaysCopy").semicolon().nl();
            //Constructor
            cb.explicitKeyword().sp().className().paren((_) -> cb.s32Type().sp().configBitsVar()).colon().nl().indent((_) -> {
                cb.configBitsVar().paren((_) -> cb.configBitsVar()).comma().nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.camelExceptFirst(bit.name()).paren((_) -> cb.paren((_) -> cb.configBitsAndBitName(bit.name())).eq().id(bit.name() + "_BIT")).comma().nl()
                );
                cb.id("platform").paren((_) -> cb.configBitsAnd().intHexValue(0xf)).comma().nl();
                cb.id("alwaysCopy").paren(_ -> cb.pling().camelExceptFirst("MINIMIZE_COPIES")).comma().nl();
                cb.id("device").paren(_ ->
                        cb.paren(_ -> cb.configBitsAnd().intHexValue(0xf0)).sp().rightShift().sp().intValue(4)).braceNlIndented(_ ->
                        cb.ifKeyword().paren(_ -> cb.id("showDeviceInfo")).braceNlIndented(_ -> {
                            cb.nlSeparated(
                                    Config.bitList.stream().filter(bit -> bit.size() == 1),
                                    bit -> cb.stdCout("native " + cb.toCamelExceptFirst(bit.name()) + " ").sp().leftShift().sp().camelExceptFirst(bit.name()).sp().leftShift().sp().stdEndl().semicolon()
                            );
                            cb.nl().stdCout("native platform ").sp().leftShift().sp().id("platform").sp().leftShift().sp().stdEndl().semicolon();
                            cb.nl().stdCout("native device ").sp().leftShift().sp().id("device").sp().leftShift().sp().stdEndl().semicolon();
                        })
                );
            }).nl();

            cb.virtualKeyword().sp().tilde().className().ocparen().equals().sp().defaultKeyword().semicolon();
        }).semicolon().nl().nl();


        cb.hashIfdef("shared_cpp", (_) -> {
            cb.constKeyword().sp().s08Type().sp().asterisk().className().colon().colon().bitNamesVar().ocsbrace().equals().brace((_) -> {
                cb.nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.dquote().id(bit.name() + "_BIT").dquote().comma().nl()
                );
            }).semicolon().nl();
            cb.constKeyword().sp().s08Type().sp().asterisk().className().colon().colon().bitDescriptionsVar().ocsbrace().equals().brace((_) -> {
                cb.nl();
                Config.bitList.stream().filter(bit -> bit.size() == 1).forEach(bit ->
                        cb.dquote().id(bit.description()).dquote().comma().nl()
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
