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

package hat.backend.ffi;

import hat.Config;
import hat.codebuilders.HATCodeBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class FFIConfig implements Config {
    private int bits;
    static private List<Bit> bitList;

    @Override
    public int bits(){
        return bits;
    }
    @Override
    public void bits(int bits){
        this.bits = bits;
    }


    FFIConfig(int bits){
        bits(bits);
    }

    // These must sync with hat/backends/ffi/shared/include/config.h
    // We can create the above config by running main() below...
    // Bits 0-3 select platform id 0..5
    // Bits 4-7 select device id 0..15
    // Bits 8-15 unused at present.
    // These bits start at 16

    public static final Bit MINIMIZE_COPIES =  Bit.of(16, "MINIMIZE_COPIES");
    public static final Bit TRACE = Bit.nextBit(MINIMIZE_COPIES,"TRACE");
    public static final Bit PROFILE = Bit.nextBit(TRACE, "PROFILE");
    public static final  Bit SHOW_CODE = Bit.nextBit(PROFILE,"SHOW_CODE");
    public static final Bit SHOW_KERNEL_MODEL = Bit.nextBit(SHOW_CODE,"SHOW_KERNEL_MODEL");
    public static final Bit SHOW_COMPUTE_MODEL = Bit.nextBit(SHOW_KERNEL_MODEL,"SHOW_COMPUTE_MODEL");
    public static final Bit INFO = Bit.nextBit(SHOW_COMPUTE_MODEL, "INFO");
    public static final Bit TRACE_COPIES = Bit.nextBit(INFO, "TRACE_COPIES");
    public static final Bit TRACE_SKIPPED_COPIES = Bit.nextBit(TRACE_COPIES, "TRACE_SKIPPED_COPIES");
    public static final Bit TRACE_ENQUEUES = Bit.nextBit(TRACE_SKIPPED_COPIES,"TRACE_ENQUEUES");
    public static final Bit TRACE_CALLS= Bit.nextBit(TRACE_ENQUEUES, "TRACE_CALLS");
    public static final Bit SHOW_WHY = Bit.nextBit(TRACE_CALLS, "SHOW_WHY");
    public static final Bit SHOW_STATE = Bit.nextBit(SHOW_WHY, "SHOW_STATE");
    public static final Bit PTX = Bit.nextBit(SHOW_STATE, "PTX");
    public static final Bit INTERPRET = Bit.nextBit(PTX, "INTERPRET");

    static {
        bitList = List.of(
                MINIMIZE_COPIES,
                TRACE,
                PROFILE,
                SHOW_CODE,
                SHOW_KERNEL_MODEL,
                SHOW_COMPUTE_MODEL,
                INFO,
                TRACE_COPIES,
                TRACE_SKIPPED_COPIES,
                TRACE_ENQUEUES,
                TRACE_CALLS,
                SHOW_WHY,
                SHOW_STATE,
                PTX,
                INTERPRET
        );
    }

    public static FFIConfig of() {
        if (System.getenv("HAT") instanceof String opts) {
            System.out.println("From env " + opts);
            return of(opts);
        }
        if (System.getProperty("HAT") instanceof String opts) {
            System.out.println("From prop " + opts);
            return of(opts);
        }
        return of("");
    }

    public static FFIConfig of(int bits) {
        return new FFIConfig(bits);
    }

    public static FFIConfig of(List<Config.Bit> configBits) {
        int allBits = 0;
        for (Config.Bit configBit : configBits) {
            allBits |= configBit.shifted();
        }
        return new FFIConfig(allBits);
    }

    public static FFIConfig of(Config.Bit... configBits) {
        return of(List.of(configBits));
    }

    public FFIConfig and(Config.Bit... configBits) {
        return FFIConfig.of(FFIConfig.of(List.of(configBits)).bits & bits);
    }

    public FFIConfig or(Config.Bit... configBits) {
        return FFIConfig.of(FFIConfig.of(List.of(configBits)).bits | bits);
    }

    public static FFIConfig of(String name) {
        if (name == null || name.equals("")) {
            return FFIConfig.of(0);
        }

        for (Bit bit:bitList) {
            if (bit.name().equals(name)) {
                return new FFIConfig(bit.shifted());
            }
        }
        if (name.contains(",")) {
            List<Config.Bit> configBits = new ArrayList<>();
            Arrays.stream(name.split(",")).forEach(opt -> {
                   boolean found = false;
                   for (var bit:FFIConfig.bitList) {
                       if (bit.name().equals(opt)) {
                           configBits.add(bit);
                           found = true;
                           break;
                       }
                   }
                   if (!found){
                       throw new IllegalStateException("WHAT HAT OPT ?"+opt);

                   }
            }
            );
            return of(configBits);
        } else if (name.contains(":")) {
            var tokens = name.split(":");
            if (tokens.length == 2) {
                var token = tokens[0];
                if (token.equals("PLATFORM") || token.equals("DEVICE")) {
                    int value = Integer.parseInt(tokens[1]);
                    return new FFIConfig(value << (token.equals("DEVICE") ? 4 : 0));
                } else {
                    System.out.println("Unexpected opt '" + name + "'");
                    return FFIConfig.of(0);
                }
            } else {
                System.out.println("Unexpected opt '" + name + "'");
                return FFIConfig.of(0);
            }
        } else {
            System.out.println("Unexpected opt '" + name + "'");
            System.exit(1);
            return FFIConfig.of(0);
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (Bit bit:bitList){
            if (bit.isSet(bits)) {
                if (!builder.isEmpty()) {
                    builder.append("|");
                }
                builder.append(bit.name());

            }
        }
        return builder.toString();
    }

    public static class Main {
        public static class ConfigBuilder extends HATCodeBuilder<ConfigBuilder> {

            ConfigBuilder staticConstInt(String name, int padWidth, int value) {
                staticKeyword().space().constexprKeyword().space().intType().space().identifier(name, padWidth).space().equals().space().intHexValue(value).semicolon().nl();
                return this;
            }

            ConfigBuilder staticConstIntShiftedOne(String name, int padWidth, int shift) {
                staticKeyword().space().constexprKeyword().space().intType().space().identifier(name, padWidth).space().equals().space().intValue(1).leftShift().intHexValue(shift).semicolon().nl();
                return this;
            }

            ConfigBuilder className() {
                return identifier("BasicConfig");
            }

            ConfigBuilder bitNamesVar() {
                return identifier("bitNames");
            }

            ConfigBuilder configBitsVar() {
                return identifier("configBits");
            }

            ConfigBuilder configBitsAnd() {
                return configBitsVar().space().ampersand().space();
            }

            ConfigBuilder configBitsAndBitName(String bitName) {
                return configBitsAnd().identifier(bitName + "_BIT");
            }

            static String toCamelExceptFirst(String s) {
                String[] parts = s.split("_");
                StringBuilder camelCaseString = new StringBuilder("");
                for (String part : parts) {
                    camelCaseString.append(camelCaseString.isEmpty()
                            ? part.toLowerCase()
                            : part.substring(0, 1).toUpperCase() + part.substring(1).toLowerCase());
                }
                return camelCaseString.toString();
            }

            ConfigBuilder camelExceptFirst(String s) {
                return identifier(toCamelExceptFirst(s));
            }

            ConfigBuilder std(String s) {
                return identifier("std").colon().colon().identifier(s);
            }

            ConfigBuilder stdEndl() {
                return std("endl");
            }

            ConfigBuilder stdCout(String s) {
                return std("cout").space().leftShift().space().dquote().literal(s).dquote();
            }
        }

        public static void main(String[] args) {
            FFIConfig c = FFIConfig.of("INFO,PTX");
            ConfigBuilder cb = new ConfigBuilder();

            cb.lineComment("Auto generated from  " + FFIConfig.class.getName());
            cb.pragma("once").nl();
            cb.includeSys("iostream").nl();
            final int START_BIT_INDEX = 0x10;

            cb.structKeyword().space().className().braceNlIndented((_) -> {
                cb.staticConstInt("START_BIT_IDX", 32, START_BIT_INDEX);
                int i = START_BIT_INDEX;
                for (var bit : FFIConfig.bitList) {
                    cb.staticConstIntShiftedOne(bit.name() + "_BIT", 32, i++);
                }
                cb.staticConstInt("END_BIT_IDX", 32, i);
                cb.constKeyword().space().staticKeyword().space().charType().space().asterisk().bitNamesVar().osbrace().csbrace().semicolon().space().lineComment("See below for initialization");
                cb.intType().space().identifier("configBits").semicolon().nl();

                for (var bit : FFIConfig.bitList) {
                    cb.identifier("bool").space().camelExceptFirst(bit.name()).semicolon().nl();
                }

                cb.intType().space().identifier("platform").semicolon().nl();
                cb.intType().space().identifier("device").semicolon().nl();
                cb.identifier("bool").space().identifier("alwaysCopy").semicolon().nl();
//Constructor
                cb.explicitKeyword().space().className().paren((_) -> cb.intType().space().configBitsVar()).colon().nl().indent((_) -> {
                    cb.configBitsVar().paren((_) -> cb.configBitsVar()).comma().nl();
                    for (var bit : FFIConfig.bitList) {
                        cb.camelExceptFirst(bit.name()).paren((_) -> cb.paren((_) -> cb.configBitsAndBitName(bit.name())).eq().identifier(bit.name() + "_BIT")).comma().nl();

                    }
                    cb.identifier("platform").paren((_) -> cb.configBitsAnd().intHexValue(0xf)).comma().nl();
                    cb.identifier("alwaysCopy").paren(_->cb.pling().camelExceptFirst("MINIMIZE_COPIES")).comma().nl();
                    cb.identifier("device").paren(_ ->
                            cb.paren(_ -> cb.configBitsAnd().intHexValue(0xf0)).space().rightShift().space().intValue(4)).braceNlIndented(_ ->
                            cb.ifKeyword().paren(_ -> cb.identifier("info")).braceNlIndented(_ -> {
                                for (var bit : FFIConfig.bitList) {
                                    cb.stdCout("native " + ConfigBuilder.toCamelExceptFirst(bit.name()) + " ").space().leftShift().space().camelExceptFirst(bit.name()).space().leftShift().space().stdEndl().semicolon().nl();
                                }
                            })
                    );
                }).nl().nl();

                cb.virtualKeyword().space().tilde().className().paren((_) -> {
                }).equals().space().defaultKeyword().semicolon();
            }).semicolon().nl().nl();


            cb.hashIfdef("shared_cpp", (_) -> {
                cb.constKeyword().space().charType().space().asterisk().className().colon().colon().bitNamesVar().sbrace(_ -> {}).equals().brace((_) -> {
                    cb.nl();
                    for (var bit : FFIConfig.bitList) {
                        cb.dquote().identifier(bit.name() + "_BIT").dquote().comma().nl();
                    }
                }).semicolon().nl();
            });
            System.out.println(cb);
        }
    }
}
