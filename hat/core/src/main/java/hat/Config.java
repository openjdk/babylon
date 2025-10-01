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

import java.util.List;

public class Config {

    public record Bit(int index, int size, String name, String description) implements Comparable<Bit> {
        static Bit of(int index, int size, String name, String description){
            return new Bit(index,size,name,description);
        }
        public static Bit of(int index, String name, String description){
            return new Bit(index,1,name,description);
        }

        public static Bit nextBit(Bit bit, int size, String name, String description){
            return new Bit(bit.index+bit.size,size,name, description);
        }
        public static Bit nextBit(Bit bit, String name, String description){
            return nextBit(bit, 1,name,description);
        }
        @Override
        public int compareTo(Bit bit) {
            return Integer.compare(index, bit.index);
        }

        public boolean isSet(int bits){
            return (shifted()&bits) == shifted();
        }
        public int shifted(){
            return 1<<index;
        }
    }


    // Bits 0-3 select platform id 0..5
    // Bits 4-7 select device id 0..15
    public static final Bit PLATFORM =  Bit.of(0,4, "PLATFORM", "FFI ONLY platform id (0-15)");
    public static final Bit DEVICE = Bit.nextBit(PLATFORM, 4, "DEVICE","FFI ONLY device id (0-15)");
    public static final Bit MINIMIZE_COPIES =  Bit.nextBit(DEVICE, "MINIMIZE_COPIES","FFI ONLY Try to minimize copies");
    public static final Bit TRACE = Bit.nextBit(MINIMIZE_COPIES,"TRACE", "FFI ONLY trace code");
    public static final Bit PROFILE = Bit.nextBit(TRACE, "PROFILE", "FFI ONLY Turn on profiling");
    public static final Bit SHOW_CODE = Bit.nextBit(PROFILE,"SHOW_CODE","Show generated code (PTX/OpenCL/CUDA)");
    public static final Bit SHOW_KERNEL_MODEL = Bit.nextBit(SHOW_CODE,"SHOW_KERNEL_MODEL", "Show (via OpWriter) Kernel Model");
    public static final Bit SHOW_COMPUTE_MODEL = Bit.nextBit(SHOW_KERNEL_MODEL,"SHOW_COMPUTE_MODEL", "Show (via OpWriter) Compute Model");
    public static final Bit INFO = Bit.nextBit(SHOW_COMPUTE_MODEL, "INFO", "FFI ONLY Show platform and device info");
    public static final Bit TRACE_COPIES = Bit.nextBit(INFO, "TRACE_COPIES", "FFI ONLY trace copies");
    public static final Bit TRACE_SKIPPED_COPIES = Bit.nextBit(TRACE_COPIES, "TRACE_SKIPPED_COPIES", "FFI ONLY Trace skipped copies (see MINIMIZE_COPIES) ");
    public static final Bit TRACE_ENQUEUES = Bit.nextBit(TRACE_SKIPPED_COPIES,"TRACE_ENQUEUES", "FFI ONLY trace enqueued tasks");
    public static final Bit TRACE_CALLS= Bit.nextBit(TRACE_ENQUEUES, "TRACE_CALLS", "FFI ONLY trace calls (enter/leave)");
    public static final Bit SHOW_WHY = Bit.nextBit(TRACE_CALLS, "SHOW_WHY", "FFI ONLY show why we decided to copy buffer (H to D)");
    public static final Bit SHOW_STATE = Bit.nextBit(SHOW_WHY, "SHOW_STATE", "Show iface buffer state changes");
    public static final Bit PTX = Bit.nextBit(SHOW_STATE, "PTX", "FFI (NVIDIA) ONLY pass PTX rather than C99 CUDA code");
    public static final Bit INTERPRET = Bit.nextBit(PTX, "INTERPRET", "Interpret the code model rather than converting to bytecode");

    public static final List<Bit> bitList = List.of(
            PLATFORM,
            DEVICE,
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



    private int bits;


    public int bits(){
        return bits;
    }
    public void bits(int bits){
        this.bits = bits;
    }

    Config(int bits){
        bits(bits);
    }

    // These must sync with hat/backends/ffi/shared/include/config.h
    // We can create the above config by running main() below...

    public static Config of() {
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

    public static Config of(int bits) {
        return new Config(bits);
    }

    public static Config of(List<Bit> configBits) {
        int allBits = 0;
        for (Bit configBit : configBits) {
            allBits |= configBit.shifted();
        }
        return new Config(allBits);
    }

    public static Config of(Bit... configBits) {
        return of(List.of(configBits));
    }

    public Config and(Bit... configBits) {
        return Config.of(Config.of(List.of(configBits)).bits & bits);
    }

    public Config or(Bit... configBits) {
        return Config.of(Config.of(List.of(configBits)).bits | bits);
    }

    public record BitValue(Bit bit, int value){}

    public static Config of(String spec) {
        if (spec == null || spec.equals("")) {
            return Config.of(0);
        }
        for (Bit bit:bitList) {
            if (bit.name().equals(spec)) {
                return new Config(bit.shifted());
            }
        }
        if (spec.contains(",")) {
            var bits = 0;
            for (var opt: spec.split(",")) {
                var split = opt.split(":");
                var valName=split[0];
                var value=split.length==1?1:Integer.parseInt(split[1]);
                var bitValue = Config.bitList.stream()
                        .filter(bit ->bit.name().equals(valName))
                        .map(bit -> new BitValue(bit, value))
                        .findFirst()
                        .orElseThrow();
                bits |= bitValue.value << bitValue.bit.index();
            }
            return of(bits);
        } else {
            System.out.println("Unexpected spec '" + spec + "'");
            System.exit(1);
            return Config.of(0);
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

}
