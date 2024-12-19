/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.MemorySegment.Scope;
import static java.lang.foreign.ValueLayout.*;
import oneapi.levelzero.ze_api_h;
import static oneapi.levelzero.ze_api_h.*;
import oneapi.levelzero.ze_device_mem_alloc_desc_t;
import oneapi.levelzero.ze_host_mem_alloc_desc_t;

public final class UsmArena implements Arena {
    private final Arena lzArena;
    private final MemorySegment contextHandle;
    private final MemorySegment deviceHandle;

    public UsmArena(MemorySegment contextHandle, MemorySegment deviceHandle) {
        lzArena = Arena.global();
        this.contextHandle = contextHandle;
        this.deviceHandle = deviceHandle;
    }

    public MemorySegment allocate(long byteSize, long byteAlignment) {
        MemorySegment pDeviceMemAllocDesc = lzArena.allocate(ze_device_mem_alloc_desc_t.layout());
        ze_device_mem_alloc_desc_t.stype(pDeviceMemAllocDesc, ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC());
        ze_device_mem_alloc_desc_t.ordinal(pDeviceMemAllocDesc, 0);
        MemorySegment pHostMemAllocDesc = lzArena.allocate(ze_host_mem_alloc_desc_t.layout());
        ze_host_mem_alloc_desc_t.stype(pHostMemAllocDesc, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC());
        MemorySegment pBuffer = lzArena.allocate(ADDRESS);
        check(zeMemAllocShared(contextHandle, pDeviceMemAllocDesc, pHostMemAllocDesc, byteSize, byteAlignment, deviceHandle, pBuffer));
        long address = pBuffer.get(JAVA_LONG, 0);
        return MemorySegment.ofAddress(address).reinterpret(byteSize);
    }

    public void close() {
    }

    public MemorySegment.Scope scope() {
        return null;
    }

    public void freeSegment(MemorySegment segment) {
        check(zeMemFree(contextHandle, segment));
    }

    private static void check(int result) {
        if (result != ZE_RESULT_SUCCESS()) {
            throw new RuntimeException(String.format("Call failed: 0x%x (%d)", result, result));
        }
    }
}