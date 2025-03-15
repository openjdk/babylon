package hat.ifacemapper;

import hat.buffer.Buffer;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.util.Objects;



    /*


     See backend_ffi_shared/include/shared.h

     Make sure the final static values below match the #defines
      // hat iface buffer bitz
    // hat iface bffa   bitz
    // 4a7 1face bffa   b175



    struct state{
       long magic1; // MAGIC
       int bits;
       int mode;
       void * vendorPtr; // In OpenCL this points to native OpenCL::Buffer
       long magic2; // MAGIC
    }
     */

public record BufferState(MemorySegment segment, long paddedSize) {
    public static final long alignment = ValueLayout.JAVA_LONG.byteSize();
    // hat iface buffer bitz
    // hat iface bffa   bitz
    // 4a7 1face bffa   b175
    public static final long MAGIC = 0x4a71facebffab175L;
    public static int NONE = 0;
    public static int BIT_HOST_NEW = 1<< 0;
    public static int BIT_DEVICE_NEW = 1 << 1;
    public static int BIT_HOST_DIRTY = 1 << 2;
    public static int BIT_DEVICE_DIRTY = 1 << 3;
    public static int BIT_HOST_CHECKED = 1 << 4;

    public static int NO_STATE = 0;
    public static int NEW_STATE = 1;
    public static int HOST_OWNED = 2;
    public static int DEVICE_OWNED = 3;
    public static int DEVICE_VALID_HOST_HAS_COPY = 4;
    public static String[] stateNames = new String[]{
            "NO_STATE",
            "NEW_STAT",
            "HOST_OWNED",
            "DEVICE_OWNED",
            "DEVICE_VALID_HOST_HAS_COPY"
    };
    static final MemoryLayout stateMemoryLayout = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("magic1"),
            ValueLayout.JAVA_LONG.withName("length"),
            ValueLayout.JAVA_INT.withName("bits"),
            ValueLayout.JAVA_INT.withName("state"),
            ValueLayout.ADDRESS.withName("vendorPtr"),
            ValueLayout.JAVA_LONG.withName("magic2")
    ).withName("state");

    static long byteSize() {
        return stateMemoryLayout.byteSize();
    }

    static final VarHandle magic1 = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("magic1")
    );
    static final VarHandle length = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("length")
    );
    static final VarHandle bits = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("bits")
    );
    static final VarHandle state = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("state")
    );

    static final VarHandle magic2 = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("magic2")
    );

    static final VarHandle vendorPtr = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("vendorPtr")
    );

    public static long getLayoutSizeAfterPadding(GroupLayout layout) {
        return layout.byteSize() +
                ((layout.byteSize() % BufferState.alignment) == 0 ? 0 : BufferState.alignment - (layout.byteSize() % BufferState.alignment));
    }

    public static <T> BufferState of(T t) {
        Buffer buffer = (Buffer) Objects.requireNonNull(t);
        MemorySegment s = Buffer.getMemorySegment(buffer);
        return new BufferState(s, s.byteSize() - BufferState.byteSize());
    }
    BufferState setState(int newState) {
        BufferState.state.set(segment, paddedSize, newState);
        return this;
    }

    BufferState setLength(long newLength) {
        BufferState.length.set(segment, paddedSize, newLength);
        return this;
    }
    BufferState setMagic() {
        BufferState.magic1.set(segment, paddedSize, MAGIC);
        BufferState.magic2.set(segment, paddedSize, MAGIC);
        return this;
    }

    public BufferState assignBits(int bits) {
        BufferState.bits.set(segment, paddedSize, bits);
        return this;
    }

    public BufferState and(int bitz) {
        BufferState.bits.set(segment, paddedSize, getBits() & bitz);
        return this;
    }

    public BufferState or(int bitz) {
        BufferState.bits.set(segment, paddedSize, getBits() | bitz);
        return this;
    }

    public BufferState xor(int bitz) {
        // if getBits() = 0b0111 (7) and bitz = 0b0100 (4) xored = 0x0011 3
        // if getBits() = 0b0011 (3) and bitz = 0b0100 (4) xored = 0x0111 7
        BufferState.bits.set(segment, paddedSize, getBits() ^ bitz);
        return this;
    }

    public BufferState andNot(int bitz) {
        // if getBits() = 0b0111 (7) and bitz = 0b0100 (4) andNot = 0b0111 & 0b1011 = 0x0011 3
        // if getBits() = 0b0011 (3) and bitz = 0b0100 (4) andNot = 0b0011 & 0b1011 = 0x0011 3
        BufferState.bits.set(segment, paddedSize, getBits() & ~bitz);
        return this;
    }


    public int getBits() {
        return (Integer) BufferState.bits.get(segment, paddedSize);
    }

    public MemorySegment getVendorPtr() {
        return (MemorySegment) BufferState.vendorPtr.get(segment, paddedSize);
    }

    public void setVendorPtr(MemorySegment vendorPtr) {
        BufferState.vendorPtr.set(segment, paddedSize, vendorPtr);
    }

    public boolean all(int bitz) {
        return (getBits() & bitz) == bitz;
    }

    public boolean any(int bitz) {
        return (getBits() & bitz) != 0;
    }

    public BufferState setHostDirty(boolean dirty) {
        if (dirty) {
            or(BIT_HOST_DIRTY);
        } else {
            andNot(BIT_HOST_DIRTY);
        }
        return this;
    }

    public BufferState setHostChecked(boolean checked) {
        if (checked) {
            or(BIT_HOST_CHECKED);
        } else {
            andNot(BIT_HOST_CHECKED); // this is wrong we want bits&=!BIT_DEVICE_DIRTY
        }
        return this;
    }

    public BufferState setDeviceDirty(boolean dirty) {
        if (dirty) {
            or(BIT_DEVICE_DIRTY);
        } else {
            andNot(BIT_DEVICE_DIRTY); // this is wrong we want bits&=!BIT_DEVICE_DIRTY
        }
        return this;
    }

    public boolean isHostNew() {
        return all(BIT_HOST_NEW);
    }

    public boolean isHostDirty() {
        return all(BIT_HOST_DIRTY);
    }

    public boolean isHostChecked() {
        return all(BIT_HOST_CHECKED);
    }

    public boolean isHostNewOrDirty() {
        return all(BIT_HOST_NEW | BIT_HOST_DIRTY);
    }

    public boolean isDeviceDirty() {
        return all(BIT_DEVICE_DIRTY);
    }

    public BufferState clearHostChecked() {
        return xor(BIT_HOST_CHECKED);
    }

    public BufferState clearDeviceDirty() {
        return xor(BIT_DEVICE_DIRTY);
    }

    public BufferState resetHostDirty() {
        return xor(BIT_HOST_DIRTY);
    }

    public BufferState resetHostNew() {
        return xor(BIT_HOST_NEW);
    }

    public long magic1() {
        return (Long) BufferState.magic1.get(segment, paddedSize);
    }

    public long magic2() {
        return (Long) BufferState.magic2.get(segment, paddedSize);
    }

    public boolean ok() {
        return MAGIC == magic1() && MAGIC == magic2();
    }

    static String paddedString(int bits) {
        String s = Integer.toBinaryString(bits);
        String s32 = "                                  ";
        return s32.substring(0, s32.length() - s.length()) + s;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        if (ok()) {
            builder.append("State:ok").append("\n");
            builder.append("State:Bits:").append(paddedString(getBits()));
            if (all(BIT_HOST_DIRTY)) {
                builder.append(",").append("HOST_DIRTY");
            }
            if (all(BIT_DEVICE_DIRTY)) {
                builder.append(",").append("DEVICE_DIRTY");
            }
            if (all(BIT_HOST_NEW)) {
                builder.append(",").append("HOST_NEW");
            }
            var vendorPtr = getVendorPtr();
            builder.append(",").append("VENDOR_PTR:").append(Long.toHexString(vendorPtr.address()));
            builder.append("\n");


        } else {
            builder.append("State: not ok").append("\n");
        }
        return builder.toString();
    }


}
