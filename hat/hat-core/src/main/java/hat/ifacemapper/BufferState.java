package hat.ifacemapper;

import hat.buffer.Buffer;

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

    public static final int NO_STATE = 0;
    public static final int NEW_STATE = 1;
    public static final int HOST_OWNED = 2;
    public static final int DEVICE_OWNED = 3;
    public static final int DEVICE_VALID_HOST_HAS_COPY = 4;
    public static String[] stateNames = new String[]{
            "NO_STATE",
            "NEW_STATE",
            "HOST_OWNED",
            "DEVICE_OWNED",
            "DEVICE_VALID_HOST_HAS_COPY"
    };
    static final MemoryLayout stateMemoryLayout = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("magic1"),
            ValueLayout.ADDRESS.withName("ptr"),
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
    static final VarHandle ptr = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("ptr")
    );
    static final VarHandle length = stateMemoryLayout.varHandle(
            MemoryLayout.PathElement.groupElement("length")
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
    public BufferState setState(int newState) {
        BufferState.state.set(segment, paddedSize, newState);
        return this;
    }
    public BufferState setPtr(MemorySegment  ptr) {
        BufferState.ptr.set(segment, paddedSize, ptr);
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


    public int getState() {
        return (Integer)BufferState.state.get(segment, paddedSize);
    }
    public String getStateString(){
        return stateNames[getState()];
    }
    public MemorySegment getVendorPtr() {
        return (MemorySegment) BufferState.vendorPtr.get(segment, paddedSize);
    }
    public void setVendorPtr(MemorySegment vendorPtr) {
        BufferState.vendorPtr.set(segment, paddedSize, vendorPtr);
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


    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        if (ok()) {
            builder.append("State:ok").append("\n");
            var vendorPtr = getVendorPtr();
            builder.append(",").append("VENDOR_PTR:").append(Long.toHexString(vendorPtr.address()));
            builder.append("\n");
        } else {
            builder.append("State: not ok").append("\n");
        }
        return builder.toString();
    }

}
