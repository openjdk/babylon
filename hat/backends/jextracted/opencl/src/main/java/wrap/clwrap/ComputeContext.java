package wrap.clwrap;

import opencl.opencl_h;
import wrap.Wrap;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.Map;

import static java.lang.foreign.MemorySegment.NULL;

public class ComputeContext {

    public MemorySegmentState register(MemorySegment segment) {
        return memorySegmentToStateMap.computeIfAbsent(segment, MemorySegmentState::new);
    }

    public MemorySegmentState getState(MemorySegment memorySegment) {
        return memorySegmentToStateMap.get(memorySegment);
    }

    public int incEventc() {
        eventc++;
        return eventc;
    }

    public int eventc() {
        return alwaysBlock ? 0 : eventc;
    }

    public MemorySegment eventsPtr() {
        return (alwaysBlock || eventc == 0) ? NULL : eventsPtr;
    }

    public MemorySegment nextEventPtrSlot() {
        if (alwaysBlock) {
            return NULL;
        } else {
            var seg = eventsPtr.asSlice(eventc * opencl_h.cl_event.byteSize(), opencl_h.cl_event);
            incEventc();
            return seg;
        }
    }

    public void resetEventc() {
        eventc = 0;
    }

    public void waitForEvents() {
        if (!alwaysBlock) {
            CLStatusPtr status = CLStatusPtr.of(arena);
            status.set(opencl_h.clWaitForEvents(eventc(), eventsPtr));
            if (!status.isOK()) {
                System.out.println("failed to wait for events " + status);
            }
            resetEventc();
        }
    }

    public int blockInt() {
        return alwaysBlock ? opencl_h.CL_TRUE() : opencl_h.CL_FALSE();
    }

    public record ClMemPtr(MemorySegment ptr) implements Wrap.Ptr {
        public static ClMemPtr of(Arena arena, MemorySegment clmem){
            return new ClMemPtr(arena.allocateFrom(AddressLayout.ADDRESS,clmem));
        }
        MemorySegment get(){
            return ptr.get(ValueLayout.ADDRESS,0);
        }
        @Override
        public long sizeof() {
            return AddressLayout.ADDRESS.byteSize();
        }
    }
    public static class MemorySegmentState {
        public final MemorySegment memorySegment;
        public ClMemPtr clMemPtr;
        public boolean copyToDevice;
        public boolean copyFromDevice;

        MemorySegmentState(MemorySegment memorySegment) {
            this.memorySegment = memorySegment;
            this.copyToDevice = true;
            this.copyFromDevice = true;
        }
    }

    final int maxEvents;
    private int eventc;
    final MemorySegment eventsPtr;
    final boolean alwaysBlock;
    final Arena arena;

    public ComputeContext(Arena arena, int maxEvents) {
        this.arena = arena;
        this.maxEvents = maxEvents;
        this.eventc = 0;
        this.eventsPtr = arena.allocate(opencl_h.cl_event, maxEvents);
        this.alwaysBlock = false;
    }

    private final Map<MemorySegment, MemorySegmentState> memorySegmentToStateMap = new HashMap<>();
}
