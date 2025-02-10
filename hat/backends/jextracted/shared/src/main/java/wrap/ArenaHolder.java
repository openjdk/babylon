package wrap;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public interface ArenaHolder {
    public static ArenaHolder wrap(Arena arena) {
        return ()-> arena;
    }

    Arena arena();

    default Wrap.IntPtr intPtr(int value){
        return Wrap.IntPtr.of(arena(), value);
    }
    default Wrap.LongPtr longPtr(long value){
        return Wrap.LongPtr.of(arena(), value);
    }
    default Wrap.FloatPtr floatPtr(float value){
        return Wrap.FloatPtr.of(arena(), value);
    }
    default Wrap.IntArr ofInts(int ...values){
        return  Wrap.IntArr.of(arena(), values);
    }
    default Wrap.FloatArr ofFloats(float ...values){
        return  Wrap.FloatArr.of(arena(), values);
    }
    default Wrap.CStrPtr cstr(MemorySegment segment){
        return Wrap.CStrPtr.of( segment);
    }

    default Wrap.CStrPtr cstr(String s){
        return Wrap.CStrPtr.of(arena(), s);
    }
    default Wrap.CStrPtr cstr(long size){
        return Wrap.CStrPtr.of(arena(), (int)size);
    }
    default Wrap.CStrPtr cstr(int size){
        return Wrap.CStrPtr.of(arena(), size);
    }
    default Wrap.PtrArr ptrArr(MemorySegment ... memorySegments) {
        return Wrap.PtrArr.of(arena(), memorySegments);
    }
    default Wrap.PtrArr ptrArr(Wrap.Ptr ...ptrs) {
        return Wrap.PtrArr.of(arena(), ptrs);
    }
    default Wrap.PtrArr ptrArr(int len) {
        return Wrap.PtrArr.of(arena(), len);
    }

    default Wrap.PtrArr ptrArr(String ...strings) {
        return Wrap.PtrArr.of(arena(), strings);
    }
}
