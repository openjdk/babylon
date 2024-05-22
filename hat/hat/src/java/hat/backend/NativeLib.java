package hat.backend;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

public class NativeLib {
    final public String name;
    public final boolean available;


    final public Linker nativeLinker;

    final public SymbolLookup loaderLookup;

    NativeLib(String name) {
        this.name = name;

        boolean nonFinalAvailable = true;
        try {
            Runtime.getRuntime().loadLibrary(name);
        } catch (UnsatisfiedLinkError e) {
            nonFinalAvailable = false;
        }
        this.available = nonFinalAvailable;

        this.nativeLinker = Linker.nativeLinker();

        this.loaderLookup = SymbolLookup.loaderLookup();
    }


    MethodHandle voidFunc(String name, MemoryLayout... args) {
        return loaderLookup.find(name)
                .map(symbolSegment -> nativeLinker.downcallHandle(symbolSegment,
                        FunctionDescriptor.ofVoid(args)))
                .orElse(null);
    }

    MethodHandle typedFunc(String name, MemoryLayout returnLayout, MemoryLayout... args) {
        return loaderLookup.find(name)
                .map(symbolSegment -> nativeLinker.downcallHandle(symbolSegment,
                        FunctionDescriptor.of(returnLayout, args)))
                .orElse(null);
    }

    MethodHandle longFunc(String name, MemoryLayout... args) {
        return typedFunc(name, JAVA_LONG, args);
    }

    MethodHandle booleanFunc(String name, MemoryLayout... args) {
        return typedFunc(name, JAVA_BOOLEAN, args);
    }

    MethodHandle intFunc(String name, MemoryLayout... args) {
        return typedFunc(name, JAVA_INT, args);
    }
}
