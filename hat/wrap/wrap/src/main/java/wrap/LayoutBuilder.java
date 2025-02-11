package wrap;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static java.lang.foreign.MemoryLayout.structLayout;
import static java.lang.foreign.MemoryLayout.unionLayout;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

public class LayoutBuilder {
    //String name;
    List<MemoryLayout> layouts = new ArrayList<>();

    // final public MemoryLayout memoryLayout;
    LayoutBuilder() {
        //this.memoryLayout = memoryLayout;
    }

    LayoutBuilder struct(String name, Consumer<LayoutBuilder> consumer) {
        LayoutBuilder lb = new LayoutBuilder();
        consumer.accept(lb);
        MemoryLayout layout = structLayout(lb.layouts.toArray(new MemoryLayout[0]));
        if (name != null) {
            layout.withName(name);
        }
        layouts.add(layout);
        return this;
    }

    LayoutBuilder union(String name, Consumer<LayoutBuilder> consumer) {
        LayoutBuilder lb = new LayoutBuilder();
        consumer.accept(lb);
        MemoryLayout layout = unionLayout(lb.layouts.toArray(new MemoryLayout[0]));
        if (name != null) {
            layout.withName(name);
        }
        layouts.add(layout);
        return this;
    }

    public LayoutBuilder i32(String name) {
        layouts.add(JAVA_INT.withName(name));
        return this;
    }
    public LayoutBuilder i64(String name) {
        layouts.add(JAVA_LONG.withName(name));
        return this;
    }

    public MemoryLayout memoryLayout(){
        return layouts.getFirst();
    }

    public LayoutBuilder i8Seq(String name, long elementCount) {
        layouts.add(MemoryLayout.sequenceLayout(elementCount, ValueLayout.JAVA_BYTE).withName(name));
        return this;
    }
    public static LayoutBuilder structBuilder(String name, Consumer<LayoutBuilder> consumer) {
        return new LayoutBuilder().struct(name, consumer);
    }
    public static GroupLayout structOf(String name, Consumer<LayoutBuilder> consumer) {
        return (GroupLayout) structBuilder(name, consumer).memoryLayout();
    }
}
