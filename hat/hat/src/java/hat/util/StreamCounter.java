package hat.util;


import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class StreamCounter<E> {
    private int value;

    public static <E> void of(Iterable<E> iterable, BiConsumer<StreamCounter<E>, E> counterConsumer) {
        StreamCounter<E> sc = new StreamCounter<>();
        iterable.spliterator();
        iterable.forEach((e) -> {
            sc.convey = e;
            counterConsumer.accept(sc, e);
            sc.inc();
        });

    }

    E convey;

    public static <E> void of(Stream<E> stream, BiConsumer<StreamCounter<E>, E> counterConsumer) {
        StreamCounter<E> sc = new StreamCounter<>();
        stream.forEach((e) -> {
            sc.convey = e;
            counterConsumer.accept(sc, e);
            sc.inc();
        });

    }

    public static void of(Consumer<StreamCounter> counterConsumer) {
        counterConsumer.accept(new StreamCounter());
    }

    public int value() {
        return value;
    }

    public boolean isFirst() {
        return value == 0;
    }

    public boolean isNotFirst() {
        return value != 0;
    }

    public boolean onFirst(Consumer<E> consumer) {
        if (isFirst()) {
            consumer.accept(convey);
            return true;
        }
        return false;
    }

    public boolean onNotFirst(Consumer<E> consumer) {
        if (!isFirst()) {
            consumer.accept(convey);
            return true;
        }
        return false;
    }

    public void inc() {
        value++;
    }

    private StreamCounter() {
        this.value = 0;
    }

}
