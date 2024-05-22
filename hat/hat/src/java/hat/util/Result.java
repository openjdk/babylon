package hat.util;

import java.util.Optional;

public class Result<R> {
    private Optional<R> value = Optional.empty();

    public void of(R value) {
        this.value = Optional.of(value);
    }

    public boolean isPresent() {
        return value.isPresent();
    }

    public R get() {
        return value.orElseThrow();
    }

    public Result(R initial) {
        of(initial);
    }

    public Result() {

    }
}
