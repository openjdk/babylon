package shade;

public interface vec2Value {
    float x();

    float y();


    record Impl(float x, float y) implements vec2Value {
    }

    static vec2Value of(float x, float y) {
        return new Impl(x, y);
    }
}
