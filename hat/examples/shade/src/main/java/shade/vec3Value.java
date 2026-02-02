package shade;

public interface vec3Value {
    float x();

    float y();

    float z();

    record Impl(float x, float y, float z) implements vec3Value {
    }

    static vec3Value of(float x, float y, float z) {
        return new Impl(x, y, z);
    }
}
