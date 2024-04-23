package java.lang.reflect.code;

/**
 * Source location information.
 *
 * @param sourceRef the reference to the source
 * @param line the line in the source
 * @param column the column in the source
 */
public record Location(String sourceRef, int line, int column) {

    /**
     * The location value, {@code null}, indicating no location information.
     */
    public static final Location NO_LOCATION = null;

    public Location(int line, int column) {
        this(null, line, column);
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append(line).append(":").append(column);
        if (sourceRef != null) {
            s.append(":").append(sourceRef);
        }
        return s.toString();
    }

    public static Location fromString(String s) {
        String[] split = s.split(":", 3);
        if (split.length < 2) {
            throw new IllegalArgumentException();
        }

        int line = Integer.parseInt(split[0]);
        int column = Integer.parseInt(split[1]);
        String sourceRef;
        if (split.length == 3) {
            sourceRef = split[2];
        } else {
            sourceRef = null;
        }
        return new Location(sourceRef, line, column);
    }
}
