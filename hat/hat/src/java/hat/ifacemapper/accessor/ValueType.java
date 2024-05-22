package hat.ifacemapper.accessor;

public enum ValueType {
    VALUE(false), INTERFACE(true);

    private final boolean virtual;

    ValueType(boolean virtual) {
        this.virtual = virtual;
    }

    public boolean isVirtual() {
        return virtual;
    }
}
