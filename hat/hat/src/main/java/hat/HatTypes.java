package hat;

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Objects;

public class HatTypes {
    public abstract sealed static class HatType implements TypeElement permits HatPtrType {
        String name;
        HatType(String name){
            this.name = name;
        }
    }

    public static final class HatPtrType extends HatType {
        static final String NAME = "hat.ptr";
        final TypeElement type;

        public HatPtrType(TypeElement type) {
            super(NAME);
            this.type = type;

        }

        public TypeElement type() {
            return type;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            HatPtrType hatPtrType = (HatPtrType) o;
            return Objects.equals(type, hatPtrType.type);
        }

        @Override
        public int hashCode() {
            return Objects.hash(type);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(type.externalize()));
        }

        @Override
        public String toString() {
            return externalize().toString();
        }
    }
}
