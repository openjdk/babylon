package hat;

public interface Config {
    int bits();
    void bits(int bits);
    record Bit(int index, int size, String name) implements Comparable<Bit> {
        static Bit of(int index, int size, String name){
            return new Bit(index,size,name);
        }
        public static Bit of(int index, String name){
            return new Bit(index,1,name);
        }
        public static Bit nextBit(Bit bit, String name){
            return new Bit(bit.index+1,1,name);
        }

        @Override
        public int compareTo(Bit bit) {
            return Integer.compare(index, bit.index);
        }

        public boolean isSet(int bits){
            return (shifted()&bits) == shifted();
        }
        public int shifted(){
            return 1<<index;
        }
    }
}
