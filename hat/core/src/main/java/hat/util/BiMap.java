package hat.util;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;

import java.util.LinkedHashMap;
import java.util.Map;

public class BiMap<T1 extends Block.Parameter, T2 extends Op> {
    public Map<T1, T2> t1ToT2 = new LinkedHashMap<>();
    public Map<T2, T1> t2ToT1 = new LinkedHashMap<>();

    public void add(T1 t1, T2 t2) {
        t1ToT2.put(t1, t2);
        t2ToT1.put(t2, t1);
    }

    public T1 get(T2 t2) {
        return t2ToT1.get(t2);
    }

    public T2 get(T1 t1) {
        return t1ToT2.get(t1);
    }

    public boolean containsKey(T1 t1) {
        return t1ToT2.containsKey(t1);
    }

    public boolean containsKey(T2 t2) {
        return t2ToT1.containsKey(t2);
    }
}
