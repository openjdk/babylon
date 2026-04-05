package optkl.util;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

public class Interner<N> {
    protected final Map<N, N> interned = new LinkedHashMap<>();

    public N intern(N n, Consumer<N> ifAbsent) {
        if (!interned.containsKey(n)) {
            interned.put(n, n);
            ifAbsent.accept(n);
        }
        return interned.get(n);
    }

    public boolean add(N n) {
        boolean[] added ={false};
        intern(n,_->added[0]=true);
        return added[0];
    }
}
