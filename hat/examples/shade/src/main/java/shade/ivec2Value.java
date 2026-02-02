package shade;

public interface ivec2Value {
    int x();

    int y();
    interface Mutable extends ivec2Value{
        void x(int x);
        void y(int y);
        default ivec2Value of(int x, int y){
            x(x);y(y);
            return this;
        }
        default ivec2Value of(ivec2Value ivec2){
            of(ivec2.x(),ivec2.y());
            return this;
        }
    }

    record Impl(int x, int y) implements ivec2Value {
    }

    static ivec2Value of(int x, int y) {
        return new Impl(x, y);
    }


    static ivec2Value ivec2(int x, int y) {
        return new ivec2Value.Impl(x, y);
    }
    static ivec2Value ivec2(ivec2Value ivec2Value) {return ivec2(ivec2Value.x(),ivec2Value.y());}
    static ivec2Value ivec2(int scalar) {return ivec2(scalar,scalar);}

    static ivec2Value add(ivec2Value l, ivec2Value r) {return ivec2(l.x()+r.x(),l.y()+r.y());}
    default ivec2Value add(ivec2Value rhs){return add(this,rhs);}
    default ivec2Value add(int scalar){return add(this,ivec2(scalar));}

    static ivec2Value sub(ivec2Value l, ivec2Value r) {return ivec2(l.x()-r.x(),l.y()-r.y());}
    default ivec2Value sub( int scalar) {return sub(this, ivec2(scalar));}
    default ivec2Value sub(ivec2Value rhs){return sub(this,rhs);}

    static ivec2Value mul(ivec2Value l, ivec2Value r) {return ivec2(l.x()*r.x(),l.y()*r.y());}
    default ivec2Value mul( int scalar) {return mul(this, ivec2(scalar));}
    default ivec2Value mul(ivec2Value rhs){return mul(this,rhs);}

    static ivec2Value div(ivec2Value l, ivec2Value r) {return ivec2(l.x()/r.x(),l.y()/r.y());}
    default ivec2Value div( int scalar) {return div(this, ivec2(scalar));}
    default ivec2Value div(ivec2Value rhs){return div(this,rhs);}
}
