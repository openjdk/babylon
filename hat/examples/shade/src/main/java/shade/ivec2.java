package shade;

// This is immutable
public interface ivec2 {
    int x();
    int y();

    // A mutable form needed for interface mapping.
    interface Field extends ivec2 {
        void x(int x);
        void y(int y);
        default ivec2 of(int x, int y){
            x(x);y(y);
            return this;
        }
        default ivec2 of(ivec2 ivec2){
            of(ivec2.x(),ivec2.y());
            return this;
        }
    }

    record Impl(int x, int y) implements ivec2 {
    }

    //static ivec2 of(int x, int y) {
     //   return new Impl(x, y);
   // }


    static ivec2 ivec2(int x, int y) {
        return new ivec2.Impl(x, y);
    }
    static ivec2 ivec2(ivec2 ivec2) {return ivec2(ivec2.x(), ivec2.y());}
    static ivec2 ivec2(int scalar) {return ivec2(scalar,scalar);}

    static ivec2 add(ivec2 l, ivec2 r) {return ivec2(l.x()+r.x(),l.y()+r.y());}
    default ivec2 add(ivec2 rhs){return add(this,rhs);}
    default ivec2 add(int scalar){return add(this,ivec2(scalar));}

    static ivec2 sub(ivec2 l, ivec2 r) {return ivec2(l.x()-r.x(),l.y()-r.y());}
    default ivec2 sub(int scalar) {return sub(this, ivec2(scalar));}
    default ivec2 sub(ivec2 rhs){return sub(this,rhs);}

    static ivec2 mul(ivec2 l, ivec2 r) {return ivec2(l.x()*r.x(),l.y()*r.y());}
    default ivec2 mul(int scalar) {return mul(this, ivec2(scalar));}
    default ivec2 mul(ivec2 rhs){return mul(this,rhs);}

    static ivec2 div(ivec2 l, ivec2 r) {return ivec2(l.x()/r.x(),l.y()/r.y());}
    default ivec2 div(int scalar) {return div(this, ivec2(scalar));}
    default ivec2 div(ivec2 rhs){return div(this,rhs);}
}
