package shade;

public interface vec2 {
    float x();

    float y();
    // A mutable form needed for interface mapping.
    interface Field extends vec2 {
        void x(float x);
        void y(float y);
        default vec2 of(float x, float y){
            x(x);y(y);
            return this;
        }
        default vec2 of(vec2 vec2){
            of(vec2.x(),vec2.y());
            return this;
        }
    }

    record Impl(float x, float y) implements vec2 {
    }

    static vec2 vec2(float x, float y) {
        return new Impl(x, y);
    }

    static vec2 vec2(vec2 vec2) {return vec2(vec2.x(), vec2.y());}
    static vec2 vec2(float scalar) {return vec2(scalar,scalar);}

    static vec2 add(vec2 l, vec2 r) {return vec2(l.x()+r.x(),l.y()+r.y());}
    default vec2 add(vec2 rhs){return add(this,rhs);}
    default vec2 add(float scalar){return add(this,vec2(scalar));}

    static vec2 sub(vec2 l, vec2 r) {return vec2(l.x()-r.x(),l.y()-r.y());}
    default vec2 sub(float scalar) {return sub(this, vec2(scalar));}
    default vec2 sub(vec2 rhs){return sub(this,rhs);}

    static vec2 mul(vec2 l, vec2 r) {return vec2(l.x()*r.x(),l.y()*r.y());}
    default vec2 mul(float scalar) {return mul(this, vec2(scalar));}
    default vec2 mul(vec2 rhs){return mul(this,rhs);}

    static vec2 div(vec2 l, vec2 r) {return vec2(l.x()/r.x(),l.y()/r.y());}
    default vec2 div(float scalar) {return div(this, vec2(scalar));}
    default vec2 div(vec2 rhs){return div(this,rhs);}
}
