package shade;

//immutable form
public interface vec4 {

    float x();

    float y();

    float z();

    float w();

    // A mutable variant needed for interface mapping
    interface Field extends vec4 {
        void x(float x);
        void y(float y);
        void z(float z);
        void w(float w);
        default vec4 of(float x, float y, float z, float w){
            x(x);y(y);z(z);w(w);
            return this;
        }
        default vec4 of(vec4 vec4){
            of(vec4.x(),vec4.y(),vec4.z(),vec4.w());
            return this;
        }
    }

    record Impl(float x, float y, float z, float w) implements vec4 {
    }


    static vec4 vec4(float x, float y, float z, float w) {
        return new Impl(x, y, z, w);
    }
    static vec4 vec4(vec4 vec4) {return vec4(vec4.x(), vec4.y(), vec4.z(), vec4.w());}
    static vec4 vec4(float scalar) {return vec4(scalar,scalar,scalar,scalar);}

    static vec4 add(vec4 l, vec4 r) {return vec4(l.x()+r.x(),l.y()+r.y(), l.z()+r.z(),l.w()+r.w());}
    default vec4 add(vec4 rhs){return add(this,rhs);}
    default vec4 add(float scalar){return add(this,vec4(scalar));}

    static vec4 sub(vec4 l, vec4 r) {return vec4(l.x()-r.x(),l.y()-r.y(), l.z()-r.z(),l.w()-r.w());}
    default vec4 sub(float scalar) {return sub(this, vec4(scalar));}
    default vec4 sub(vec4 rhs){return sub(this,rhs);}

    static vec4 mul(vec4 l, vec4 r) {return vec4(l.x()*r.x(),l.y()*r.y(), l.z()*r.z(),l.w()*r.w());}
    default vec4 mul(float scalar) {return mul(this, vec4(scalar));}
    default vec4 mul(vec4 rhs){return mul(this,rhs);}

    static vec4 div(vec4 l, vec4 r) {return vec4(l.x()/r.x(),l.y()/r.y(), l.z()/r.z(),l.w()/r.w());}
    default vec4 div(float scalar) {return div(this, vec4(scalar));}
    default vec4 div(vec4 rhs){return div(this,rhs);}
}
