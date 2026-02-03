package shade;

public interface vec3 {
    float x();

    float y();

    float z();

    // A mutable variant needed for interface mapping
    interface Field extends vec3 {
        void x(float x);
        void y(float y);
        void z(float z);
        default vec3 of(float x, float y, float z){
            x(x);y(y);z(z);
            return this;
        }
        default vec3 of(vec3 vec3){
            of(vec3.x(),vec3.y(),vec3.z());
            return this;
        }
    }
    record Impl(float x, float y, float z) implements vec3 {
    }

    static vec3 vec3(float x, float y, float z) {
        return new Impl(x, y, z);
    }

    static vec3 vec3(vec3 vec3) {return vec3(vec3.x(), vec3.y(), vec3.z());}
    static vec3 vec3(float scalar) {return vec3(scalar,scalar,scalar);}

    static vec3 add(vec3 l, vec3 r) {return vec3(l.x()+r.x(),l.y()+r.y(), l.z()+r.z());}
    default vec3 add(vec3 rhs){return add(this,rhs);}
    default vec3 add(float scalar){return add(this,vec3(scalar));}

    static vec3 sub(vec3 l, vec3 r) {return vec3(l.x()-r.x(),l.y()-r.y(), l.z()-r.z());}
    default vec3 sub(float scalar) {return sub(this, vec3(scalar));}
    default vec3 sub(vec3 rhs){return sub(this,rhs);}

    static vec3 mul(vec3 l, vec3 r) {return vec3(l.x()*r.x(),l.y()*r.y(), l.z()*r.z());}
    default vec3 mul(float scalar) {return mul(this, vec3(scalar));}
    default vec3 mul(vec3 rhs){return mul(this,rhs);}

    static vec3 div(vec3 l, vec3 r) {return vec3(l.x()/r.x(),l.y()/r.y(), l.z()/r.z());}
    default vec3 div(float scalar) {return div(this, vec3(scalar));}
    default vec3 div(vec3 rhs){return div(this,rhs);}
}
