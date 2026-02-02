package shade;

import optkl.IfaceValue;

public interface vec4Value {

    float x();

    float y();

    float z();

    float w();

    interface Mutable extends vec4Value {
        void x(float x);
        void y(float y);
        void z(float z);
        void w(float w);
        default vec4Value of(float x, float y, float z, float w){
            x(x);y(y);z(z);w(w);
            return this;
        }
        default vec4Value of(vec4Value vec4){
            of(vec4.x(),vec4.y(),vec4.z(),vec4.w());
            return this;
        }
    }

    record Impl(float x, float y, float z, float w) implements vec4Value {
    }


    static vec4Value vec4(float x, float y, float z, float w) {
        return new Impl(x, y, z, w);
    }
    static vec4Value vec4(vec4Value vec4Value) {return vec4(vec4Value.x(),vec4Value.y(), vec4Value.z(), vec4Value.w());}
    static vec4Value vec4(float scalar) {return vec4(scalar,scalar,scalar,scalar);}

    static vec4Value add(vec4Value l, vec4Value r) {return vec4(l.x()+r.x(),l.y()+r.y(), l.z()+r.z(),l.w()+r.w());}
    default vec4Value add(vec4Value rhs){return add(this,rhs);}
    default vec4Value add(float scalar){return add(this,vec4(scalar));}

    static vec4Value sub(vec4Value l, vec4Value r) {return vec4(l.x()-r.x(),l.y()-r.y(), l.z()-r.z(),l.w()-r.w());}
    default vec4Value sub( float scalar) {return sub(this, vec4(scalar));}
    default vec4Value sub(vec4Value rhs){return sub(this,rhs);}

    static vec4Value mul(vec4Value l, vec4Value r) {return vec4(l.x()*r.x(),l.y()*r.y(), l.z()*r.z(),l.w()*r.w());}
    default vec4Value mul( float scalar) {return mul(this, vec4(scalar));}
    default vec4Value mul(vec4Value rhs){return mul(this,rhs);}

    static vec4Value div(vec4Value l, vec4Value r) {return vec4(l.x()/r.x(),l.y()/r.y(), l.z()/r.z(),l.w()/r.w());}
    default vec4Value div( float scalar) {return div(this, vec4(scalar));}
    default vec4Value div(vec4Value rhs){return div(this,rhs);}
}
