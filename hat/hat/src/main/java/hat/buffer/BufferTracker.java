package hat.buffer;

public interface BufferTracker {

     void preMutate(Buffer b);

     void postMutate(Buffer b) ;

     void preAccess(Buffer b);

     void postAccess(Buffer b);

     void preEscape(Buffer b);

     void postEscape(Buffer b) ;

}
