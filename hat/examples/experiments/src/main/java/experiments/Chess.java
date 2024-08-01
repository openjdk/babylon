package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.ChessState;
import hat.buffer.S32Array;

import java.lang.runtime.CodeReflection;

public class Chess {
    /*
    See how small you can pack board/game state.
    My guess is around 96 bits + castle status (whether king rook(s) have moved since game start)
    Create a function that can create a list of ‘legal’ moves from any ‘game state’
    Create a ‘score function’.  So simplest Pawn = 1, Knight=2, Bishop=3., Rook=4, Queen=6, King = infinity :wink:. Negative values for Black pieces.
    So you can turn your game state into a value.
    So get all the legal moves possible moves from state and determine the ‘score’ for each legal move.
    For each of these.  Repeat above (creating legal opposition moves for each legal move)  We call this a ‘ply’
    Do as many ply’s as you have memory for.  You have lots of GPU memory.  So maybe three plys.  That would beat a novice play (ELO rating up to 800 prob)
    At the leaves of this tree add all ‘white’ scores from the leaf to the root state and subtract the ‘black’ scores.
    Whichever root move leads to the highest score for white (’take it)
    Repeat… the above ;)
    Looks like you need 4 bits to encode a piece (including its color) so ‘0000’ empty ‘0001’ white pawn., 1001 black pawn  etc (edited)
    Sp 4bits * 64 spaces.  Oh thats way more than 96 bits… I meant bytes

    So a board/state  would just start with bits for RBNKQNBRPPPPPPPEEEEEEEEEEEEEEEEPPPPPPPPPRBNBKQNBR + bits for Wcancastle Bcancastle Wcheck Bcheck about 67 bytes
    Then add maybe an int index to the start of the ‘possible states’ + ‘char length of possible states’ from moves in list of states.
    So all of the gamestates will just be in one huge array of gamestate structs  with links from/to children.

    Thinking more.  Make your ‘list of possible moves’ fixed size. Say a max of 10.  Then as you evaluate the local score of each.  For move 11 if it has a higher score than the top 10 replace the lowest with this new one.
    Without this the problem is hard to do in parallel.
    It also means you can predict the size of each ply.

    Yes the kernel will take the index to a board state (now).  And to a an area to populate its 10 selection .
    It will iterate through possible legal moves (non-trivial actually)  ‘local score it’ (meaning what is the value after this one move.) and store that with the board states at the list of 10 to populate.
    The problem with this strategy is that your player will possibly never trade pieces for tactical advantage. Hmmm
    Lots of options though. Say keeping best  8 and worst 2 (locally losing in the next move) will be worst ;)
    */

    // using PeSTO (https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function)
    // and progszach piece-square tables (https://www.chessprogramming.org/Simplified_Evaluation_Function)
    // and rustic (https://rustic-chess.org/search/ordering/how.html)

    static int [] pEval = {
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    static int [] nEval = {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    };

    static int [] bEval = {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    };

    static int [] rEval = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    };

    static int [] qEval = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    };

    static int [] kMidEval = {
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    };

    static int [] kEndEval = {
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    };

    static int [] pieceVal = {100, 300, 300, 500, 900, 0};
    static int [][] psqt = {pEval, nEval, bEval, rEval, qEval, kMidEval};

    @CodeReflection
    public static int eval(ChessState board) {
        int blk = 0;
        int wht = 0;
        for (int i = 0; i < 64; i++) {
            int piece = board.array(i);
            if (piece == 0) continue;
            boolean white = piece > 0;
            piece = Math.abs(piece);
            if (white) {
                wht += psqt[piece - 1][i];
            } else {
                blk += psqt[piece - 1][8 * (7 - (i / 8)) + (i % 8)];
            }
        }
        return wht - blk;
    }

    @CodeReflection
    public static void alphaBeta(KernelContext kc, hat.buffer.ChessState board, hat.buffer.S32Array data) {
        if (kc.x<kc.maxX){
            data.array(0, eval(board));
        }
    }

    @CodeReflection
    public static void compute(ComputeContext cc, hat.buffer.ChessState board, hat.buffer.S32Array data) {
        cc.dispatchKernel(64,
                kc -> alphaBeta(kc, board, data)
        );
    }

    public static void main(String[] args) {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);//new JavaMultiThreadedBackend());
        var board = ChessState.create(accelerator, 64);
        for (int i = 0; i < 64; i++) {
            int row = i / 8;
            int col = i % 8;
            if (row == 0) {
                if (col == 4) {
                    board.array(i, (byte) -6);
                } else if (col == 3) {
                    board.array(i, (byte) -5);
                } else if (col == 2 || col == 5) {
                    board.array(i, (byte) -3);
                } else if (col == 1 || col == 6) {
                    board.array(i, (byte) -2);
                } else {
                    board.array(i, (byte) -4);
                }
            } else if (row == 7){
                if (col == 4) {
                    board.array(i, (byte) 6);
                } else if (col == 3) {
                    board.array(i, (byte) 5);
                } else if (col == 2 || col == 5) {
                    board.array(i, (byte) 3);
                } else if (col == 1 || col == 6) {
                    board.array(i, (byte) 2);
                } else {
                    board.array(i, (byte) 4);
                }
            } else if (row == 1) {
                board.array(i, (byte) -1);
            } else if (row == 6) {
                board.array(i, (byte) 1);
            }else {
                board.array(i, (byte) 0);
            }
        }
        board.bCanCastle(true);
        board.wCanCastle(true);
        var data = S32Array.create(accelerator, 1);
        accelerator.compute(
                cc -> Chess.compute(cc, board, data)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        System.out.println(data.array(0));
        char[] whtPieces = new char []{' ', '♙', '♘', '♗', '♖', '♕', '♔'};
        char[] blkPieces = new char []{' ', '♟', '♞', '♝', '♜', '♛', '♚'};
        for (int i = 0; i < 64; i++) {
            if (i % 8 == 0) System.out.print((71 - i) / 8 + " ");
            if (board.array(i) < 0) {
                System.out.print(blkPieces[-board.array(i)]);
            } else {
                System.out.print(whtPieces[board.array(i)]);
            }
            if (i % 8 == 7) {
                System.out.println();
            } else {
                System.out.print(" ");
            }
        }
        System.out.println("  A B C D E F G H");
    }
}
