/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.ChessState;

import java.util.Scanner;
import java.lang.runtime.CodeReflection;
import java.util.*;

public class Chess {

    /*
        referencing PeSTO (https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function)
        and progszach piece-square tables (https://www.chessprogramming.org/Simplified_Evaluation_Function)
        and rustic (https://rustic-chess.org/search/ordering/how.html)

        no en passant or moving pawns by 2 spaces for now
        move by entering the piece you want to move, the starting position, and the ending position
        e.g. pe2e3 or ng1f3

        Chess board layout (idx):
        8 |  0,  1,  2,  3,  4,  5,  6,  7,
        7 |  8,  9, 10, 11, 12, 13, 14, 15,
        6 | 16, 17, 18, 19, 20, 21, 22, 23,
        5 | 24, 25, 26, 27, 28, 29, 30, 31,
        4 | 32, 33, 34, 35, 36, 37, 38, 39,
        3 | 40, 41, 42, 43, 44, 45, 46, 47,
        2 | 48, 49, 50, 51, 52, 53, 54, 55,
        1 | 56, 57, 58, 59, 60, 61, 62, 63
            A   B   C   D   E   F   G   H
     */

    static final int N = -8;
    static final int S = 8;
    static final int E = 1;
    static final int W = -1;
    static final int NE = -7;
    static final int NW = -9;
    static final int SE = 9;
    static final int SW = 7;

    // moveset for each piece
    static int [] pMoves = {N};
    static int [] nMoves = {N + NE, N + NW, S + SE, S + SW, E + NE, E + SE, W + NW, W + SW};
    static int [] bMoves = {NE, NW, SE, SW};
    static int [] rMoves = {N, E, S, W};
    static int [] qMoves = {N, E, S, W, NE, NW, SE, SW};
    static int [] kMoves = {N, E, S, W, NE, NW, SE, SW};

    // piece squares tables for finding best position for each piece
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

    static final int [] nEval = {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    };

    static final int [] bEval = {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    };

    static final int [] rEval = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    };

    static final int [] qEval = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    };

    static final int [] kMidEval = {
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    };

    // was going to implement later; piece square table for endgame king movement
    static final int [] kEndEval = {
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    };

    static int [][] moves = {new int[0], pMoves, nMoves, bMoves, rMoves, qMoves, kMoves};
    static int [][] psqt = {new int[0], pEval, nEval, bEval, rEval, qEval, kMidEval};
    static List<Character> pieces = new ArrayList<>(Arrays.asList(' ', 'p', 'n', 'b', 'r', 'q', 'k'));

    public static int boardIdx(int idx, boolean white) {
        return (white) ? idx : 8 * (7 - (idx / 8)) + (idx % 8);
    }

    // score the board using the piece squares tables
    public static int evalBoard(ChessState board) {
        int blk = 0;
        int wht = 0;
        for (int i = 0; i < 64; i++) {
            int piece = board.array(i);
            if (piece == 0) continue;
            boolean white = piece > 0;
            piece = Math.abs(piece);
            if (white) {
                wht += psqt[piece][boardIdx(i, white)];
            } else {
                blk += psqt[piece][boardIdx(i, white)];
            }
        }
        return wht - blk;
    }

    // score the board only based on the current move (no iterating through the board)
    public static int evalMove(ChessState board, int piece, int start, int end) {
        int score = board.score();
        boolean white = piece > 0;
        piece = Math.abs(piece);
        score += psqt[piece][boardIdx(end, white)] - psqt[piece][boardIdx(start, white)];

        int nextSpace = Math.abs(board.array(end));
        if (nextSpace > 0) {
            score += psqt[nextSpace][boardIdx(end, white)];
        }
        return score;
    }

    // check if movement goes off the board
    public static boolean inBounds(int start, int end) {
        return (end >= 0 && end <= 63
                && (start % 8) + (end % 8) > 0 && (start % 8) + (end % 8) < 8
                && (start / 8) + (end / 8) > 0 && (start / 8) + (end / 8) < 8);
    }

    // is the current player in check?
    public static boolean inCheck(ChessState board) {
        int king = 0;

        // determines whose turn it is
        boolean white = board.turn() % 2 == 0;

        // locate the king
        for (int i = 0; i < 64; i++) {
            if (board.array(i) == ((white) ? 6 : -6)) {
                king = i;
                break;
            }
        }

        // scan all rays from king
        for (int dir : moves[5]) {
            int nextSquare = king;
            while (inBounds(king, nextSquare + dir)) {
                // if king is reachable by opponent's piece, return true
                if ((white && board.array(nextSquare + dir) < 0) || (!white && board.array(nextSquare + dir) > 0)) {
                    int opponentPiece = Math.abs(board.array(nextSquare + dir));
                    if (opponentPiece == 5
                            || (opponentPiece == 3 && (Math.abs(dir) == 1 || Math.abs(dir) == 8))
                            || (opponentPiece == 4 && (Math.abs(dir) == 7 || Math.abs(dir) == 9))
                            || (opponentPiece == 1 && ((white && king - nextSquare == 8) || (!white && king - nextSquare == -8)))) {
                        return true;
                    }
                }
                if (board.array(nextSquare + dir) == 0) {
                    nextSquare += dir;
                } else {
                    break;
                }
            }
        }

        for (int dir : moves[2]) {
            if (inBounds(king, king + dir)
                    && ((white && board.array(king + dir) < 0) || (!white && board.array(king + dir) > 0))
                    && Math.abs(board.array(king + dir)) == 2) return true;
        }
        return false;
    }

    // get the next move that doesn't put us in check
    public static int[] getNextMove(ChessState board) {
        int[] nextMove = new int[2];
        // stores previous location
        nextMove[0] = -1;
        // stores next location
        nextMove[1] = -1;
        int maxScore = board.score();
        boolean white = board.turn() % 2 == 0;
        for (int i = 0; i < 64; i++) {
            int piece = board.array(i);

            // if we run into an ally piece, move on
            if ((white && piece <= 0) || (!white && piece >= 0)) continue;

            // go through all rays of movement
            for (int dir : moves[Math.abs(piece)]) {
                if (!white) dir = -dir;
                int nextSquare = i + dir;
                int replacedVal;
                while (inBounds(i, nextSquare)
                        && ((white && board.array(nextSquare) <= 0) || (!white && board.array(nextSquare) >= 0))) {
                    replacedVal = board.array(nextSquare);
                    // do the move
                    board.array(i, (byte) 0);
                    board.array(nextSquare, (byte) piece);
                    if (!inCheck(board)) {
                        int score = evalMove(board, piece, i, nextSquare);
                        if (score > maxScore || nextMove[0] == -1) {
                            maxScore = score;
                            nextMove[0] = i;
                            nextMove[1] = nextSquare;
                        }
                    }
                    // undo move
                    board.array(i, (byte) piece);
                    board.array(nextSquare, (byte) replacedVal);
                    if (Math.abs(piece) < 3 || Math.abs(piece) > 5 || board.array(nextSquare) != 0) break;
                    nextSquare += dir;
                }
            }
        }
        return nextMove;
    }

    public static void move(ChessState board, int start, int end) {
        int piece = board.array(start);
        board.array(start, (byte) 0);
        board.array(end, (byte) piece);
    }

    @CodeReflection
    // was supposed to be an alpha beta tree
    public static void alphaBeta(KernelContext kc, hat.buffer.ChessState board) {
        if (kc.x<kc.maxX){
            int[] nextMove = getNextMove(board);
            if (nextMove[0] < 0) {
                System.out.println("white wins!");
            } else {
                move(board, nextMove[0], nextMove[1]);
                board.turn(board.turn() + 1);
            }
        }
    }

    @CodeReflection
    public static void compute(ComputeContext cc, hat.buffer.ChessState board) {
        cc.dispatchKernel(1,
                kc -> alphaBeta(kc, board)
        );
    }

    public static void main(String[] args) {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);//new JavaMultiThreadedBackend());
        var board = initBoard(accelerator);
        printBoard(board);

        Scanner scanner = new Scanner(System.in);
        String str = "";
        int piece = 0;
        int start = -1;
        int end = -1;
        boolean validInput = false;
        boolean checkmate = false;

        while (!checkmate) {
            while (!validInput) {
                System.out.println("Your move : ");
                str = scanner.nextLine().toLowerCase();
                if (!("pnbrqk".contains(String.valueOf(str.charAt(0)))
                        && Character.isLetter(str.charAt(1))
                        && str.charAt(1) > 96
                        && str.charAt(1) < 105
                        && Character.isDigit(str.charAt(2))
                        && str.charAt(2) > 48
                        && str.charAt(2) < 57
                        && Character.isLetter(str.charAt(3))
                        && str.charAt(3) > 96
                        && str.charAt(3) < 105
                        && Character.isDigit(str.charAt(4))
                        && str.charAt(4) > 48
                        && str.charAt(4) < 57)) {
                    System.out.println("Invalid input!");
                } else {
                    piece = pieces.indexOf(str.charAt(0));
                    start = ((str.charAt(1) - 97) + (7 - (str.charAt(2) - 49)) * 8);
                    if (Math.abs(board.array(start)) != piece) {
                        System.out.println("Invalid input!");
                        continue;
                    }
                    end = ((str.charAt(3) - 97) + (7 - (str.charAt(4) - 49)) * 8);
                    for (int i : moves[piece]) {
                        if (i == end - start || (piece >= 3 && piece <= 5 && ((end - start) & i) == 0)) {
                            validInput = true;
                            break;
                        }
                    }
                    if (!validInput) System.out.println("Invalid input!");
                }
            }

            board.score(evalMove(board, piece, start, end));
            move(board, start, end);

            if (inCheck(board) && getNextMove(board)[0] < 0) {
                System.out.println("black wins!");
                checkmate = true;
            } else {
                board.turn(board.turn() + 1);
                printBoard(board);
                accelerator.compute(
                        cc -> Chess.compute(cc, board)  //QuotableComputeContextConsumer
                );                                     //   extends Quotable, Consumer<ComputeContext>
                System.out.println();
                printBoard(board);
                validInput = false;
            }
        }
        System.out.println("game over");
    }

    private static ChessState initBoard(Accelerator accelerator) {
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
        board.score(evalBoard(board));
        return board;
    }

    public static void printBoard(ChessState board) {
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
