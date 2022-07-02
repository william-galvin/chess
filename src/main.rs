/// Author:  William Galvin
/// Since:   Summer 2022
/// License: Yes

/// This program plays chess! It is (mostly) compatible with UCI chess programs---compile
/// this .rs file to a .exe file and you're good to go. It is currently unclear at what
/// ELO this program plays, or if it has a name.

use std::io::{self, BufRead, Write};
use std::fs::File;
use chess::{Board, MoveGen, ChessMove, Square, Piece, Color};
use vampirc_uci::{UciMessage, parse_one, UciFen};
use std::str::FromStr;
use std::collections::{HashMap};
use rand::Rng;
use reqwest;
use json;
use nalgebra::{DMatrix, VecStorage};
use tch::CModule;
use tch::Tensor;
use nalgebra::{Matrix};

/// # Macro for making a hashmap
/// Example:
/// ```Rust
/// let map = hashmap!["a" => 1, "b" => "2"];
/// map.get("a") // = 1
/// ```
/// Code from https://stackoverflow.com/questions/28392008/more-concise-hashmap-initialization
#[allow(unused_mut)]
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

#[allow(non_snake_case)]
#[allow(unused_mut)]
/// # Wrapper Class for chess::Board and vampirc_uci
/// Manages the interface with chess GUI frontend and finding the
/// best moves
struct BoardManager {
    id           : String,                  // name of the chess bot (displayed in GUI)
    author       : String,                  // author                (displayed in GUI)
    board        : Board,                   // The current state of the game
    move_stack   : Vec<ChessMove>,          // Moves such that if each are played, current board is reached
    startpos     : String,                  // Start position fen
    DEPTH        : i32,                     // Number of moves to look ahead - default is 5, can be overridden
    LOG_FILE     : File,                    // File to which logs are written - default is "log.txt", can be overridden
    PIECE_WEIGHTS: HashMap<Piece, f64>,     // The values of each of the pieces
    OPENING_BOOK : String,                  // The Lichess opening explorer url
    ENDGAME_TABLE: String,                  // Lichess tablebase url
    in_opening   : bool,                    // Whether or not we should look into the opening book
    lookup_table : HashMap<u64, (i32, f64)>,// <hash, (depth, eval)>
    NN_model     : CModule                  // The model used for NN eval
}

#[allow(unused_must_use)]
#[allow(non_snake_case)]
#[allow(unused_mut)]
impl BoardManager {
    /// # constructor function
    ///
    /// By default, sets board tp Board::default(), move_stack to an empty stack,
    /// DEPTH to 4, LOG_FILE to "log.txt", and piece weights to standard map.
    pub fn new() -> Self {
        Self {
            id: "William's Chess-Bot".to_string(),
            author:"William Galvin".to_string(),
            board: Board::default(),
            move_stack: vec![],
            startpos: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
            DEPTH: 4,
            LOG_FILE: File::create("log.txt").expect("ERROR"),
            PIECE_WEIGHTS: hashmap![Piece::Pawn   => 1.0, 
                                    Piece::Bishop => 3.0, 
                                    Piece::Knight => 3.0, 
                                    Piece::Rook   => 5.0, 
                                    Piece::Queen  => 9.0, 
                                    Piece::King   => 0.0],
            OPENING_BOOK: "https://explorer.lichess.ovh/masters?fen=".to_string(),
            ENDGAME_TABLE: "https://tablebase.lichess.ovh/standard?fen=".to_string(),
            in_opening: true,
            lookup_table: hashmap![],
            NN_model: CModule::load("NN.pt").unwrap()
        }
    }

    /// Reports id to UCI, logs it
    pub fn report_id(&mut self) {
        let output = format!("id name {id}\nid author {author}\noption\nuciok", id = self.id, author = self.author);
        println!("{}", output);
        writeln!(self.LOG_FILE, "{}", output);
    }

    /// Reports ready status, logs it
    pub fn report_ready(&mut self) {
        println!("readyok");
        writeln!(self.LOG_FILE, "readyok");
    }

    /// Initializes board to default, logs it
    pub fn new_game(&mut self) {
        self.board = Board::default();
        writeln!(self.LOG_FILE, "New game -> board initialized to default");
        self.startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string();
        self.in_opening = true;
    }

    /// Sets board to game state specified by incoming UCI position
    /// Move stack is populated after this
    pub fn handle_position(&mut self, startpos: bool, fen: Option<UciFen>, moves: Vec<ChessMove>) {
        self.board = Board::default();
        self.move_stack = vec![];
        if !startpos {
            match fen {
                None => {},
                Some(f) => {
                    let option = Board::from_str(&f.to_string());
                    println!("option: {:?}", option);
                    self.startpos = f.to_string();
                    self.board = option.unwrap();
                    writeln!(self.LOG_FILE, "custom position: {}", &f.to_string());
                }
            }
        }
        for _move in moves {
            self.push_move(_move)
        }

        writeln!(self.LOG_FILE, "new position is: {:?}", self.board.to_string());
    }

    /// searches for the best move from the current position
    /// ignores time and search control
    pub fn go(&mut self) {
        // Try to read from opening book if possible
        if self.in_opening {
            match self.get_opening_move() {
                None => self.in_opening = false,
                Some(_move) => {
                    let move_str = self.check_castling(&_move.to_string());
                    println!("bestmove {}", move_str);
                    writeln!(self.LOG_FILE, "(opening book) bestmove {}", move_str);
                    return
                }
            }
        }

        // Try to read from endgame tablebase if possible
        match self.check_endgame() {
            Some(_move) => {
                let move_str = &_move.to_string();
                println!("bestmove {}", move_str);
                writeln!(self.LOG_FILE, "(endgame tablebase) bestmove {}", move_str);
                return
            },
            _ => {}
        }

        // Use alpha-beta tree seach to score legal moves from position
        // This is the "default" behavior of this function
        let mut moves_and_scores = vec![];
        let mut max = f64::NEG_INFINITY;
        let mut min = f64::INFINITY;
        for _move in MoveGen::new_legal(&self.board) {
            self.push_move(_move);
            let score = self.nega_max(self.DEPTH, f64::INFINITY, f64::NEG_INFINITY);
            if score > max {max = score};
            if score < min {min = score};
            moves_and_scores.push((_move, score));
            self.pop_move();
        }

        // Collect the best moves according to tree search
        // Translate score to avoid negative numbers
        if min < 0f64 {min = -min};
        max += min;
        let mut moves = vec![];
        let THRESHOLD = 0.75;
        for (_move, score) in moves_and_scores {
            let trans_score = score + min;
            if trans_score / THRESHOLD >= max {
                moves.push(_move);
            }
        }

        // Sort the best moves according to NN eval
        let mut scored_moves = self.get_best_move_tournament(moves);
        if self.board.side_to_move() == Color::White {scored_moves.reverse()}
        let best_move = scored_moves.pop().unwrap();

        let m_best = best_move.to_string();

        println!("bestmove {}", m_best);
        writeln!(self.LOG_FILE, "bestmove {}", m_best);
    }

    /// If the current position exists in the Lichess endgame database, returns the best move from
    /// the position. Else returns None.
    fn check_endgame(&mut self) -> Option<ChessMove> {
        let url = format!("{}{}", self.ENDGAME_TABLE, self.board.to_string());
        let result = reqwest::blocking::get(url).unwrap().text();
        let result_string = format!("{:#}", result.unwrap());
        let parsed = json::parse(&result_string).unwrap();
        return if parsed["category"].eq("win") {
            let str = parsed["moves"][0]["uci"].to_string();
            println!("bestmove {}", str);
            Some(ChessMove::from_str(&*str).unwrap())
        } else {
            None
        }
    }

    /// Takes a Vec<(ChessMove, f64)>, and returns a vec<ChessMove>,
    /// sorted by the value from the map
    pub fn sort_moves(&self, mut moves_and_scores: Vec<(ChessMove, f64)>) -> Vec<ChessMove> {
        moves_and_scores.sort_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap());
        let mut moves: Vec<ChessMove> = vec![];
        for (move_, _) in moves_and_scores.iter() {
            moves.push(*move_);
        }
        moves.reverse();
        return moves
    }

    /// To fix buggy UCI notation from lichess
    fn check_castling(&mut self, move_str: &str) -> String {
        match move_str {
            "e1h1" => "e1g1".to_string(),
            "e8h8" => "e8g8".to_string(),
            "e1a1" => "e1c1".to_string(),
            "e8a8" => "e8c8".to_string(),
            _ => move_str.to_string()
        }
    }

    /// Pushes a move onto the stack, updates board
    fn push_move(&mut self, _move: ChessMove) {
        self.move_stack.push(_move);
        self.board = self.board.make_move_new(_move);
    }

    /// Pops a move from the stack, updates board
    fn pop_move(&mut self) {
        self.move_stack.pop();
        let option = Board::from_str(&self.startpos);
        self.board = option.unwrap();
        for _move in &self.move_stack {
            self.board = self.board.make_move_new(*_move);
        }
    }

    /// Returns the list of sorted best moves for the player whose turn it is to move.
    /// Based off of a "tournament" between resulting positions scored with NN evals.
    /// Uses Kenneth Massey's power rankings algorithm.
    /// First move in Vec is what NN thinks best move is--all moves included so they
    /// can be used in move ordering for DFS
    fn get_best_move_tournament(&mut self, moves: Vec<ChessMove>) -> Vec<ChessMove> {
        if moves.len() == 0 {return moves}

        let mut moves_and_tensors = vec![];
        let mut n = 0;
        for _move in moves {
            n += 1;
            self.push_move(_move);
            let tensor = self.to_tensor();
            moves_and_tensors.push((_move, tensor));
            self.pop_move();
        }

        let mut A: Matrix<f64, _, _, VecStorage<f64, _, _>>
           = DMatrix::zeros(n * (n - 1) + 1, n);

        let mut row = 0;
        let mut rhs = vec![];

        for (i, this_item) in moves_and_tensors.iter().enumerate() {
            for (j, other_item) in moves_and_tensors.iter().enumerate() {
                if i == j { continue }
                A[(row, i)] = 1f64;
                A[(row, j)] = -1f64;
                rhs.push(self.compare_positions(&this_item.1, &other_item.1));
                row += 1;
            }
        }

        for i in 0..n {
            A[(row, i)] = 1f64;
        }
        rhs.push(100f64);

        let mut b = DMatrix::from_vec(n * (n - 1) + 1,1, rhs);
        b = A.transpose() * b;
        A = A.transpose() * A;

        let x = A.qr().solve(&b).unwrap();

        let mut moves_and_scores = vec![];
        for (i, item) in moves_and_tensors.iter().enumerate() {
            moves_and_scores.push((item.0, x[i]))
        }

        return self.sort_moves(moves_and_scores)
    }

    /// Implementation of minimax with alpha-beta pruning
    /// Returns a score for the given position
    fn nega_max(&mut self, depth: i32, mut alpha: f64, beta: f64) -> f64 {
        let hash = self.board.get_hash();
        if self.lookup_table.contains_key(&hash) {
            let result = self.lookup_table.get(&hash).unwrap();
            let table_depth = result.0;
            if table_depth >= depth {
                return result.1;
            }
        }

        if self.board.status() != chess::BoardStatus::Ongoing { return self.evaluate(depth) }
        if depth == 0 {
            return self.material_eval();
        }

        let mut min = f64::INFINITY;
        for _move in MoveGen::new_legal(&self.board) {
            self.push_move(_move);
            let score = -1.0 * self.nega_max(depth - 1, -beta, -alpha);
            min = if score < min { score } else { min };
            self.pop_move();
            alpha = if min < alpha { min } else { alpha };
            if alpha <= beta { break }
        }

        self.lookup_table.insert(hash, (depth, min));
        return min;
    }


    /// static evaluation function
    /// Returns val *relative* to side of **player to move**, where larger +number
    /// is better for side to move, and larger -number is better for opponent
    /// Only checks for checkmates and stalemates
    fn evaluate(&self, depth: i32) -> f64 {
        match self.board.status() {
            chess::BoardStatus::Checkmate => { (depth * 999999) as f64 },
            chess::BoardStatus::Stalemate => { 0.0 },
            _ => panic!("ERROR")
        }
    }

    /// Another evaluation function
    /// Returns val *relative* to side of **player to move**, where larger +number
    /// is better for side to move, and larger -number is better for opponent
    /// Checks material
    fn material_eval(&mut self) -> f64 {
        let mut total = 0.0;
        for square in *self.board.combined() {
            let piece = self.board.piece_on(square).unwrap();
            let color = self.board.color_on(square).unwrap();
            total += self.PIECE_WEIGHTS.get(&piece).unwrap() * if color == Color::Black { -1.0 } else { 1.0 };
        }
        return total * if self.board.side_to_move() == Color::Black { 1.0 } else { -1.0 };
    }


    /// Logs ignored UCI message
    pub fn report_ignore(&mut self) {
        writeln!(self.LOG_FILE, "msg ignored");
        return
    }

    /// Reads from an opening book, and returns a ChessMove if there
    /// is an available book move, otherwise returns None
    pub fn get_opening_move(&self) -> Option<ChessMove> {
        let url = format!("{}{}", self.OPENING_BOOK, self.board.to_string());
        let result = reqwest::blocking::get(url).unwrap().text();
        let result_string = format!("{:#}", result.unwrap());
        let parsed = json::parse(&result_string).unwrap();

        let len = parsed["moves"].len();
        return match len {
            0 => None,
            _ => {
                let mut rng = rand::thread_rng();
                let index = rng.gen_range(0..len);
                let _move = parsed["moves"][index]["uci"].to_string();
                let from = Square::from_str(&_move[0..2]);
                let to = Square::from_str(&_move[2..4]);
                Some(ChessMove::new(from.unwrap(), to.unwrap(), None))
            }
        }
    }


    /// Returns array representation of the board, but as a tensor
    pub fn to_tensor(&self) -> Tensor {
        let array = self.to_array();
        let tensor = Tensor::of_slice(&array);
        return tensor.reshape(&[1, 780]);
    }

    /// Returns an array of 780 bits representing the the chess board.
    /// Pretty arbitrary how we got here, but this format is needed to
    /// be used in the neural network evaluation
    fn to_array(&self) -> [f32; 780] {
        let letters = vec!["p", "P", "n", "N", "b", "B", "r", "R", "q", "Q", "k", "K"];

        let mut result = [0.0; 780];

        for square in *self.board.combined() {
            let symbol = self.board.piece_on(square).unwrap().to_string(self.board.color_on(square).unwrap());
            let mut index: i32 = letters.iter().position(|&r| r == symbol).unwrap() as i32;
            index += 12 * square.to_int() as i32;
            result[index as usize] = 1f32;
        }

        let turn = if self.board.side_to_move() == Color::White { 0 } else { 1 };
        result[(64 * 12 + turn) as usize] = 1f32;

        return result;
    }

    /// Takes two tensors and returns a f64.
    /// Larger number -> first position is better for white;
    /// smaller number -> second position is better for white
    fn compare_positions(&mut self, pos1: &Tensor, pos2: &Tensor) -> f64 {
        let input = Tensor::cat(&[pos1, pos2], 1);

        let mut out = 0.0;
        tch::no_grad(|| {
            let output = input.apply(&self.NN_model);
            let v = Vec::<f64>::from(&output);
            out = v[0] - v[1];
        });

        return out;
    }
}

fn main() {
    let mut b = BoardManager::new();
    for line in io::stdin().lock().lines() {
        let msg: UciMessage = parse_one(&line.unwrap());
        writeln!(b.LOG_FILE, "{}", msg).expect("error");
        match msg {
            UciMessage::Uci                               => b.report_id(),
            UciMessage::IsReady                           => b.report_ready(),
            UciMessage::UciNewGame                        => b.new_game(),
            UciMessage::Position {startpos, fen, moves}   => b.handle_position(startpos, fen, moves),
            UciMessage::Go {time_control: _, search_control: _} => b.go(),
            _                                             => b.report_ignore()
        }
    }
}

// TODO:
//      - NN eval function
//              - Where do we actually use the evals? Some ideas:
//                  1. Pure comparison between current position and positions immediately after ("tournament" between all possible positions?)
//                      - Use this to CHOOSE or to ORDER?
//                      - If order, do it with smallest matrix possible
//                      - If choose, need to do a lot more training
//                  2. Pure comparison between current position and positions far after (see deepchess for how to do alpha-beta)
//                  3. Some combination of deep look ahead for material, shallow for comparison
//                      - Only compare moves within some material eval threshold (AFTER doing DFS)?
//      - Figure out iterative deepening?
//      - Move ordering? (Ideally, this should come FROM iterative deepening)
//      - Persistent lookup_table?
//              - (local? cloud?) database for persistent lookup_table?
//              - What does this look like without tree search?
//      - Over the horizon evaluation (quiescence?)
//      - More sophisticated by-hand-eval (pawn structure, king safety)
//      - Set params from GUI?
//      - Think on players time--precompute moves and store in lookup_table?
//          - How do we pick which moves to search for?
//          - Must be able to stop searching INSTANTLY when GO signal comes in--otherwise, not useful at all
//              - Want to use threading; unsure how to (or even if SHOULD) kill thread on new GO
//          - DO TREE SEARCH THINKING HERE
//      - Make it actually UCI compatible
//

