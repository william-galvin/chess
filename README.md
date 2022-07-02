# Chess Programming!
I made a (mostly) UCI-compatible chess engine built with PyTorch and Rust that plays like a very smart third grader and sometimes beats me.

The chess programming was fun, the Rust was headache-inducing, and the fact that the ML worked as well as it did and seemed to sort of understand chess by the end was awe-inspiring.

# Table of contents
  * [Installation and Usage](#installation-and-usage)
  * [Motivation](#motivation)
  * [Project Structure](#project-structure)
  * [Neural Networks](#neural-networks)
    + [Acknowledgements](#acknowledgements)
    + [Overview](#overview)
    + [Chess Position --> Tensor](https://github.com/william-galvin/chess/edit/main/README.md#chess-position-longrightarrow-tensor)
    + [Autoencoder](#autoencoder)
    + [Inputs, Outputs, and Architecture](https://github.com/william-galvin/chess/edit/main/README.md#chess-position-longrightarrow-tensor)
  * [Chess Server](#chess-server)
    + [UCI](#uci)
    + [Nega-Max](#nega-max)
    + [Evaluations](#evaluations)
    + [Using the Model](#using-the-model)
    + ["Tournament" Method](https://github.com/william-galvin/chess/edit/main/README.md#tournament-method)
    + [Opening Book and Endgame Tablebase](#opening-book-and-endgame-tablebase)
  * [Reflections](#reflections)
  * [Next Steps / TODOs](https://github.com/william-galvin/chess/edit/main/README.md#next-steps--todos)

## Installation and Usage
This repo is not meant to be immediately usable in its current form. Theoretically, you could download `NN.pt` and `chess_bot.exe` and direct a chess GUI to the `.exe` and it *might* work.

## Motivation
I wanted to start learning several things, all at once, with very little prior experience in any of them: Rust, Machine Learning, and being good at chess. I also had a $300 Google Cloud credit to spend, and what better way to do that than spending weeks training the same models over and over again, anxiously tweaking the hyperparameters?

## Project Structure
There are two main components to this project. The first is the actual chess-playing logic, which is in `src/main.rs`. (More on that below.)

The second, larger but arguably less interesting, part is the neural networks written in PyTorch. If it looks suspiciously like the code in `data` was lifted directly from tutorials and forums and whatnot, it's probably because it was. Many of the design choices in this section are arbitrary, too—if you ask me, *Why are there x layers instead of y?* Or, *How did you pick that activation function?* Or, *What's up with specifically 15 epochs?* I won't have a good answer.

## Neural Networks
### Acknowledgements
Thank you to the author's of [DeepChess](https://www.researchgate.net/publication/306081185_DeepChess_End-to-End_Deep_Neural_Network_for_Automatic_Learning_in_Chess), the paper that provided a roadmap for my program. By roadmap, I mean I basically implemented the whole thing, just not as well.

I would also like to specifically thank the maintaners of the unofficial [Lichess Master's Database](https://database.nikonoel.fr/), from which my data was sourced.

And of course, every kind (and not so kind) soul that's ever answered any question on any forum (yes, I mean *all of them*).

### Overview
The big idea behind the neural network component of this project was this: If I had a giant database full of chess positions annoted with who won the game from which the position came, I could then train a NN to look at two positions, and tell me which one white was more likely to win. (This is dircetly from DeepChess.)

Then, that could be used to order some or all of the legal moves from a position *where we don't know who's going to win because we're in the middle of the game*.

### Chess Position $\longrightarrow$ Tensor
To transform a chess position into a usable input for a neural network, I represented the board as a 780 $\times$ 1 vector, where each entry is a bit (1 or 0) and is determined as follows: 

For each square (starting at A1 and ending at H8), create a 12 $\times$ 1 zero-tensor, $T$. If the square is occupied, consider:

| Piece        | Index |
| ------------ | ----- |
| White Pawn   | 0     |
| Black Pawn   | 1     |
| White Knight | 2     |
| Black Knight | 3     |
| White Bishop | 4     |
| Black Bishop | 5     |
| White Rook   | 6     |
| Black Rook   | 7     |
| White Queen  | 8     |
| Black Queen  | 9     |
| White King   | 10    |
| Black King   | 11      |

And let 
$T_{index} \leftarrow 1$
. Then append 
$T$
to the previous 
$T$
and continue. 

Then, if the position is white-to-move, append 
$\left[ \begin{array}{c}  1 & 0 & 0 & \dots & 0 \end{array} \right]$
, else append 
$\left[ \begin{array}{c}  0 & 1 & 0 & \dots & 0 \end{array} \right]$.

In code: 
```Rust
fn to_array(&self) -> [f32; 780] {
	let letters = vec!["p", "P", "n", "N", "b", "B", "r", "R", "q", "Q", "k", "K"];

	let mut result = [0.0; 780];

	for square in *self.board.combined() {
		let symbol =
			self.board.piece_on(square)
			.unwrap()
			.to_string(self.board.color_on(square)
				.unwrap());
		let mut index: i32 = letters.iter().position(|&r| r == symbol).unwrap() 
			as i32;
		index += (12 * square.to_int() as i32);
		result[index as usize] = 1f32;
	}

	let turn = if self.board.side_to_move() == Color::White { 0 } else { 1 };
	result[(64 * 12 + turn) as usize] = 1f32;
	
	return result;
}
```

### Autoencoder
Shockingly, the above encodings don't work particularly well for training a NN. Something about ✨discontinuity✨ or something. So the first step was to pre-train an autoencoder to compress and then decompress a tensor representation of a position from 780 numbers to 600, then from 600 to 400, then 400 to 200, and finally 200 to 100. 

This is supposed to automate *feature extraction* and lets us have a tensor with only 100 values in our actual input.

### Inputs, Outputs, and Architecture
I started with lots of chess positions and annotations for whether or not white eventually won their games. I inputted two at a time into the NN, where one had white win and one had black win, and the output was 0 if white won the first position and 1 if white won the second position.

The architecture of the model was copied, with slight variation, from DeepChess. It was trained on 10,000,000 pairs of positions for 100 epochs.


## Chess Server
### UCI
The big idea with UCI is that a chess GUI can talk to a chess engine via predefined phrases in standard IO. For example, if the GUI sends `go`, then the chess engine should send back the best move in the position, like `bestmove e2e4`, which the GUI will then play on behalf of the engine.

This is very helpful because it means I don't have to be good at front end things. I used the [vampirc_uci](https://docs.rs/vampirc-uci/latest/vampirc_uci/) crate to help with this. In code, using UCI looks like:
```Rust
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
```
Where each function uses `writeln!()` to communicate with the GUI.

### Nega-Max
To do the tree-search component, I used an implementation of $\alpha$-$\beta$ search called [nega-max](https://en.wikipedia.org/wiki/Negamax#:~:text=Negamax%20search%20is%20a%20variant,the%20value%20to%20player%20B.). I used both pruning and a hashmap-based lookup table to improve search speed. 

Code for this is in `fn nega_max` and `pub fn go`.

### Evaluations
One of the assumptions that $\alpha$-$\beta$ search relies on is that for any given position, we have some halfway-decent heuristic for evaluating how good the position is for either side.

If the position is a checkmate, it's pretty easy. We can just return `depth * 999999` which is clever in two ways: first we don't have to worry about whose turn it is, since a player can't have put themselves in checkmate the previous turn; and second, using `depth * 999999` instead of some more rustic representation of $\infty$ because this will prioritize quicker checkmates.

For positions in which the game isn't over, the simplest possible evaluation function is to just count material, which is what I ended up doing. Crafting (or trying to) a handmade eval function was not only tedious and made the program execute slower, but also didn't always make for a better chess program. But in future iterations, this could be a low-hanging way to add performance.

### Using the Model
Instead of personally spending hours perfecting a handmade eval function, I tried to let the NN handle that. Good eval functions look at not just material, but also positional advantages, tempo, and other fancy chess things, and that's theoretically what the NN was learning to recognize as it trained.

To actually use the model in Rust, I used the [tch](https://docs.rs/tch/latest/tch/) crate:

```Rust
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
```

But there's still a problem: I wanted a function that evaluated *one* position, but this one compares *two*.

### "Tournament" Method
This is probably the most significant diversion between my program and DeepChess. While they use the comparison directly in their tree-search, I do a non-NN-based tree-search, and then use the results to find the best candidate moves.

Once I have a subset of the legal moves,which according to negamax and the pimitive but fast eval function are the best moves, I can use the NN to compare them to each other to find the best move. The hope is that in this way, "best" can be more about positional advantages than about material. 

Letting $f(a, b)$ be the NN comparison between positions $a$ and $b$, we should note that the model is not interiely commutative (it slightly prefers white in objectively equal positons): $f(a, b) \neq -f(b, a)$. It follows therein that if $f(a, b) \longrightarrow$ a > b and  $f(b, c) \longrightarrow$ b > c, it is **not necessarily true** that  $f(a, c) \longrightarrow$ a > c.

This reminded me of power rankings in sports leagues, where the Yankees can beat the Red Sox, the Red Sox can beat the Cubs, and the Cubs can beat the Yankees. How do we pick the best team? [Kenneth Massey](https://en.wikipedia.org/wiki/Kenneth_Massey) devised a technique that I adapted and will briefly descibe below:

Let $P_{1} \dots P_{n}$ be the positions resulting from making the moves 1 . . . n, and $f(a, b)$ be the neural network function described above, then create the matrices as follows:

$$
\begin{array} {rcl}
P_1 - P_2 & = & f(P_1, P_2) \\
P_1 - P_3 & = & f(P_1, P_3) \\
& \vdots & \\
P_1 - P_n & = & f(P_1, P_n) \\
P_2 - P_1 & = & f(P_2, P_1) \\
P_2 - P_3 & = & f(P_2, P_3) \\
P_2 - P_n & = & f(P_2, P_n) \\
& \vdots & \\
P_n - P_{n-1} & = & f(P_{n}, P_{n-1}) \\
\end{array}
\longrightarrow
\left( \begin{array} {r}
1 & -1 & 0 & \dots & 0 \\
1 & 0 & -1 & \dots & 0 \\
\vdots \\
1 & 0 & 0 & \dots & -1 \\
-1 & 1 & 0 & \dots & 0 \\
0 & 1 & -1 & \dots & 0 \\
\vdots \\
0 & 1 & 0 & \dots & -1 \\
\vdots \\
\vdots \\
0 & 0 & \dots & -1 & 1 \\
\end{array} \right) 
\left( \begin{array} {c} P_1 \\ P_2 \\ \dots \\ P_n \end{array} \right)^{T}=\left(=\begin{array} {c}
f(P_1, P_2) \\
f(P_1, P_3) \\
\vdots & \\
f(P_1, P_n) \\
f(P_2, P_1) \\
f(P_2, P_3) \\
f(P_2, P_n) \\
 \vdots  \\
f(P_{n}, P_{n-1}) \\
\end{array} \right)
$$


Letting 
$A$ 
equal the 
$n(n - 1) \times n$ 
matrix on the left, 
$x$
equal 
$\left( \begin{array} {c} P_1 & P_2 & \dots & P_n \end{array} \right)^T$
and 
$b$
equal the right hand side vector, then we can solve the over determined system of equations by taking $A^TAx = A^Tb$ and solving for $x$ using QR-decomposition (or any other method you like). 

(But to determine a unique solution, first append $\left( \begin{array} {c} 1 & 1 & \dots & 1 \end{array} \right)$  to $A$ and any arbitrary constant to $b$ .)

The resulting vector 
$x$
, which was 
$\left( \begin{array} {c} P_1 & P_2 & \dots & P_n \end{array} \right)^T$
, gives us the *relative scores* of each position $P$, such that $x_i$ is the score for $P_i$, and finding the maximum $P$ gives us the best move. 

The code for this is the extraordinarily long function called `fn get_best_move_tournament`

### Opening Book and Endgame Tablebase
In a moment of weakness and desire for this chess bot (my son, my only son) to crush its opponents and not play garbage openings and lose completely winning endgames, it reads from the Lichess opening and closing books. Stop looking at me like that, it's *not* cheating!

## Reflections
There's no way to get around it: this is not a world-class chess program. But, for the most part, it makes halfway decent moves and doesn't hang pieces. And I think that is incredible that a neural network can pick those moves, especially one that *I* made, because I have no idea what I'm doing.

## Next Steps / TODOs
- Make the program *actually* UCI-compatible
- Add functionality to play on Lichess
	- [https://github.com/ShailChoksi/lichess-bot](https://github.com/ShailChoksi/lichess-bot) could be helpful
	- Ultimately only worth it if I can have it run indefinitely on the cloud
- Implement iterative deepening—theoretically, this shouldn’t be too hard, but in all my attempts, has never improved search times
- Move ordering in $\alpha$-$\beta$ search
	- If not from IDDFS, then potentially use the NN eval function? (Would need a way to do the eval faster; see DeepChess paper)
- Persistent lookup table
- Over-the-horizon (quiescence) search
- More refined manual-static evaluation function
	- King safety + pawn structure would be a decent place to start
- Pondering on opponent's time
