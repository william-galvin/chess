# {Title Goes Here}
{Title} is a UCI-compatible chess engine built with PyTorch and Rust that plays[link to lichess] at {elo here}.

## Table of Contents
Somehow link to each section in this document

## Installation and Usage
Some fairly detailed installation instructions? Not 100% sure what this looks like

## Motivation
Why did I make this? Some things to include:
- Needed to spend my Google cloud credits
- Thought it might make me a better chess player (it didn't)
- Wanted to learn Rust, *definetly* know what the borrow checker is

## Training - Everything Lifted From Deep Chess
With details on how implementations were done:
- Form of the Data
	- Explanation & diagram of chess position --> Tensor
	- Where did the data come from?
- Diagram of the architecture
- Training methods (pre-training autoencoder layers)
- Whole training pipeline

## Using the Model
- Getting PyTorch model into Rust
	- Code snippits!
- Turning the model output into a useable number

## "Tournament" method
- How does DeepChess do it? Why does this not work for me?
- The non-transiitve/commutiative property of the model
	- $f(a, b) \neq -f(b, a)$
	- $f(a, b) \longrightarrow$ a > b;  $f(b, c) \longrightarrow$ b > c; $f(a, c) \longrightarrow$ a < c
- What real-world siutation does this remind you of? Sports & power rankings!

Where $P_{1} \dots P_{n}$ are the positions resulting from making the moves 1 . . . n, and $f(a, b)$ is the neural network function described above, then

$\begin{array} {rcl}
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

\left( \begin{array} {c}
P_1 \\ P_2 \\ \vdots \\ P_n
\end{array} \right) 
=
\left(
\begin{array} {c}
f(P_1, P_2) \\
f(P_1, P_3) \\
\vdots & \\
f(P_1, P_n) \\
f(P_2, P_1) \\
f(P_2, P_3) \\
f(P_2, P_n) \\
 \vdots  \\
f(P_{n}, P_{n-1}) \\
\end{array} \right)$

Letting $A$ equal the $n(n - 1) \times n$ matrix on the left, $x$ equal $\left( \begin{array} {c} P_1 & P_2 & \dots & P_n \end{array} \right)^T$  and $b$ equal the right hand side vector, then we can solve the over determined system of equations by taking $A^TAx = A^Tb$ and solving for $x$ using QR-decomposition (or any other method you like). 

```Rust
let x = A.qr().solve(&b).unwrap();
let mut moves_and_scores = vec![];
for (i, item) in moves_and_tensors.iter().enumerate() {
	moves_and_scores.push((item.0, x[i]))
}

let mut result = self.sort_moves(moves_and_scores);
result.reverse();

return result;
```

To determine a unique solution, first append $\left( \begin{array} {c} 1 & 1 & \dots & 1 \end{array} \right)$  to $A$ and any arbitrary constant to $b$ 
```Rust
for i in 0..n {
	A[(row, i)] = 1f64;
}
rhs.push(100f64);
```

The resulting vector $x$, which was $\left( \begin{array} {c} P_1 & P_2 & \dots & P_n \end{array} \right)^T$, gives us the *relative scores* of each position $P$, such that $x_i$ is the score for $P_i$, and finding the maximum $P$ gives us the best move. 
## Next Steps / TODOs
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
