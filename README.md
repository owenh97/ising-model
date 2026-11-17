# 2D Ising Model with Monte Carlo Simulation (Statistical Mechanics)

A from-scratch simulation of the **2D Ising model** on a square lattice, using the **Metropolis algorithm**. It reproduces the real thermodynamic behaviour of a ferromagnet: at low temperature the system spontaneously orders, and at high temperature it dissolves into disorder. Everything here, the lattice, the energy
function, the flip rule, the dynamics — is implemented directly in NumPy with no physics libraries.

## The idea

The model is a grid of "spins," each either `+1` or `-1`. The single rule driving everything: **a spin wants to point the same way as its four neighbours.** Two aligned neighbours are a low-energy (stable) pair; two opposed neighbours are high-energy (tense). Temperature works against this ordering by randomly knocking spins out of alignment.

The competition between alignment (order) and temperature (chaos) is the whole idea, and it produces a genuine phase transition.

## How it works

**Lattice.** An `N × N` grid of `±1`, initialised randomly. The grid uses
**periodic boundary conditions** : the top edge wraps to the bottom and the left to the right, so the lattice behaves like the surface of a torus with no edges.

**Energy of a spin.** For a spin `s` with neighbour sum `S`:

```
E = -s * S
```

Aligned with its neighbourhood → low energy; clashing → high energy.

**The flip decision (Metropolis).** To decide whether a spin flips, compute the energy change it would cause:

```
ΔE = 2 * s * S
```

- If `ΔE ≤ 0`, the flip lowers (or doesn't raise) energy → **always accept it.**
- If `ΔE > 0`, the flip costs energy → **accept it only with probability**
  `p = exp(-ΔE / T)`, by drawing a random number and comparing.

That probabilistic acceptance is where temperature enters. Hot (`T` large) → costly flips happen often → chaos. Cold (`T` small) → costly flips almost neverhappen → the system freezes into order.

**Sweeps.** One *sweep* applies the flip decision to every cell once. The system is evolved by repeating sweeps many times, letting order (or disorder) spread across the lattice.

## Results

All runs use a 50×50 lattice. Yellow and purple represent the two spin states.

### T = 1 — Cold (ordered)

![Ising model at T=1](figures/Figure_1.png)

Almost the whole grid turns one colour. The spins have all lined up with their neighbours. When it's cold enough, spins agree with the ones around them and the entire grid falls into line. This is the orderedstate. The single odd dot is just one spin briefly knocked out of place; even a cold grid has a tiny bit of jitter.

### T = 0.1 — Frozen too fast (stuck)

![Ising model at T=0.1](figures/Figure_2.png)

Strangely, going *colder* makes the order *worse*, not better. Cooled this fast, the grid freezes almost instantly into whatever messy pattern it happened to be in, and gets stuck with two big blocks that can't merge. There isn't enough heat left to let the border between them move, so it never reaches the clean single-colour state it "should" end up in.

It's like metalwork: cool metal slowly and you get a clean crystal; cool it too fast and you lock in the flaws. How you cool matters as much as how cold you end up. (This is the idea behind a method called **simulated annealing**.)

### T = 5 — Hot (disordered)

![Ising model at T=5](figures/Figure_3.png)

The order is gone. You still get little clumps forming and breaking apart, neighbours still tug on each other a bit, but no single colour ever takes over. This is the **disordered** state or paramagnetic phase: it's so hot that the random jostling beats the spins' urge to line up.

Somewhere between cold and hot there's a tipping point, around **T ≈ 2.27** — where the grid flips from ordered to disordered. That exact number is a well-known result (first solved by Onsager).

## Running it

Requirements:

```
pip install numpy matplotlib
```

Run:

```
python ising.py
```

Adjust the lattice size (`size`) and temperature (`T`) at the top of the script to
reproduce the figures above.

## Roadmap

- [ ] Measure **magnetisation vs. temperature** to locate the phase transition and
      see order collapse near T꜀.
- [ ] Train a neural network to **detect the phase transition directly from spin
      configurations** — reproducing Carrasquilla & Melko, *Nature Physics* (2017).

## Notes

Built from scratch as an exercise in implementing, statistical mechanics: lattice with periodic boundaries, local energy, the Metropolis acceptance rule, and Monte Carlo sweeps.