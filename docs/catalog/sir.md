# General PINN problem (DA INSERIRE IN FILE SEPARATO)

## ODE - Forward problem
We assume that the unknown variable $\mathbf{u}(t): \ [0,T] \rightarrow \mathbb{R}^n$ satisfies a physical equation expressed as a first order ODE:

$$F(\dot{\mathbf{u}}(t),\mathbf{u}(t))=0$$

with initial conditios (ICs) $\mathbf{u}(0)=\mathbf{u}_0$.

We further assume to have $N_D$ data points, $ {\mathbf{v}}_{1}, \dots , {\mathbf{v}}_{N_D} $ (where ${\mathbf{v}}_i \in \mathbb{R}^m$)  collected in subsequent moments $ t_1, \dots, t_{D}$  (where $t_i\in[0,T]$). Each data point $\mathbf{v}_i$ is considered as a perturbed observation of the unknown variable $\mathbf{u}$ at time $t_i$ through a known observation operator $H$: \mathbb{R}^n \mathbb{R}^m$:

$$\mathbf{v}_i=H(\mathbf{u}(t_i))+\boldsymbol{\epsilon}_i$$

where $\boldsymbol{\epsilon}_i\in \mathbb{R}^m$ is an observation error.

This library applies PINN by approximating $\mathbf{u}$  as a NN $\hat{\mathbf{u}}$ having $N_L$ layers $N_N$ neurons.. The parameters defining $\hat{\mathbf{u}}$ are traiend  with  


# SIR Epidemic Model

```bash
anypinn create my-project --template sir
```

Classic susceptible (S), infected (I), recovered (R) compartmental model.

The inverse problem recovers the value of the transmission rate β.
The data is composed of partial observations of the infected counts (directly linked to the $I$ compartment).

## Problem

Two fields (S, I), one learnable scalar parameter (β), with known recovery rate δ and population N:

$$
\begin{cases}
\frac{dS}{dt} &= -\beta \frac{SI}{N} \\
\frac{dI}{dt} &= \beta \frac{SI}{N} - \delta I
\end{cases}
$$

Reference initial conditions:
$$
\begin{cases}
S(0)=N-1\\
I(0)=1 
\end{cases}
$$

## Features Demonstrated

- Scalar `Parameter` recovery
- `ValidationRegistry` for ground-truth comparison
- `DataScaling` callback for population-scale normalization

## Results

![SIR Inverse results](../examples/sir_inverse/results/sir-inverse.png)
