# General PINN problem (DA INSERIRE IN FILE SEPARATO)

## ODE - Forward problem
We assume that the unknown variable $\mathbf{u}(t): \ [0,T] \rightarrow \mathbb{R}^n$ satisfies a physical equation expressed as a first order ODE:

$$F(\dot{\mathbf{u}}(t),\mathbf{u}(t))=0$$

with initial conditions (ICs) $\mathbf{u}(0)=\mathbf{u}_0$.

We further assume to have $N_D$ data points, $ {\mathbf{v}}_{1}, \dots , {\mathbf{v}}_{N_D} $ (where ${\mathbf{v}}_i \in \mathbb{R}^m$)  collected in subsequent moments $ t_1, \dots, t_{D}$  (where $t_i\in[0,T]$). Each data point $\mathbf{v}_i$ is considered as a perturbed observation of the unknown variable $\mathbf{u}$ at time $t_i$ through a known observation operator $H$: \mathbb{R}^n \mathbb{R}^m$:

$$\mathbf{v}_i=H(\mathbf{u}(t_i))+\boldsymbol{\epsilon}_i$$

where $\boldsymbol{\epsilon}_i\in \mathbb{R}^m$ is a possible observation error.

AnyPINN approximates (??) each component of (??) $\mathbf{u}(t)$ as a neural network $\hat{\mathbf{u}}(t,\boldsymbol{theta})$, where $\boldsymbol{theta}\in \mathbb{R}^{N_L}$ are the free parameters defined in the $N_L$ layers $N_N$ neurons of the NN. 

$\boldsymbol{theta}$ is trained by minimizing the loss function $\mathcal{L}$ that combines both a loss related to the physical equations, $\mathcal{L}^{\text{eq}}$, and a loss  related to the data, $\mathcal{L}^{\text{data}}$:

$$\mathcal{L}(\boldsymbol{theta})=\mathcal{L}^{\text{eq}} (\boldsymbol{theta}) + \mathcal{L}^{\text{data}} (\boldsymbol{theta})$$

The loss on the equations consists on the residual on $N_C$ collocation points, $\tau_1, \dots \tau_{N_C}$ (where $\tua_i \in [0,T]$):

$$\mathcal{L}^{\text{eq}}(\boldsymbol{theta})=\frac{1}{N_C} \sum_{i=1}^{N_C} F(\dot{\hat{\mathbf{u}}}(\tau_i,\boldsymbol{theta}),\hat{\mathbf{u}}(\tau_i,\boldsymbol{theta}))$$



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
