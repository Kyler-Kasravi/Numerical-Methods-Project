import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 8, 'lines.linewidth': 1.5,
    'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
})
#The ODE system is such that S+E+I+R=1. The ICS match the ABM (agent-based-model) ICS
#The solution is used to reference the ABM results and show convergence as N increases.
def seir_ode(t, y, beta, sigma, gamma): #we define t despite the fact that the system is autonomous because solve_ivp requires it.
    S, E, I, R = y
    dS = -beta * S * I
    dE =  beta * S * I - sigma * E
    dI =  sigma * E    - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]
#This block runs one simulation and returns the time array and counts of
#our variables at each time step.
def run_abm(N, beta, sigma, gamma, dt, T, I0_count, E0_count, seed=None):
    rand = np.random.default_rng(seed) #random number gen with seed for reproducibility
    n_steps = int(T / dt) + 1 #num time steps
    t_arr = np.linspace(0, T, n_steps) #time array
    p_EI = 1.0 - np.exp(-sigma * dt)   #Probability of exposed -> infected per timestep
    p_IR = 1.0 - np.exp(-gamma * dt)   #Probability of infected -> recovered per timestep
    state = np.zeros(N, dtype=int) #initializes all the agents as susceptible

    #Randomly choose some agents to be exposed and infected
    chosen = rand.choice(N, size=E0_count + I0_count, replace=False)
    state[chosen[:E0_count]] = 1  #we set a random subset of agents to be exposed ~ 0.1%
    state[chosen[E0_count:]] = 2  #We set a random subset of agents to be infected ~ 0.1%

    counts = np.zeros((n_steps, 4), dtype=int) #store counts of S,E,I,R in array
    for s in range(4): #initial counts of each state
        counts[0, s] = np.sum(state == s)
    for step in range(1, n_steps):#loop through each time step
        I_index = np.where(state == 2)[0] #index of infected agents
        n_I = len(I_index)#num infected agents
        n_S = np.sum(state == 0)#num susceptible agents
        if n_I > 0 and n_S > 0: #If we have infected and susceptible agents, we can have exposures
            n_contacts = rand.poisson(beta * dt, size=n_I) #agents make poisson(beta*dt) contacts per timestep
            total_contacts = int(np.sum(n_contacts)) #total contacts made by all infected agents
            if total_contacts > 0:
                targets = rand.integers(0, N, size=total_contacts) #randomly chosen agents for contact
                S_targets = targets[state[targets] == 0] #considers only susceptible agents
                S_targets = np.unique(S_targets)#if agent is contacted multiple times it is counted as 1 exposure
                state[S_targets] = 1 #susceptible contacted agents are now exposed
        E_index = np.where(state == 1)[0]#index of exposed agents
        #Exposed agents are infected with probability p_EI
        if len(E_index) > 0:
            transitions = rand.random(len(E_index)) < p_EI 
            state[E_index[transitions]] = 2 
        #Infected agents recover with probability p_IR
        I_index = np.where(state == 2)[0]
        if len(I_index) > 0:
            transitions = rand.random(len(I_index)) < p_IR
            state[I_index[transitions]] = 3
        for s in range(4):#store counts of each state at this time step
            counts[step, s] = np.sum(state == s)

    return t_arr, counts

#Runs multiple simulations and returns time and 3D array accounts for each run and time step.
def run_sim(N, beta, sigma, gamma, dt, T, I0_count, E0_count, n_runs=30, base_seed=42):
    all_counts = []
    for run in range(n_runs):
        t_arr, counts = run_abm(
            N, beta, sigma, gamma, dt, T,
            I0_count, E0_count, seed=base_seed + run
        )
        all_counts.append(counts)
    return t_arr, np.array(all_counts)
def main():
    print("=" * 60)
    print("  PHYS 3450 — Agent-Based SEIR Model")
    print("=" * 60)
    #Parameters
    beta  = 4.5 / 14.0     #transmission rate
    sigma = 10.0           #latency rate (fast E -> I)
    gamma = 1.0 / 14.0     #recovery rate
    T     = 80.0           #total simulation time (days)
    dt    = 0.05           #ABM time step
    E0_frac = 0.001        #Faction of initially exposed population (0.1%)
    I0_frac = 0.001        #Fraction of initially infected population (0.1%)
    #ODE ICs
    y0_ode = [1.0 - E0_frac - I0_frac, E0_frac, I0_frac, 0.0]
    print("\nComputing ODE reference (Radau)...", flush=True)
    t_ode = np.linspace(0, T, 2000)
    sol = solve_ivp(seir_ode, (0, T), y0_ode, args=(beta, sigma, gamma),t_eval=t_ode, method='Radau', rtol=1e-12, atol=1e-14)
    print("  Done.")
    #Agent-based model parameters
    N_abm  = 50_000
    n_runs = 40
    I0_count = max(1, int(round(I0_frac * N_abm)))
    E0_count = max(1, int(round(E0_frac * N_abm)))
    #This is the block where we compute mean and std of our model runs and compare to ODE sols. 
    print(f"\nRunning {n_runs} ABM realizations (N={N_abm:,})...", flush=True)
    t0 = time.perf_counter()
    t_abm, all_counts = run_sim(
        N_abm, beta, sigma, gamma, dt, T,
        I0_count, E0_count, n_runs=n_runs)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/n_runs:.2f}s per run).")

    #Find mean and std of I(t) across runs
    all_fracs = all_counts / N_abm
    I_fracs = all_fracs[:, :, 2]       #shape (n_runs, n_steps)
    I_mean  = np.mean(I_fracs, axis=0)
    I_std   = np.std(I_fracs, axis=0)

    N_values    = [1_000, 5_000, 10_000, 50_000, 100_000]
    n_runs_conv = 30
    convergence_data = {}
    #find mean and std of I(t) across different N values to show ODE convergence as N -> infinity.
    print(f"\nConvergence study across N values...", flush=True)
    for N_val in N_values:
        I0_c = max(1, int(round(I0_frac * N_val)))
        E0_c = max(1, int(round(E0_frac * N_val)))

        t0c = time.perf_counter()
        t_c, counts_c = run_sim(
            N_val, beta, sigma, gamma, dt, T,
            I0_c, E0_c, n_runs=n_runs_conv)
        el = time.perf_counter() - t0c

        fracs_c  = counts_c / N_val
        I_mean_c = np.mean(fracs_c[:, :, 2], axis=0)
        I_std_c  = np.std(fracs_c[:, :, 2], axis=0)
        convergence_data[N_val] = (t_c, I_mean_c, I_std_c)
        print(f"  N={N_val:>7,}: {el:.1f}s", flush=True)

    #generate figure comparing ABM runs to ODE solution, and showing convergence as N increases.
    print("\nGenerating Figure 6...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: individual runs + mean + ODE
    ax = axes[0]
    for run in range(min(n_runs, 40)):
        ax.plot(t_abm, I_fracs[run], color='0.75', lw=0.3, alpha=0.5)

    ax.fill_between(t_abm, I_mean - 2*I_std, I_mean + 2*I_std,
                    color='#1f77b4', alpha=0.15,
                    label=r'ABM mean $\pm\,2\sigma$')
    ax.plot(t_abm, I_mean, color='#1f77b4', lw=2,
            label=f'ABM mean (N={N_abm:,})')
    ax.plot(sol.t, sol.y[2], 'r--', lw=2, label='ODE (Radau)')
    ax.set_xlabel("Days")
    ax.set_ylabel("Fraction Infected  I(t)")
    ax.set_title(f"Agent-Based SEIR: {n_runs} Runs (N = {N_abm:,})")
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.15)

    #Show extinct runs
    final_R = all_fracs[:, -1, 3]
    n_extinct = np.sum(final_R < 0.01)
    if n_extinct > 0:
        ax.text(0.97, 0.55, f'{n_extinct}/{n_runs} runs\nextinct early',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='0.4', style='italic')

    #Right panel: convergence across N values
    ax = axes[1]
    ax.plot(sol.t, sol.y[2], 'k-', lw=2, label='ODE (deterministic)')
    colors_N = ['#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4', '#d62728']
    for i, N_val in enumerate(N_values):
        t_c, I_mean_c, I_std_c = convergence_data[N_val]
        ax.fill_between(t_c, I_mean_c - 2*I_std_c, I_mean_c + 2*I_std_c,
                        color=colors_N[i], alpha=0.1)
        ax.plot(t_c, I_mean_c, color=colors_N[i], lw=1.2,
                label=f'N = {N_val:,}')

    ax.set_xlabel("Days")
    ax.set_ylabel("Fraction Infected  I(t)")
    ax.set_title(r"Convergence to ODE as $N \to \infty$")
    ax.legend(loc='upper left', fontsize=7)
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    path = 'fig06_abm_vs_ode.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    #Print statistics comparing ABM results and showing convergence.
    print("\n" + "-" * 60)
    print("  Summary Statistics")
    print("-" * 60)
    print(f"  ODE peak I:          {np.max(sol.y[2]):.4f}")
    print(f"  ABM mean peak I:     {np.max(I_mean):.4f}  "
          f"(N={N_abm:,}, {n_runs} runs)")
    print(f"  ABM std at peak:     {I_std[np.argmax(I_mean)]:.4f}")
    print(f"  Stochastic extinct:  {n_extinct}/{n_runs} runs "
          f"({100*n_extinct/n_runs:.0f}%)")

    print(f"\n  Convergence (peak I std across N):")
    for N_val in N_values:
        _, m, s = convergence_data[N_val]
        peak_idx = np.argmax(m)
        print(f"    N = {N_val:>7,}:  peak std = {s[peak_idx]:.5f}  "
              f"(~ 1/sqrt(N) = {1/np.sqrt(N_val):.5f})")

    print("\n" + "=" * 60)
    print("  Figure 6 generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()