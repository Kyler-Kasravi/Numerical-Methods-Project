#PHYS 3450 Project Figure Generation
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import time
import warnings
import os

warnings.filterwarnings('ignore')

CSV_PATH = r"covid_data.csv"
OUTPUT_DIR = "."
N = 60_360_000  #Italy population

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

#each solver takes a right-hand side function, time span, initial conditions,
#step size, and optional args for parameters.
#explicit methods evaluate f directly, implicit methods use Newton's method
#with Jacobian to solve for the next step.

#forward Euler: explicit, first order. just steps forward by dt*f(t,y).
def forward_euler(rhs, t0, tmax, y0, dt, args=()):
    n = int((tmax - t0) / dt) + 1
    t = np.linspace(t0, tmax, n)
    y0 = np.array(y0, dtype=float)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + dt * rhs(t[i], y[i], *args)
    return t, y

#midpoint RK2: explicit, second order. evaluates slope at midpoint of the interval.
def midpoint_rk2(rhs, t0, tmax, y0, dt, args=()):
    n = int((tmax - t0) / dt) + 1
    t = np.linspace(t0, tmax, n)
    y0 = np.array(y0, dtype=float)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        k1 = rhs(t[i], y[i], *args)
        k2 = rhs(t[i] + dt / 2, y[i] + dt / 2 * k1, *args)
        y[i + 1] = y[i] + dt * k2
    return t, y

#RK4: explicit, fourth order. four function evaluations per step.
def rk4(rhs, t0, tmax, y0, dt, args=()):
    n = int((tmax - t0) / dt) + 1
    t = np.linspace(t0, tmax, n)
    y0 = np.array(y0, dtype=float)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        k1 = rhs(t[i], y[i], *args)
        k2 = rhs(t[i] + dt / 2, y[i] + dt / 2 * k1, *args)
        k3 = rhs(t[i] + dt / 2, y[i] + dt / 2 * k2, *args)
        k4 = rhs(t[i] + dt, y[i] + dt * k3, *args)
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y

#backward Euler: implicit, first order. solves y_{n+1} = y_n + dt*f(t_{n+1}, y_{n+1})
#using Newton's method
def backward_euler(rhs, t0, tmax, y0, dt, args=()):
    n = int((tmax - t0) / dt) + 1
    t = np.linspace(t0, tmax, n)
    y0 = np.array(y0, dtype=float)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0
    I_m = np.eye(m) #identity matrix for Newton solve
    eps = 1e-8 
    for i in range(n - 1):
        yn = y[i]
        #initial guess via forward Euler
        yg = yn + dt * rhs(t[i + 1], yn, *args)
        for _ in range(50): #Newton iterations, max 50
            fg = rhs(t[i + 1], yg, *args)
            F_vec = yg - yn - dt * fg #residual of implicit equation
            if np.linalg.norm(F_vec) < 1e-10: #converged
                break
            #build numerical Jacobian by finite difference
            J = np.zeros((m, m))
            for j in range(m):
                yp = yg.copy()
                yp[j] += eps
                J[:, j] = (rhs(t[i + 1], yp, *args) - fg) / eps
            yg += np.linalg.solve(I_m - dt * J, -F_vec) #Newton update
        y[i + 1] = yg
    return t, y

#Crank-Nicolson: implicit, second order. averages forward and backward Euler slopes.
#uses Newton's method with numerical Jacobian.
def crank_nicolson(rhs, t0, tmax, y0, dt, args=()):
    n = int((tmax - t0) / dt) + 1
    t = np.linspace(t0, tmax, n)
    y0 = np.array(y0, dtype=float)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0
    I_m = np.eye(m)
    eps = 1e-8
    for i in range(n - 1):
        yn = y[i]
        fn = rhs(t[i], yn, *args) #slope at current time
        #initial guess with forward Euler
        yg = yn + dt * fn
        for _ in range(50):
            fg = rhs(t[i + 1], yg, *args)
            F_vec = yg - yn - (dt / 2) * (fn + fg) #average of both slopes
            if np.linalg.norm(F_vec) < 1e-10:
                break
            J = np.zeros((m, m))
            for j in range(m):
                yp = yg.copy()
                yp[j] += eps
                J[:, j] = (rhs(t[i + 1], yp, *args) - fg) / eps
            yg += np.linalg.solve(I_m - (dt / 2) * J, -F_vec)
        y[i + 1] = yg
    return t, y

#SEIR right-hand side with constant beta. 
def seir(t, y, beta, sigma, gamma):
    S, E, I, R = y
    dS = -beta * S * I
    dE = beta * S * I - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])

#SEIR with time-varying beta to model lockdown. beta transitions from b1 to b2
def seir_tv(t, y, b1, b2, sigma, gamma, t_lock, k):
    S, E, I, R = y
    beta = b2 + (b1 - b2) / (1 + np.exp(k * (t - t_lock)))
    dS = -beta * S * I
    dE = beta * S * I - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])

#Jacobian of the SEIR system for stiffness analysis.
def seir_jacobian(t, y, beta, sigma, gamma):
    S, E, I, R = y
    return np.array([
        [-beta * I,  0,       -beta * S,  0],
        [ beta * I, -sigma,    beta * S,  0],
        [ 0,         sigma,   -gamma,     0],
        [ 0,         0,        gamma,     0],
    ])



#stores each solver with its plotting style and convergence order
SOLVERS = {
    'Forward Euler':  {'fn': forward_euler,  'color': 'b',  'ls': '-',  'order': 1},
    'Midpoint (RK2)': {'fn': midpoint_rk2,   'color': 'r',  'ls': '-',  'order': 2},
    'RK4':            {'fn': rk4,            'color': 'k',  'ls': '-',  'order': 4},
    'Backward Euler': {'fn': backward_euler, 'color': 'b',  'ls': '--', 'order': 1},
    'Crank-Nicolson': {'fn': crank_nicolson, 'color': 'r',  'ls': '--', 'order': 2},
}

#demo parameters for the stiff SEIR system
beta_demo = 4.5 / 14       #R0 = 4.5 with gamma = 1/14
sigma_demo = 10.0           #fast latency -> stiffness
gamma_demo = 1.0 / 14.0    #14-day recovery
seir_args = (beta_demo, sigma_demo, gamma_demo)
y0_demo = [1.0 - 20 / N, 10 / N, 10 / N, 0.0] #ICs: almost all susceptible, small E and I
t0_demo, tf_demo = 0, 80

def main():
    print("=" * 60)
    print("  PHYS 3450 Project — Figure Generation")
    print("=" * 60)


    #reads Italy COVID data from OWID csv, filtered to Feb 21 - May 31 2020
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    italy = df[df['country'] == 'Italy'].reset_index(drop=True)
    italy = italy[(italy['date'] >= '2020-02-21')
                  & (italy['date'] <= '2020-05-31')].reset_index(drop=True)
    italy['day'] = (italy['date'] - italy['date'].min()).dt.days.astype(float)
    #active cases approximated by 14-day rolling average of new cases
    #accounts for under-reporting of recoveries in the raw data
    italy['active_14day'] = (pd.Series(italy['new_cases'])
                             .rolling(14, center=True, min_periods=1)
                             .mean().values)
    italy['total_cases'] = italy['total_cases'].ffill().bfill()
    italy['new_cases'] = italy['new_cases'].ffill().bfill()
    italy['total_deaths'] = italy['total_deaths'].ffill().bfill()
    italy['active_14day'] = italy['active_14day'].ffill().bfill()

    days = italy['day'].values.astype(float)
    IR_data = italy['total_cases'].values
    I_data = italy['active_14day'].values
    nc = italy['new_cases'].values
    deaths = italy['total_deaths'].values
    nc7 = pd.Series(nc).rolling(7, center=True, min_periods=1).mean().values #7-day smoothed new cases
    print(f"  {len(italy)} data points loaded (Feb 21 - May 31, 2020).")

    #Radau solve used as reference for error comparisons
    t_fine = np.linspace(0, 80, 5000)
    ref = solve_ivp(seir, (t0_demo, tf_demo), y0_demo, args=seir_args,
                    t_eval=t_fine, method='Radau', rtol=1e-12, atol=1e-14)
    ref_interp = interp1d(ref.t, ref.y, axis=1, kind='cubic') #interpolator for comparing at arbitrary times
    print("  Done.")

    #plots total cases, active cases, deaths, and daily new cases
    print("\nFigure 1: Italy raw data:", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(days, IR_data / 1e3, 'k.', ms=3, label='Cumulative Cases')
    ax.plot(days, I_data / 1e3, 'b^', ms=3, label='Active Cases')
    ax.plot(days, deaths / 1e3, 'rv', ms=3, label='Cumulative Deaths')
    ax.axvline(17, color='red', ls='--', alpha=0.6, label='Lockdown (Mar 9)')
    ax.set_xlabel("Days since Feb 21, 2020")
    ax.set_ylabel("Count (thousands)")
    ax.set_title("Italy COVID-19 Raw Data")
    ax.legend(loc='upper left', fontsize=8)

    ax = axes[1]
    ax.plot(days, nc, color='0.75', lw=0.8, label='Daily New Cases')
    ax.plot(days, nc7, 'b-', lw=2, label='7-day Average')
    ax.axvline(17, color='red', ls='--', alpha=0.6, label='Lockdown')
    ax.set_xlabel("Days since Feb 21, 2020")
    ax.set_ylabel("New Cases per Day")
    ax.set_title("Daily New Cases (Italy)")
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig01_italy_data.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    #runs each solver at multiple step sizes on the stiff SEIR system
    #to show how explicit methods fail at large dt while implicit methods stay stable
    print("\nFigure 2: ", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    #explicit solvers on top row, each tested at 3 step sizes
    explicit_configs = [
        ('Forward Euler',  [0.5, 0.2, 0.15]),
        ('Midpoint (RK2)', [0.5, 0.2, 0.1]),
        ('RK4',            [0.2, 0.1, 0.05]),
    ]
    for col, (name, dt_vals) in enumerate(explicit_configs):
        ax = axes[0, col]
        ax.plot(ref.t, ref.y[2], 'k-', lw=2, label='Reference (Radau)')
        info = SOLVERS[name]
        for dt_v, ls_v in zip(dt_vals, [':', '-.', '--']):
            ts, ys = info['fn'](seir, t0_demo, tf_demo, y0_demo, dt_v, args=seir_args)
            ax.plot(ts, np.clip(ys[:, 2], -0.3, 0.6),
                    color=info['color'], ls=ls_v, alpha=0.8,
                    label=f'dt={dt_v}')
        ax.set_ylim(-0.15, 0.55)
        ax.set_xlabel("Days")
        ax.set_ylabel("I(t)")
        order_str = info['order']
        ax.set_title(f"{name}  (Explicit, O(dt^{order_str}))")
        ax.legend(loc='upper left', fontsize=7)

    #implicit solvers on bottom row
    implicit_configs = [
        ('Backward Euler',  [0.5, 0.2, 0.1]),
        ('Crank-Nicolson',  [0.5, 0.2, 0.1]),
    ]
    for col, (name, dt_vals) in enumerate(implicit_configs):
        ax = axes[1, col]
        ax.plot(ref.t, ref.y[2], 'k-', lw=2, label='Reference (Radau)')
        info = SOLVERS[name]
        for dt_v, ls_v in zip(dt_vals, [':', '-.', '--']):
            ts, ys = info['fn'](seir, t0_demo, tf_demo, y0_demo, dt_v, args=seir_args)
            ax.plot(ts, ys[:, 2], color=info['color'], ls=ls_v, alpha=0.8,
                    label=f'dt={dt_v}')
        ax.set_xlabel("Days")
        ax.set_ylabel("I(t)")
        order_str = info['order']
        ax.set_title(f"{name}  (Implicit, O(dt^{order_str}), A-stable)")
        ax.legend(loc='upper left', fontsize=7)

    #bottom-right panel: all 5 solvers at the same step size for direct comparison
    ax = axes[1, 2]
    ax.plot(ref.t, ref.y[2], 'k-', lw=2.5, label='Reference')
    dt_cmp = 0.1
    for name, info in SOLVERS.items():
        try:
            ts, ys = info['fn'](seir, t0_demo, tf_demo, y0_demo, dt_cmp, args=seir_args)
            ax.plot(ts, np.clip(ys[:, 2], -0.3, 0.6),
                    color=info['color'], ls=info['ls'], alpha=0.85, label=name)
        except Exception as e:
            print(f"    {name} failed at dt={dt_cmp}: {e}")
    ax.set_ylim(-0.15, 0.55)
    ax.set_xlabel("Days")
    ax.set_ylabel("I(t)")
    ax.set_title(f"All Solvers at dt = {dt_cmp} days")
    ax.legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig02_stability_demo.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    #runs each solver at many step sizes and measures relative error of I(t)
    #against the Radau reference. also times each run for the efficiency plot.
    print("\nFigure 3: ", flush=True)
    dt_vals_conv = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
    rows = []

    for dt in dt_vals_conv:
        for name, info in SOLVERS.items():
            try:
                t0_timer = time.perf_counter()
                ts, ys = info['fn'](seir, t0_demo, tf_demo, y0_demo, dt, args=seir_args)
                elapsed = time.perf_counter() - t0_timer
                #compare points within the reference solution's domain
                mask = (ts >= ref.t[0]) & (ts <= ref.t[-1])
                yr = ref_interp(ts[mask]).T
                #relative RMS error of the I component
                err = np.sqrt(np.mean((ys[mask, 2] - yr[:, 2])**2))
                err /= (np.max(np.abs(yr[:, 2])) + 1e-15)
                #skip if error is too large
                if 0 < err < 10 and not np.isnan(err):
                    rows.append({
                        'dt': dt, 'method': name, 'error': err,
                        'time': elapsed, 'order': info['order'],
                    })
            except:
                pass
        print(f"    dt={dt:.3f} done", flush=True)

    comp = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    #left panel: error vs step size (log-log), slope is convergence order
    ax = axes[0]
    for name, info in SOLVERS.items():
        sub = comp[comp['method'] == name]
        if len(sub) > 0:
            ax.loglog(sub['dt'], sub['error'], 'o-',
                      color=info['color'], ls=info['ls'],
                      label=name, markersize=5)
    #reference slopes for comparison
    dr = np.array([0.008, 2.5])
    ax.loglog(dr, 0.3 * dr**1, color='gray', ls=':', alpha=0.4,
              label=r'$O(\Delta t)$')
    ax.loglog(dr, 0.3 * dr**2, color='gray', ls='-.', alpha=0.4,
              label=r'$O(\Delta t^2)$')
    ax.loglog(dr, 0.01 * dr**4, color='gray', ls='--', alpha=0.4,
              label=r'$O(\Delta t^4)$')
    ax.set_xlabel(r"Step Size $\Delta t$ (days)")
    ax.set_ylabel("Relative Error (I component)")
    ax.set_title(r"Convergence: Error vs $\Delta t$")
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    #right panel: error vs computation time (efficiency)
    ax = axes[1]
    for name, info in SOLVERS.items():
        sub = comp[comp['method'] == name]
        if len(sub) > 0:
            ax.loglog(sub['time'] * 1000, sub['error'], 'o-',
                      color=info['color'], ls=info['ls'],
                      label=name, markersize=5)
    ax.set_xlabel("Computation Time (ms)")
    ax.set_ylabel("Relative Error")
    ax.set_title("Efficiency: Error vs Computation Time")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig03_convergence.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    #computes the Jacobian eigenvalues along the trajectory
    #to show how the stiffness ratio and Euler stability limit change over time
    print("\nFigure 4: ", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sr = np.zeros(len(ref.t)) #stiffness ratio at each time
    dt_euler_lim = np.full(len(ref.t), np.inf) #max stable dt for forward Euler
    for i in range(len(ref.t)):
        J = seir_jacobian(ref.t[i], ref.y[:, i], *seir_args)
        eigs = np.linalg.eigvals(J)
        ae = np.abs(eigs)
        nz = ae > 1e-15 #ignore zero eigenvalues
        if np.sum(nz) >= 2:
            sr[i] = np.max(ae[nz]) / np.min(ae[nz])
        #forward Euler stability: dt < 2/|lambda_max|
        re_neg = np.real(eigs)[np.real(eigs) < -1e-15]
        if len(re_neg) > 0:
            dt_euler_lim[i] = 2.0 / np.max(np.abs(re_neg))

    ax.semilogy(ref.t, sr, 'k-', lw=1.5)
    ax.set_xlabel("Days")
    ax.set_ylabel(r"Stiffness Ratio $|\lambda_{\max}|/|\lambda_{\min}|$")
    ax.set_title(r"SEIR Stiffness Ratio ($\sigma=10$, $\gamma=1/14$)")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.semilogy(ref.t, dt_euler_lim, 'k-', lw=1.5)
    ax.axhline(1.0, color='red', ls='--', alpha=0.5, label=r'$\Delta t = 1$ day')
    ax.axhline(0.2, color='orange', ls=':', alpha=0.5, label=r'$\Delta t = 0.2$ days')
    ax.axhline(0.1, color='blue', ls=':', alpha=0.5, label=r'$\Delta t = 0.1$ days')
    ax.set_xlabel("Days")
    ax.set_ylabel(r"Max Stable $\Delta t$ (days)")
    ax.set_title("Forward Euler Stability Limit")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig04_stiffness.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    #fits the time-varying SEIR model to the Italy data using differential evolution
    #sigma is 10 (fast latency), gamma is 1/14 (14-day recovery)
    #optimizes beta1, beta2, t_lock, and k to minimize error of I+R fraction
    sig_fit = 10.0
    gam_fit = 1.0 / 14.0
    E0 = 5 / N
    I0_fit = 15 / N
    y0_fit = [1.0 - E0 - I0_fit, E0, I0_fit, 0.0]
    t_span_fit = (days[0], days[-1])
    IR_frac = italy['total_cases'].values / N #observed I+R as fraction of population

    #objective function: sum of squared differences between model I+R and data I+R
    def objective(params):
        b1, b2, tl, ks = params
        try:
            sol = solve_ivp(seir_tv, t_span_fit, y0_fit,
                            args=(b1, b2, sig_fit, gam_fit, tl, ks),
                            t_eval=days, method='Radau', rtol=1e-8, atol=1e-10)
            if not sol.success:
                return 1e10
            return np.sum((sol.y[2] + sol.y[3] - IR_frac)**2)
        except:
            return 1e10

    res = differential_evolution(
        objective,
        bounds=[(0.1, 2.0), (0.01, 0.3), (10, 35), (0.05, 3.0)],
        seed=42, maxiter=300, tol=1e-8,
    )
    b1f, b2f, tlf, kf = res.x
    R0_pre = b1f / gam_fit
    R0_post = b2f / gam_fit
    print(f"  Fit results: R0_pre={R0_pre:.2f}, R0_post={R0_post:.2f}, "
          f"t_lock=Day {tlf:.1f}, k={kf:.2f}")

    #solve with fitted parameters for plotting
    sol_fit = solve_ivp(seir_tv, t_span_fit, y0_fit,
                        args=(b1f, b2f, sig_fit, gam_fit, tlf, kf),
                        t_eval=days, method='Radau', rtol=1e-10, atol=1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    #left image: model fit vs data
    ax = axes[0]
    ax.plot(days, IR_frac * N / 1e3, 'ko', ms=2.5, label='Data (I+R)')
    ax.plot(days, I_data / 1e3, 'b^', ms=2.5, label='Active (14-day)')
    ax.plot(sol_fit.t, (sol_fit.y[2] + sol_fit.y[3]) * N / 1e3,
            'r-', lw=2, label='SEIR fit (I+R)')
    ax.plot(sol_fit.t, sol_fit.y[2] * N / 1e3,
            'r--', lw=1.5, label='SEIR fit (I)')
    ax.axvline(17, color='gray', ls=':', alpha=0.6, label='Lockdown (Mar 9)')
    ax.set_xlabel("Days since Feb 21, 2020")
    ax.set_ylabel("People (thousands)")
    ax.set_title("SEIR Model Fit to Italy COVID-19")
    ax.legend(loc='upper left', fontsize=8)

    #right image: time-varying R0 showing the lockdown pushing R0 below 1
    ax = axes[1]
    tp = np.linspace(0, 100, 500)
    bt = b2f + (b1f - b2f) / (1 + np.exp(kf * (tp - tlf)))
    ax.plot(tp, bt / gam_fit, 'r-', lw=2, label=r'$R_0(t) = \beta(t)/\gamma$')
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label=r'$R_0 = 1$ threshold')
    ax.axvline(17, color='gray', ls=':', alpha=0.6, label='Lockdown (Mar 9)')
    ax.fill_between(tp, 0, bt / gam_fit, where=bt / gam_fit > 1,
                    alpha=0.08, color='red')
    ax.fill_between(tp, 0, bt / gam_fit, where=bt / gam_fit <= 1,
                    alpha=0.08, color='green')
    ax.set_xlabel("Days since Feb 21, 2020")
    ax.set_ylabel(r"Effective $R_0(t)$")
    ax.set_title(r"Time-Varying Reproduction Number $R_0(t)$")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, R0_pre * 1.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig05_seir_fit.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print("\n" + "=" * 60)
    print("  All 5 figures generated:")
    print("  fig01_italy_data.png      — Raw COVID-19 data")
    print("  fig02_stability_demo.png  — Stability: 5 solvers on stiff SEIR")
    print("  fig03_convergence.png     — Convergence + efficiency")
    print("  fig04_stiffness.png       — Stiffness ratio + Euler limit")
    print("  fig05_seir_fit.png        — SEIR fit to Italy data")
    print("=" * 60)


if __name__ == '__main__':
    main()
