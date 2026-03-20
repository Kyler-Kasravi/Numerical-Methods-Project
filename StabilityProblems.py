import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

def implicit_euler(rhs, Y0, t, *params): #implicit euler from notebook
    nt = t.size
    dt = t[1] - t[0]
    n = Y0.size
    Y = np.zeros((nt, n))
    Y[0, :] = Y0
    f_i = 0
    def func(Y, t, Y0):
        return Y - Y0 - rhs(t, Y, *params) * dt
    for i in range(1, nt):
        sol = fsolve(func, Y[i-1, :], args=(t[i-1], Y[i-1, :]), full_output=True)
        Y[i, :] = sol[0]
        f_i += sol[1]["nfev"]
    print("Number of function evaluations for implicit Euler:", f_i)
    return Y, f_i
def rk4(rhs, Y0, t, *params): #RK4 method
    nt = t.size
    dt = t[1] - t[0]
    n = Y0.size
    Y = np.zeros((nt, n))
    Y[0, :] = Y0
    for i in range(1, nt):
        k1 = rhs(t[i-1], Y[i-1, :], *params)
        k2 = rhs(t[i-1] + dt/2, Y[i-1, :] + dt/2 * k1, *params)
        k3 = rhs(t[i-1] + dt/2, Y[i-1, :] + dt/2 * k2, *params)
        k4 = rhs(t[i-1] + dt, Y[i-1, :] + dt * k3, *params)
        Y[i, :] = Y[i-1, :] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return Y, 4 * (nt - 1)
def rhs_stiff_2(t, y, alpha, beta): #rhs stiff solver
    return alpha * np.exp(t) - beta * y

#parameters for problem 1
tmax = 0.16
nt = 28
t = np.linspace(0, tmax, nt)
y0 = np.array([0.0])
alpha = 501
beta = 500
print("=" * 60)
print("Problem 1: Implicit Euler vs RK4 for stiff-2")
print("=" * 60)
y_implicit, nfev_ie = implicit_euler(rhs_stiff_2, y0, t, *(alpha, beta))
y_rk4, nfev_rk4 = rk4(rhs_stiff_2, y0, t, *(alpha, beta))
y_exact = np.exp(t) - np.exp(-500 * t)  #exact: y(t) = e^t - e^{-500t}

fig = plt.figure(figsize=(14, 6))
#plotting for problem 1
ax = fig.add_subplot(121)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Solution $y(t)$", size=18)
ax.set_title(r"Stiff ODE: $y' = 501e^t - 500y$,  $y(0)=0$" + "\nExact vs numerical solutions", size=13)
ax.plot(t, y_exact, "k-", linewidth=2, label=r"Exact: $e^t - e^{-500t}$")
ax.plot(t, y_implicit.flatten(), "o-", label=f"Implicit Euler ({nfev_ie} f-evals)")
ax.plot(t, y_rk4.flatten(), "s-", label=f"RK4 ({nfev_rk4} f-evals)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = fig.add_subplot(122)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Absolute error $|y_{num} - y_{exact}|$", size=18)
ax.set_title("Error: RK4 diverges due to stiffness" + "\n" + r"($\beta=500$ exceeds RK4 stability region)", size=13)
err_implicit = np.abs(y_implicit.flatten() - y_exact)
err_rk4 = np.abs(y_rk4.flatten() - y_exact)
ax.semilogy(t[1:], err_implicit[1:], "o-", label=f"Implicit Euler ({nfev_ie} f-evals)")
ax.semilogy(t[1:], err_rk4[1:], "s-", label=f"RK4 ({nfev_rk4} f-evals)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("problem1.png", dpi=300)
plt.show()


#Problem 2: Flame growth model with dy/dt = y^2 - y^3, y0=0.01, tmax=2/y0, rtol=1e-4

def rhs_flame(t, y):
    return y**2 - y**3

print("\n" + "=" * 60)
print("Problem 2: Flame growth model")
print("=" * 60)
#tmax is chosen to show transition from slow to rapid growth, y0 is small for stiffness
y0_flame = 0.01
tmax_flame = 2.0 / y0_flame

#Implicit Euler
nt_flame = 500
t_flame = np.linspace(0, tmax_flame, nt_flame)
y0_arr = np.array([y0_flame])

y_flame_implicit, nfev_ie_flame = implicit_euler(rhs_flame, y0_arr, t_flame)

#using solve_ivp with Radau, which is an implicit RK method
sol_radau = solve_ivp(rhs_flame, (0, tmax_flame), y0_arr,
                       method="Radau", rtol=1e-4)
print("Number of function evaluations for Radau:", sol_radau["nfev"])

#solve_ivp with BDF, which is another implicit method for stiff problems (multistep)
sol_bdf = solve_ivp(rhs_flame, (0, tmax_flame), y0_arr,
                     method="BDF", rtol=1e-4)
print("Number of function evaluations for BDF:", sol_bdf["nfev"])

#using solve_ivp with RK45, to show it struggles with stiffness as an explicit method.
sol_rk45 = solve_ivp(rhs_flame, (0, tmax_flame), y0_arr,
                      method="RK45", rtol=1e-4)
print("Number of function evaluations for RK45:", sol_rk45["nfev"])

fig = plt.figure(figsize=(14, 6))
#plotting for problem 2
ax = fig.add_subplot(121)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Flame intensity $y(t)$", size=18)
ax.set_title(r"Flame Growth: $y' = y^2 - y^3$" + "\n" + r"$y_0 = 0.01$, stiff transition to $y=1$", size=13)
ax.plot(t_flame, y_flame_implicit, label=f"Implicit Euler ({nfev_ie_flame} f-evals)")
ax.plot(sol_radau.t, sol_radau.y[0], "o-", label=f"Radau ({sol_radau.nfev} f-evals)", markersize=4)
ax.plot(sol_bdf.t, sol_bdf.y[0], "s-", label=f"BDF ({sol_bdf.nfev} f-evals)", markersize=4)
ax.plot(sol_rk45.t, sol_rk45.y[0], "^-", label=f"RK45 ({sol_rk45.nfev} f-evals)", markersize=4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = fig.add_subplot(122)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Flame intensity $y(t)$", size=18)
ax.set_title("Zoomed: stiff transition region", size=14)
ax.set_xlim(150, 210)
ax.plot(t_flame, y_flame_implicit, label=f"Implicit Euler ({nfev_ie_flame} f-evals)")
ax.plot(sol_radau.t, sol_radau.y[0], "o-", label=f"Radau ({sol_radau.nfev} f-evals)", markersize=4)
ax.plot(sol_bdf.t, sol_bdf.y[0], "s-", label=f"BDF ({sol_bdf.nfev} f-evals)", markersize=4)
ax.plot(sol_rk45.t, sol_rk45.y[0], "^-", label=f"RK45 ({sol_rk45.nfev} f-evals)", markersize=4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("problem2.png", dpi=300)
plt.show()



#problem 3: 2x2 stiff system, dy/dt = A*y, A=[[98,198],[-99,-199]], y0=(1,1), t is in [0,10]

print("\n" + "=" * 60)
print("Problem 3: 2x2 stiff system")
print("=" * 60)
#array A has eigenvalues with large negative real parts, making it stiff
A = np.array([[98, 198], [-99, -199]])
eig = np.linalg.eig(A)
print("Eigenvalues of A:", eig[0])
dt_max = 2.0 / np.max(np.abs(eig[0]))
print(f"Forward Euler stability limit: dt < {dt_max:.6f}")
#here we define the right hand side function for our ODE system
def rhs_p3(t, Y):
    A = np.array([[98, 198], [-99, -199]])
    return np.dot(A, Y)
#here we have y0 as a 2D vector and tspan for integration
Y0 = np.array([1.0, 1.0])
tspan = (0, 10)

#solving with RK45
sol_rk45 = solve_ivp(rhs_p3, tspan, Y0, method="RK45", rtol=1e-5)
print("Number of time steps needed for RK45:", sol_rk45.t.size)
print("Number of function evaluations for RK45:", sol_rk45["nfev"])

#Solving with Radau
sol_radau = solve_ivp(rhs_p3, tspan, Y0, method="Radau", rtol=1e-5)
print("Number of time steps needed for Radau:", sol_radau.t.size)
print("Number of function evaluations for Radau:", sol_radau["nfev"])

#Solving with BDF
sol_bdf = solve_ivp(rhs_p3, tspan, Y0, method="BDF", rtol=1e-5)
print("Number of time steps needed for BDF:", sol_bdf.t.size)
print("Number of function evaluations for BDF:", sol_bdf["nfev"])

fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(121)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Solution components $y_i(t)$", size=18)
ax.set_title(r"Stiff $2\times2$ System: $\mathbf{y}'=A\mathbf{y}$" + "\n" + r"$\lambda = -1, -100$ (stiffness ratio 100)", size=13)
ax.plot(sol_rk45.t, sol_rk45.y[0], "-", color="C0", label=r"$y_1$ — RK45")
ax.plot(sol_rk45.t, sol_rk45.y[1], "--", color="C0", label=r"$y_2$ — RK45")
ax.plot(sol_radau.t, sol_radau.y[0], "o", color="C1", label=r"$y_1$ — Radau", markersize=4)
ax.plot(sol_radau.t, sol_radau.y[1], "s", color="C1", label=r"$y_2$ — Radau", markersize=4)
ax.plot(sol_bdf.t, sol_bdf.y[0], "^", color="C2", label=r"$y_1$ — BDF", markersize=4)
ax.plot(sol_bdf.t, sol_bdf.y[1], "v", color="C2", label=r"$y_2$ — BDF", markersize=4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)

ax = fig.add_subplot(122)
methods_p3 = ["RK45\n(explicit)", "Radau\n(implicit)", "BDF\n(implicit)"]
fevals_p3 = [sol_rk45.nfev, sol_radau.nfev, sol_bdf.nfev]
colors_p3 = ["#e74c3c", "#2ecc71", "#3498db"]
bars = ax.bar(methods_p3, fevals_p3, color=colors_p3, edgecolor="black", width=0.6)
ax.set_ylabel("Function evaluations", size=18)
ax.set_title("Solver efficiency comparison\n(lower is better)", size=13)
for bar, val in zip(bars, fevals_p3):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fevals_p3)*0.02,
            str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("problem3.png", dpi=300)
plt.show()

#Problem 4: Oscillating system, dy/dt = [[0, 0.1],[-0.1, 0]] * y, y0=(1,1), t in [0,50]
print("\n" + "=" * 60)
print("Problem 4: Oscillating system")
print("=" * 60)
#A4 has imaginary eigenvalues, so the system shouldn't be too stiff, but we will have infinite oscillation.
A4 = np.array([[0, 0.1], [-0.1, 0]])
eig4 = np.linalg.eig(A4)
print("Eigenvalues for A:", eig4[0])

def rhs_p4(t, Y):
    A = np.array([[0, 0.1], [-0.1, 0]])
    return np.dot(A, Y)

Y0_p4 = np.array([1.0, 1.0])
tspan_p4 = (0, 50)

#solving with RK45
sol_rk45 = solve_ivp(rhs_p4, tspan_p4, Y0_p4, method="RK45", rtol=1e-5)
print("Number of function evaluations for RK45:", sol_rk45["nfev"])

#solving with Radau
sol_radau = solve_ivp(rhs_p4, tspan_p4, Y0_p4, method="Radau", rtol=1e-5)
print("Number of function evaluations for Radau:", sol_radau["nfev"])

#solving with BDF
sol_bdf = solve_ivp(rhs_p4, tspan_p4, Y0_p4, method="BDF", rtol=1e-5)
print("Number of function evaluations for BDF:", sol_bdf["nfev"])

fig = plt.figure(figsize=(14, 10))

ax = fig.add_subplot(221)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Solution components $y_i(t)$", size=18)
ax.set_title(r"Oscillator ($\lambda=\pm 0.1i$): RK45" + f"\n({sol_rk45.nfev} f-evals)", size=13)
ax.plot(sol_rk45.t, sol_rk45.y[0], label=r"$y_1(t)$")
ax.plot(sol_rk45.t, sol_rk45.y[1], label=r"$y_2(t)$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

ax = fig.add_subplot(222)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Solution components $y_i(t)$", size=18)
ax.set_title(r"Oscillator ($\lambda=\pm 0.1i$): Radau" + f"\n({sol_radau.nfev} f-evals)", size=13)
ax.plot(sol_radau.t, sol_radau.y[0], label=r"$y_1(t)$")
ax.plot(sol_radau.t, sol_radau.y[1], label=r"$y_2(t)$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

ax = fig.add_subplot(223)
ax.set_xlabel(r"Time $t$", size=18)
ax.set_ylabel(r"Solution components $y_i(t)$", size=18)
ax.set_title(r"Oscillator ($\lambda=\pm 0.1i$): BDF" + f"\n({sol_bdf.nfev} f-evals)", size=13)
ax.plot(sol_bdf.t, sol_bdf.y[0], label=r"$y_1(t)$")
ax.plot(sol_bdf.t, sol_bdf.y[1], label=r"$y_2(t)$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

ax = fig.add_subplot(224)
ax.set_xlabel(r"$y_1$", size=18)
ax.set_ylabel(r"$y_2$", size=18)
ax.set_title("Phase portrait (RK45)\nExpect circular orbit", size=13)
ax.plot(sol_rk45.y[0], sol_rk45.y[1])
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("problem4.png", dpi=300)
plt.show()



#Problem 5: Robertson's autocatalytic reaction, dx/dt = -0.04x + 1e4*y*z, 
#dy/dt =  0.04x - 1e4*y*z - 3e7*y^2, dz/dt =  3e7*y^2, y0 = (1, 0, 0), tspan = (0, 500)


print("\n" + "=" * 60)
print("Problem 5: Robertson's autocatalytic reaction")
print("=" * 60)

def rhs_robertson(t, Y):
    x, y, z = Y
    dxdt = -0.04 * x + 1e4 * y * z
    dydt = 0.04 * x - 1e4 * y * z - 3e7 * y**2
    dzdt = 3e7 * y**2
    return np.array([dxdt, dydt, dzdt])
Y0_rob = np.array([1.0, 0.0, 0.0])
tspan_rob = (0, 500)

#solving with RK45
sol_rk45 = solve_ivp(rhs_robertson, tspan_rob, Y0_rob,
                      method="RK45", rtol=1e-6, atol=1e-8)
print("Number of function evaluations for RK45:", sol_rk45["nfev"])
print("Number of time steps for RK45:", sol_rk45.t.size)

#Solving with Radau
sol_radau = solve_ivp(rhs_robertson, tspan_rob, Y0_rob,
                       method="Radau", rtol=1e-6, atol=1e-8)
print("Number of function evaluations for Radau:", sol_radau["nfev"])
print("Number of time steps for Radau:", sol_radau.t.size)

#Solving with BDF
sol_bdf = solve_ivp(rhs_robertson, tspan_rob, Y0_rob,
                     method="BDF", rtol=1e-6, atol=1e-8)
print("Number of function evaluations for BDF:", sol_bdf["nfev"])
print("Number of time steps for BDF:", sol_bdf.t.size)
#plotting the results for problem 5.
fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(121)
ax.set_xlabel(r"Time $t$ (s, log scale)", size=18)
ax.set_ylabel("Species concentration", size=18)
ax.set_title("Robertson's Autocatalytic Reaction" + "\n" + r"(— Radau,  -- BDF,  $\cdots$ RK45)", size=13)
# Radau (solid) - with species labels
ax.semilogx(sol_radau.t[1:], sol_radau.y[0, 1:], 'b-', linewidth=2, label=r"$x$ (reactant)")
ax.semilogx(sol_radau.t[1:], sol_radau.y[1, 1:] * 1e4, 'r-', linewidth=2, label=r"$y \times 10^4$ (intermediate)")
ax.semilogx(sol_radau.t[1:], sol_radau.y[2, 1:], 'g-', linewidth=2, label=r"$z$ (product)")
# BDF (dashed)
ax.semilogx(sol_bdf.t[1:], sol_bdf.y[0, 1:], 'b--', linewidth=1.5)
ax.semilogx(sol_bdf.t[1:], sol_bdf.y[1, 1:] * 1e4, 'r--', linewidth=1.5)
ax.semilogx(sol_bdf.t[1:], sol_bdf.y[2, 1:], 'g--', linewidth=1.5)
# RK45 (dotted)
ax.semilogx(sol_rk45.t[1:], sol_rk45.y[0, 1:], 'b:', linewidth=1.5)
ax.semilogx(sol_rk45.t[1:], sol_rk45.y[1, 1:] * 1e4, 'r:', linewidth=1.5)
ax.semilogx(sol_rk45.t[1:], sol_rk45.y[2, 1:], 'g:', linewidth=1.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='center left')

ax = fig.add_subplot(122)
methods_p5 = ["RK45\n(explicit)", "Radau\n(implicit)", "BDF\n(implicit)"]
fevals_p5 = [sol_rk45.nfev, sol_radau.nfev, sol_bdf.nfev]
colors_p5 = ["#e74c3c", "#2ecc71", "#3498db"]
bars = ax.bar(methods_p5, fevals_p5, color=colors_p5, edgecolor="black", width=0.6)
ax.set_ylabel("Function evaluations", size=18)
ax.set_title("Solver efficiency: stiffness penalty" + "\n" + "for explicit RK45 (lower is better)", size=13)
ax.set_yscale('log')
for bar, val in zip(bars, fevals_p5):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
            f"{val:,}", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("problem5.png", dpi=300)
plt.show()