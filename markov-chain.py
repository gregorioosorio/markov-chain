import numpy as np
import scipy.linalg

# creamos un dictionario que contiene la etiqueta del estado correspondiente
label = {
    0 : "Hamburguesa",
    1 : "Pizza",
    2 : "Perro Caliente"
}
label

# creamos la matriz de transición
A = np.array([[0.2, 0.6, 0.2], [0.3, 0.0, 0.7], [0.5, 0.0, 0.5]])
A

def run_simulation(A, n, start_state, label):
    curr_state = start_state
    simulation = []

    i = 0
    while i < n:
        simulation.append(label[curr_state])
        curr_state = np.random.choice(range(A.shape[1]), p=A[curr_state])
        i += 1
    
    return simulation

print("*** Simulación empezando con {} ***".format(label[0]))
print(run_simulation(A, 15, 0, label))

print("*** Simulación empezando con {} ***".format(label[2]))
print(run_simulation(A, 15, 1, label))

print("*** Simulación empezando con {} ***".format(label[2]))
print(run_simulation(A, 15, 2, label))

# Calcular la distribución estacionaria usando una simulacón de Monte Carlo

def monte_carlo(A, n, start_state):
    pi =  np.array([0, 0, 0])
    curr_state = start_state

    i = 0
    while i < n:
        pi[curr_state] += 1
        curr_state = np.random.choice(range(A.shape[1]), p=A[curr_state])
        i += 1

    pi = pi/n

    return pi

print("*** Monte Carlo con 10 ***")
print(monte_carlo(A, 10, 0))
print("*** Monte Carlo con 100 ***")
print(monte_carlo(A, 100, 0))
print("*** Monte Carlo con 1000 ***")
print(monte_carlo(A, 1000, 0))
print("*** Monte Carlo con 10**6 ***")
print(monte_carlo(A, 10**6, 0))

# Encontrar el autovector izquierdo
def autovector(A):
    values, left = scipy.linalg.eig(A, right = False, left = True)
    print("autovector izquierdo = \n", left, "\n")
    print("auto valor = \n", values)
    pi = left[:,0]
    print(np.sum(pi))
    pi_normalized = [(x/np.sum(pi)).real for x in pi]
    return pi_normalized

pi_normalized = autovector(A)
print("Distribución estacionaria usando autovectores: ", pi_normalized)