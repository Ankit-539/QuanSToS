import numpy as np

#Generating random Qubit
def random_single_qubit():

    u = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    theta = np.arccos(u)
    psi = np.array([ np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2) ])
    return psi

#Testing qubit generation
psi = random_single_qubit()
print("State:")
print(psi)
print("Norm:")
print(np.vdot(psi, psi))

#Generating random product state
def generate_product_state(n_qubits):
    psi_product = np.array([1.0 + 0j])
    for _ in range(n_qubits):
        single_qubit = random_single_qubit()
        psi_product = np.kron( psi_product, single_qubit )
    return psi_product

#Testing product
for n in [15,28]:
    psi_product = generate_product_state(n)
    print( f"{n} qubits: " f"dimension = {len(psi_product)}, "    f"norm = {np.vdot(psi_product, psi_product).real:.6f}")

#Denisty matrix
def state_to_density_matrix(psi_product):
    return np.outer(  psi_product,  psi_product.conj()    )
def validate_density_matrix(rho):
    trace = np.trace(rho)
    hermitian = np.allclose(rho,  rho.conj().T  )
    purity = np.trace( rho @ rho )
    return trace, hermitian, purity
#Normailzation test
# Product state
psi_product = generate_product_state(3)
rho_product = state_to_density_matrix(psi_product)
print("PRODUCT STATE")
print("Dimension:", len(psi_product))
print("Norm:",   np.vdot( psi_product, psi_product  ).real)
print( "Validation:",  validate_density_matrix(rho_product))
