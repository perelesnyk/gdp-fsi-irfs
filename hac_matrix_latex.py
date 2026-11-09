import numpy as np

# HAC covariance matrix
HAC_matrix = np.array([
    # bL (D(FSI(-1)) linear)
    [0.0002222572565580477, -2.481779082161086e-06, -5.08352192106298e-05, 
     -0.0002267137808948176, 1.030834157974472e-05, -0.0002898723851918612],
    # cL (C linear)
    [-2.481779082161086e-06, 8.507526553601566e-06, 7.966157034518069e-06,
     1.209800074711827e-07, -7.867716489486271e-06, -6.05818704236191e-05],
    # aL (DLOG(RGDP(-1)) linear)
    [-5.08352192106298e-05, 7.966157034518069e-06, 0.0004538705072446232,
     8.752837383469131e-05, -1.539527490068881e-05, 0.0002569638771119813],
    # bN (D(FSI(-1)) nonlinear)
    [-0.0002267137808948176, 1.209800074711827e-07, 8.752837383469131e-05,
     0.0002598861717734153, -7.276884014954251e-06, 0.0003731099336860608],
    # cN (C nonlinear)
    [1.030834157974472e-05, -7.867716489486271e-06, -1.539527490068881e-05,
     -7.276884014954251e-06, 1.029641793088462e-05, -1.977877869267386e-05],
    # aN (DLOG(RGDP(-1)) nonlinear)
    [-0.0002898723851918612, -6.05818704236191e-05, 0.0002569638771119813,
     0.0003731099336860608, -1.977877869267386e-05, 0.003484345030981168]
])

# Generate LaTeX code
print("\\begin{equation}")
print("\\mathbf{\\Sigma}_{HAC} = ")
print("\\begin{bmatrix}")

for i in range(6):
    row_str = " & ".join([f"{HAC_matrix[i,j]:.6e}" for j in range(6)])
    if i < 5:
        print(f"{row_str} \\\\")
    else:
        print(f"{row_str}")

print("\\end{bmatrix}")
print("\\end{equation}")

print("\n% With row/column labels:\n")

print("\\begin{equation}")
print("\\mathbf{\\Sigma}_{HAC} = \\begin{array}{c|cccccc}")
print("& \\alpha_1 & \\alpha_0 & \\alpha_2 & \\beta_1 & \\beta_0 & \\beta_2 \\\\")
print("\\hline")

labels = ["\\alpha_1", "\\alpha_0", "\\alpha_2", "\\beta_1", "\\beta_0", "\\beta_2"]
for i, label in enumerate(labels):
    row_str = " & ".join([f"{HAC_matrix[i,j]:.6e}" for j in range(6)])
    if i < 5:
        print(f"{label} & {row_str} \\\\")
    else:
        print(f"{label} & {row_str}")

print("\\end{array}")
print("\\end{equation}")

print("\n% Scaled (×10^4) for readability:\n")

HAC_scaled = HAC_matrix * 10000

print("\\begin{equation}")
print("\\mathbf{\\Sigma}_{HAC} \\times 10^4 = \\begin{array}{c|cccccc}")
print("& \\alpha_1 & \\alpha_0 & \\alpha_2 & \\beta_1 & \\beta_0 & \\beta_2 \\\\")
print("\\hline")

for i, label in enumerate(labels):
    row_str = " & ".join([f"{HAC_scaled[i,j]:.4f}" for j in range(6)])
    if i < 5:
        print(f"{label} & {row_str} \\\\")
    else:
        print(f"{label} & {row_str}")

print("\\end{array}")
print("\\end{equation}")

print("\n% Note: α₁ = bL (D(FSI(-1)) linear), α₀ = cL (intercept linear), α₂ = aL (DLOG(RGDP(-1)) linear)")
print("% β₁ = bN (D(FSI(-1)) nonlinear), β₀ = cN (intercept nonlinear), β₂ = aN (DLOG(RGDP(-1)) nonlinear)")
