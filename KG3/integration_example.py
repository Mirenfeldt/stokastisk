import sympy as sp


def main():
    x, k = sp.symbols("x k")
    fx = -k * x**2 + k

    integral = sp.integrate(fx, (x, -1, 1))

    # Set the integral equal to 1 (PDF condition)
    equation = sp.Eq(integral, 1)

    # Solve for k
    solution = sp.solve(equation, k)
    print(f"{integral=}")
    print("Value of k:", solution)


if __name__ == "__main__":
    main()
