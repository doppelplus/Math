import math
import sympy as sp
import numpy as np
from prettytable import PrettyTable


class Equation:
    def __init__(self, equation):
        self.ALLOWED_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        self.ALLOWED_NAMES.update({'x': 'x'})
        self.ALLOWED_NAMES.update({'x': 'y'})
        self.ALLOWED_NAMES.update({'x': 'z'})
        self.x = sp.symbols('x')
        self.y = sp.symbols('y')
        self.z = sp.symbols('z')
        # Security: check for malicious string
        evaluate = compile(equation, "<string>", "eval")
        for name in evaluate.co_names:
            if name not in self.ALLOWED_NAMES:
                raise NameError(f"The use of '{name}' is not allowed")
        # All good sympify can be done
        self.eq = sp.sympify(equation)

    def solve_for(self, var, value) -> float:
        return self.eq.subs(var, value)

    def __repr__(self):
        return f'{self.eq.simplify()}'


def simpson_method() -> None:
    answer = []
    print("Solve Integrals with Simpson rule")
    equation_string = input("Type equation in python syntax:\t")
    n = int(input("Type value for n:\t"))
    if n < 1:
        print("Wrong Arguments, aborted!")
        return
    epsilon = float(input("Type epsilon Value like 0.001:\t"))
    rv = int(input("Type precision value like 5:\t"))
    a = float(input("Type value for a:\t"))
    b = float(input("Type value for b:\t"))
    h = float((b - a) / (2 * n))
    x = a

    iteration = 0
    while n <= 100:
        y0 = 0.0
        y1 = 0.0
        y2 = 0.0
        table = PrettyTable(['i', 'Xi', 'y0', 'y1', 'y2'])
        print("\nn = ", n)
        equation = Equation(equation_string)
        for i in range((2 * n) + 1):
            ans = round(float(equation.solve_for('x', x)), rv)
            if i == 0 or i == 2 * n:
                y0 += ans
                table.add_row([i, x, ans, ' ', ' '])

            if i % 2 != 0 and i != 0 and i != 2 * n:
                y1 += ans
                table.add_row([i, x, ' ', ans, ' '])

            if i % 2 == 0 and i != 2 * n and i > 0:
                y2 += ans
                table.add_row([i, x, ' ', ' ', ans])
            x += h

        table.add_row(['--', '----', '----', '----', '----'])
        table.add_row(['Σx', '=', round(float(y0), rv), round(float(y1), rv), round(float(y2), rv)])
        print(table)
        table.clear()

        print("Σ0 = ", round(float(y0), rv), " | Σ1 = ", round(float(y1), rv), " | Σ2 = ", round(float(y2), rv))
        approx_ans = (h / 3) * (y0 + (4 * y1) + (2 * y2))
        print("S", (2 * n), " = (", round(float(h), rv), '/', 3, ")(Σ0 + 4 * Σ1 + 2 * Σ2) = ",
              round(float(approx_ans), rv))
        answer.append(approx_ans)
        if iteration >= 1:
            precision = round(float(abs(answer[iteration] - answer[iteration - 1])), rv)
            if precision < epsilon or epsilon == 0:
                print("Precision is good enough", precision)
                break
            else:
                print("Precision not good", precision)
        n *= 2
        iteration = iteration + 1


def trapeze_method() -> None:
    answer = []
    print("Solve Integrals with Trapez rule")
    equation_string = input("Type equation in python syntax:\t")
    n = int(input("Type value for n:\t"))
    if n < 1:
        print("Wrong Arguments, aborted!")
        return
    epsilon = float(input("Type epsilon Value like 0.001:\t"))
    rv = int(input("Type precision value like 5:\t"))
    a = float(input("Type value for a:\t"))
    b = float(input("Type value for b:\t"))
    h = float((b - a) / n)
    x = a

    iteration = 0
    while n <= 100:
        y0 = 0.0
        y1 = 0.0
        table = PrettyTable(['i', 'Xi', 'y0', 'y1'])
        print("\nn = ", n)
        equation = Equation(equation_string)
        for i in range(n + 1):
            x = (a + i * h)
            ans = round(float(equation.solve_for('x', x)), rv)
            if i == 0 or i == n:
                y0 += ans
                table.add_row([i, round(x, rv), ans, ' '])
            else:
                y1 += ans
                table.add_row([i, round(x, rv), ' ', ans])

        table.add_row(['--', '----', '----', '----'])
        table.add_row(['Σx', '=', round(float(y0), rv), round(float(y1), rv)])
        print(table)
        table.clear()
        print("Σ0 = ", round(float(y0), rv), " | Σ1 = ", round(float(y1), rv))
        approx_ans = (h / 2) * (y0 + (2 * y1))
        print("T", n, " = (", round(float(h), rv), '/', 2, ")(Σ0 + 2 * Σ1 ) = ", round(float(approx_ans), rv))
        answer.append(approx_ans)
        precision = round(float(abs(answer[iteration] - answer[iteration - 1])), rv)
        if precision < epsilon or epsilon == 0:
            print("Precision is good enough", precision)
            break
        elif iteration >= 0:
            print("Precision not good", precision)
        n *= 2
        iteration = iteration + 1


def newton_method() -> None:
    x = float(input("Type start value for x:\t"))
    n = int(input("Type value for n:\t"))
    rv = int(input("Type precision value like 5:\t"))
    epsilon = float(input("Type epsilon Value like 0.001:\t"))
    f = Equation(input("Type f in python syntax:\t"))

    table = PrettyTable(['i', 'Xi'])
    if epsilon == 0:
        for i in range(n):
            x = float(f.solve_for('x', x))
            table.add_row(([i, round(x, rv)]))
    else:
        prev = 0.0
        for i in range(1000):  # make sure the loop has an end
            x = float(f.solve_for('x', x))
            table.add_row(([i, round(x, rv)]))
            if i >= 1:
                precision = abs(x - prev)
                if precision < epsilon:
                    print(f'Precision reached, approximate answer is {x}')
                    break
            prev = x
            i += 1
    print(table)
    return


def best_fit_line() -> None:
    n = int(input("Type n: "))
    xi = []
    yi = []
    xi_sq = []
    xi_yi = []
    # Data Input and generation
    for i in range(n):
        xi.append(float(input(f'Type x{i}: ')))
    for i in range(n):
        yi.append(float(input(f'Type y{i}: ')))
    for i in range(n):
        xi_sq.append(round(xi[i] ** 2))
    for i in range(n):
        xi_yi.append(round(xi[i] * yi[i]))

    # Data Output with sums
    print(f'Σxi = {xi} = {round(float(sum(xi)), 4)}')
    print(f'Σyi = {yi} = {round(float(sum(yi)), 4)}')
    print(f'Σxi² = {xi_sq} = {round(float(sum(xi_sq)), 4)}')
    print(f'Σxiyi = {xi_yi} = {round(float(sum(xi_yi)), 4)}')
    print(f'\n {round(float(sum(xi_sq)), 4)}a + {round(float(sum(xi)), 4)}b = {round(float(sum(xi_yi)), 4)}')
    print(f'\n {round(float(sum(xi)), 4)}a + {n}b = {round(float(sum(yi)), 4)}')
    return


def horner_method():
    n = int(input("Type n: ")) + 1
    col_list = ['x' + str(i) for i in range(n)]
    col_list.reverse()

    poly_list = [[None for _ in range(n)],
                 ['+' for _ in range(n)],
                 [None for _ in range(n)],
                 ['---' for _ in range(n)],
                 [None for _ in range(n)]]

    table = PrettyTable(col_list)
    for i in range(n):
        poly_list[0][i] = (float(input(f'Type x^{n - 1 - i}  ')))
    divisor = float(input("Type divisor:\t"))
    poly_list[2][0] = 0

    for i in range(n):
        poly_list[4][i] = poly_list[0][i] + poly_list[2][i]
        if i < n - 1:
            poly_list[2][i + 1] = poly_list[4][i] * divisor

    for i in range(5):
        table.add_row(poly_list[i])
    print(table)

    return


def bisection_method() -> None:
    a = float(input("Type a: \t"))
    b = float(input("Type b: \t"))
    epsilon = float(input("Type epsilon: \t"))
    rv = int(input("Type precision value: \t"))
    formula = Equation(input("Type equation: \t"))
    iterations = 0
    if formula.solve_for('x', a) * formula.solve_for('x', b) >= 0:
        print("Value for a and b wrong ")
        return
    print(f'Expected iterations: 1 + (precision + log(b-a)/log(2)= '
          f'{math.ceil(1 + (rv + math.log(b - a, 10)) / math.log(2, 10))}')
    table = PrettyTable(['n', 'an', 'bn', 'xn', 'f(an)', 'f(bn)', 'f(xn)'])
    table.align = "l"
    while True:

        xm = (a + b) / 2
        answer = formula.solve_for('x', xm)
        table.add_row([iterations, round(float(a), rv),
                       round(float(b), rv),
                       round(float(xm), rv),
                       round(float(formula.solve_for('x', a)), rv),
                       round(float(formula.solve_for('x', b)), rv),
                       round(float(answer), rv)])
        if answer == 0:
            print(f'Exact Zero point is {xm}')
            break
        elif b - a < epsilon:
            print(f'Approximate Zero point is {round(float(xm), rv)} after {iterations} iterations')
            break
        else:
            pm = formula.solve_for('x', xm) * formula.solve_for('x', a)
            if pm > 0:
                a = xm
            elif pm < 0:
                b = xm

        iterations += 1

    print(table)


def jacobi_method() -> None:
    sp.init_printing(pretty_print=True)
    mn = int(input("Type n (Matrix is n x n)\t"))
    n = int(input("Type iterations\t"))
    raw_vector = [0.0 for _ in range(mn)]
    table_columns = [f'x{_}' for _ in range(mn)]
    table = PrettyTable(['k', table_columns, 'max_i | xi^(k-1)-xi^k'])
    for i in range(mn):
        raw_vector[i] = float(input(f'Type Start vector\t {i}\t'))
    if mn <= 1:
        print("Wrong Input, aborted")
        return
    raw_matrix_a = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_l = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_d = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_r = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_b = [0.0 for _ in range(mn)]

    # Input for matrices A, D, L, R
    for j in range(mn):
        for i in range(mn):
            raw_matrix_a[j][i] = float(input(f'Type A{j}{i}\t'))
            if i == j:
                raw_matrix_d[j][i] = raw_matrix_a[j][i]
            if j > i:
                raw_matrix_l[j][i] = raw_matrix_a[j][i]
            if j < i:
                raw_matrix_r[j][i] = raw_matrix_a[j][i]

    # Input for vector b
    for i in range(mn):
        raw_matrix_b[i] = float(input(f'Type b{i}\t'))

    A = sp.Matrix(raw_matrix_a)
    b = sp.Matrix(raw_matrix_b)
    L = sp.Matrix(raw_matrix_l)
    R = sp.Matrix(raw_matrix_r)
    D = sp.Matrix(raw_matrix_d)
    G = -D.inv() * (L + R)
    x = sp.Matrix(raw_vector)
    d = D.inv() * b
    print("G = ")
    sp.pprint(G)
    print("d = ")
    sp.pprint(d)

    # Jacobi algorithm
    for i in range(n):
        prev_x = x
        x = G * x + d
        if i >= 1:
            table.add_row([i, x, max(abs(x - prev_x))])
        else:
            table.add_row([i, x, ""])
    print(f'Approximate solution {x}')
    print(table)


def gauss_seidel_method() -> None:
    mn = int(input("Type n (Matrix is n x n)\t"))
    n = int(input("Type iterations\t"))
    raw_vector = [0.0 for _ in range(mn)]
    table_columns = [f'x{_}' for _ in range(mn)]
    table = PrettyTable(['k', table_columns, 'max_i | xi^(k-1)-xi^k'])
    for i in range(mn):
        raw_vector[i] = float(input(f'Type Start vector\t {i}\t'))
    if mn <= 1:
        print("Wrong Input, aborted")
        return
    raw_matrix_a = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_l = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_d = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_r = [[0.0 for _ in range(mn)] for _ in range(mn)]
    raw_matrix_b = [0.0 for _ in range(mn)]

    # Input for matrices A, D, L, R
    for j in range(mn):
        for i in range(mn):
            raw_matrix_a[j][i] = float(input(f'Type A{j}{i}\t'))
            if i == j:
                raw_matrix_d[j][i] = raw_matrix_a[j][i]
            if j > i:
                raw_matrix_l[j][i] = raw_matrix_a[j][i]
            if j < i:
                raw_matrix_r[j][i] = raw_matrix_a[j][i]

    # Input for vector b
    for i in range(mn):
        raw_matrix_b[i] = float(input(f'Type b{i}\t'))

    A = sp.Matrix(raw_matrix_a)
    b = sp.Matrix(raw_matrix_b)
    L = sp.Matrix(raw_matrix_l)
    R = sp.Matrix(raw_matrix_r)
    D = sp.Matrix(raw_matrix_d)
    S = sp.Matrix(-(L + D)).inv() * R
    d = sp.Matrix(L + D).inv() * b
    x = sp.Matrix(raw_vector)
    print("S = ")
    sp.pprint(S)
    print("d = ")
    sp.pprint(d)
    for i in range(n):
        prev_x = x
        x = S * x + d
        if i >= 1:
            table.add_row([i, x, max(abs(x - prev_x))])
        else:
            table.add_row([i, x, ""])
    print(f'Approximate solution {x}')
    print(table)


def lagrange_interpolation() -> None:
    n = int(input("Type number of Points:\t"))
    points = []
    lg_poly_str = ["" for _ in range(n)]
    lg_poly_equations = []
    lagrange_equation_str = ""
    for i in range(n):
        a = input(f'Type point_{i} separated with comma\t')
        points.append(tuple(float(x) for x in a.split(",")))
    print("\n")
    for i in range(n):
        for j in range(n):
            if i != j:
                lg_poly_str[i] += f'((x-{points[j][0]})/({points[i][0]}-{points[j][0]}))*'
        lg_poly_str[i] += f'{points[i][1]}'

    for i in range(n):
        lg_poly_equations.append(Equation(lg_poly_str[i]))
        lagrange_equation_str += f'{lg_poly_str[i]} + '
        print(f'Polynom L_{i} = {lg_poly_equations[i]})')
    lagrange_equation = Equation(lagrange_equation_str[:-2])
    print(f'P_{n-1}(x) = {lagrange_equation}\n')


def _poly_newton_coefficient(x, y):
    """
    x: list or np array contanining x data points
    y: list or np array contanining y data points
    """

    m = len(x)

    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])

    return a


def newton_polynomial(x_data, y_data, x):
    """
    x_data: data points at x
    y_data: data points at y
    x: evaluation point(s)
    """
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]

    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p

    return p


def newton_interpolation() -> None:
    n = int(input("Type number of Points:\t"))
    coefficients = [[0.0 for _ in range(n)]for _ in range(n+1)]
    # input
    for i in range(n):
        a = input(f'Type point_{i} separated with comma\t')
        tmp = a.split(",")
        if not len(tmp) == 2:
            print("Input Error")
            return
        coefficients[0][i] = float(tmp[0])
        coefficients[1][i] = float(tmp[1])

    for j in range(2, n):
        for i in range(n-1):
            coefficients[j][i] = (coefficients[j - 1][i] - coefficients[j-1][i + 1]) / (coefficients[j-2][i] - coefficients[j-2][i + 1])
            print(f'Answer[{j}][{i}]: {coefficients[j][i]} = {coefficients[j - 1][i] } - {coefficients[j-1][i + 1]} / {coefficients[j-2][i]} - {coefficients[j-2][i + 1]}')

    print("")

    print("\n")


def print_menu() -> None:
    print("\n"
          "0 Trapeze Method\n"
          "1 Simpson Method\n"
          "2 Newtons Method\n"
          "3 Best fit line\n"
          "4 Horner Method\n"
          "5 Biscetive Method\n"
          "6 JacobiMethod\n"
          "7 Gauss-Seidel Method\n"
          "8 Lagrange interpolation\n"
          "9 Newton interpolation\n"
          "Press 'c' to close")


def main() -> None:
    choice = 'u'
    while choice != 'c':
        print_menu()
        choice = input()

        match choice:
            case 0:
                trapeze_method()
            case 1:
                simpson_method()
            case 2:
                newton_method()
            case 3:
                best_fit_line()
            case 4:
                horner_method()
            case 5:
                bisection_method()
            case 6:
                jacobi_method()
            case 7:
                gauss_seidel_method()
            case 8:
                lagrange_interpolation()
            case 9:
                newton_interpolation()
            case _:
                print("Input Error")


if __name__ == "__main__":
    main()

