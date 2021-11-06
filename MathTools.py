import math
import sympy as sp
from prettytable import PrettyTable


class Equation:
    def __init__(self, equation):
        self.ALLOWED_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        self.ALLOWED_NAMES.update({'x': 'x'})
        self.ALLOWED_NAMES.update({'y': 'y'})
        self.ALLOWED_NAMES.update({'z': 'z'})
        self.x, y, z = sp.symbols('x y z')

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
        return f'{self.__class__.__name__}('f'{self.eq})'


def simpson_method() -> None:
    answer = []
    print("Solve Integrals with Simpson rule")
    equation = Equation(input("Type equation in python syntax:\t"))
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
        for i in range((2 * n) + 1):
            ans = round(float(equation.solve_for('x', x)), rv)
            if i == 0 or i == 2 * n:
                y0 += ans
                table.add_row([i, round(float(x), rv), ans, ' ', ' '])

            if i % 2 != 0 and i != 0 and i != 2 * n:
                y1 += ans
                table.add_row([i, round(float(x), rv), ' ', ans, ' '])

            if i % 2 == 0 and i != 2 * n and i > 0:
                y2 += ans
                table.add_row([i, round(float(x), rv), ' ', ' ', ans])
            x += h

        table.add_row(['--', '----', '----', '----', '----'])
        table.add_row(['Σx', '=', round(float(y0), rv), round(float(y1), rv), round(float(y2), rv)])
        print(table)
        table.clear()
        print("Σ0 = ", round(float(y0), rv), " | Σ1 = ", round(float(y1), rv), " | Σ2 = ", round(float(y2), rv))
        approx_ans = (h / 3) * (y0 + (4 * y1) + (2 * y2))
        print("S", (2 * n), " = (", round(float(h), rv), '/', 3, ")(Σ0 + 4 * Σ1 + 2 * Σ2) = ", round(float(approx_ans), rv))
        answer.append(approx_ans)

        if epsilon == 0:
            break
        if iteration >= 1:
            precision = round(float(abs(answer[iteration] - answer[iteration - 1])), rv)
            if precision < epsilon:
                print("Precision is good enough", precision)
                break
            else:
                print("Precision not good", precision)
        n *= 2
        iteration = iteration + 1


def newton_method() -> None:
    x = float(input("Type start value for x:\t"))
    n = int(input("Type value for n:\t"))
    f = Equation(input("Type f in python syntax:\t"))

    table = PrettyTable(['i', 'Xi'])
    for i in range(n):
        x = f.solve_for('x', x)
        table.add_row(([i, x, ]))
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
    print(f'\n {round(float(sum(xi_sq)), 4)}a + {round(float(sum(xi)),4)}b = {round(float(sum(xi_yi)), 4)}')
    print(f'\n {round(float(sum(xi)), 4)}a + {n}b = {round(float(sum(yi)), 4)}')
    return


def horner_method():
    n = int(input("Type n: ")) + 1
    col_list = ['x' + str(i) for i in range(n)]
    col_list.reverse()

    poly_list = [[None for _ in range(n)], ['+' for _ in range(n)], [None for _ in range(n)], ['---' for _ in range(n)],
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
    formula = Equation(input("Type equation: \t"))
    if formula.solve_for(a) * formula.solve_for(b) >= 0:
        print("Value for a and b wrong ")
        return
    while True:
        xm = (a + b) / 2
        answer = formula.solve_for('x', xm)
        if answer == 0:
            print(f'Exact Zero point is {xm}')
            break
        elif b - a < epsilon:
            print(f'Approximate Zero point is {xm}')
            break
        else:
            pm = formula.solve_for('x', xm) * formula.solve_for(a)
            if pm > 0:
                a = xm
            elif pm < 0:
                b = xm


def jacobi_method() -> None:
    sp.init_printing(pretty_print=True)
    n = int(input("Type n (Matrix is n x n)\t"))
    if n <= 1:
        print("Wrong Input, aborted")
        return
    raw_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(n):
            print("roiroi")
            print("asldkj")
            raw_matrix[j][i] = float(input(f'Type A{j}{i}\t'))
    m = sp.Matrix(raw_matrix)
    sp.pretty_print(m)
    return


def gauss_seidel_method() -> None:
    return


def print_menu() -> None:
    print("1 Simpson Method")
    print("2 Newtons Method")
    print("3 Best fit line")
    print("4 Horner Method")
    print("5 Biscetive Method")
    print("6 JacobiMethod")
    print("7 Gauss-Seidel Method")
    print("Press 'c' to close")


def main() -> None:
    choice = 'u'
    while choice != 'c':
        print_menu()
        choice = input()

        if choice == '1':
            simpson_method()
        if choice == '2':
            newton_method()
        if choice == '3':
            best_fit_line()
        if choice == '4':
            horner_method()
        if choice == '5':
            bisection_method()
        if choice == '6':
            jacobi_method()
        if choice == '7':
            gauss_seidel_method()

        if choice == 'c':
            return
    return


"""
# only for python 3.10
    match choice:
        case 1:
            SimpsonMethod()
        case 2:
            NewtonMethod()
        case 3:
            BestFitLine()
        case 4:
            HornerMethod()
        case 5:
            BisectionMethod()
        case 6:
            JacobiMethod()
        case 7:
            GaussSeidelMethod()
        case _:
            print("Input Error")
        
"""
if __name__ == "__main__":
    main()
