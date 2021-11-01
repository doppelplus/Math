import math
import sympy
from prettytable import PrettyTable


class Formula:
    def __init__(self, formula, xVar=0.0):
        self.formula = formula
        self.x = xVar
        self.ALLOWED_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        self.ALLOWED_NAMES.update({'x': self.x})
        self.code = self.Evaluate(formula)

    def Evaluate(self, expression):
        evalCode = compile(expression, "<string>", "eval")

        # Validate allowed names

        for name in evalCode.co_names:

            if name not in self.ALLOWED_NAMES:
                raise NameError(f"The use of '{name}' is not allowed")
        return evalCode

    def Solve(self, xVar) -> float:
        self.x = xVar
        self.ALLOWED_NAMES.update({'x': self.x})
        return eval(self.code, {"__builtins__": {}}, self.ALLOWED_NAMES)

    def __repr__(self):
        return (f'{self.__class__.__name__}('f'{self.formula}, x = {self.x})')


def simpson_solver(formula, n, a, b, precision):
    if n < 1:
        print("Wrong Arguments, aborted!")
        return
    h = float((b - a) / (2 * n))
    x = a
    y0 = 0.0
    y1 = 0.0
    y2 = 0.0
    table = PrettyTable(['i', 'Xi', 'y0', 'y1', 'y2'])

    formula = Formula(formula, x)
    for i in range((2 * n) + 1):
        answer = formula.Solve(x)
        if i == 0 or i == 2 * n:
            y0 += answer
            table.add_row([i, x, answer, ' ', ' '])

        if i % 2 != 0 and i != 0 and i != 2 * n:
            y1 += answer
            table.add_row([i, x, ' ', answer, ' '])

        if i % 2 == 0 and i != 2 * n and i > 0:
            y2 += answer
            table.add_row([i, x, ' ', ' ', answer])

        x += h
    y0 = round(y0, precision)
    y1 = round(y1, precision)
    y2 = round(y2, precision)

    table.add_row(['--', '----', '----', '----', '----'])
    table.add_row(['Σx', '=', y0, y1, y2])
    print(table)
    print("Σ0 = ", y0, " | Σ1 = ", y1, " | Σ2 = ", y2)
    aproxAnswer = (h / 3) * (y0 + (4 * y1) + (2 * y2))
    print("S", (2 * n), " = (", h, '/', 3, ")(Σ0 + 4 * Σ1 + 2 * Σ2) = ", round(aproxAnswer, precision))

    return aproxAnswer


def simpson_method() -> None:
    answer = []
    print("Solve Integrals with Simpson rule")
    formula = input("Type formula in python syntax:\t")
    n = int(input(" Type value for n:\t"))
    epsilon = float(input("Type epsilon Value like 0.001:\t"))
    a = float(input("Type value for a:\t"))
    b = float(input("Type value for b:\t"))

    iteration = 0
    while n <= 100:
        print("\nn = ", n)
        answer.append(simpson_solver(formula, n, a, b, 4))
        if iteration >= 1:
            precision = round(abs(answer[iteration] - answer[iteration - 1]), 5)
            if precision < epsilon:
                print("Precision is good enough", precision)
                break
            else:
                print("Precision not good", precision)
        n *= 2
        iteration = iteration + 1


def newton_method() -> None:
    fstring = input("Type f in python syntax:\t")
    x = float(input("Type start value for x:\t"))
    n = int(input("Type value for n:\t"))

    f = Formula(fstring, x)
    table = PrettyTable(['i', 'Xi'])
    for i in range(n):
        x = f.Solve(x)
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
    print(f'Σxi = {xi} = {round(sum(xi), 4)}')
    print(f'Σyi = {yi} = {round(sum(yi), 4)}')
    print(f'Σxi² = {xi_sq} = {round(sum(xi_sq), 4)}')
    print(f'Σxiyi = {xi_yi} = {round(sum(xi_yi), 4)}')
    print(f'\n {round(sum(xi_sq), 4)}a + {round(sum(xi))}b = {round(sum(xi_yi), 4)}')
    print(f'\n {round(sum(xi), 4)}a + {n}b = {round(sum(yi), 4)}')
    return


def horner_method():
    n = int(input("Type n: ")) + 1
    divisor = 1
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
    formula = Formula(input("Type Formula: \t"))
    if formula.Solve(a) * formula.Solve(b) >= 0:
        print("Value for a and b wrong ")
        return
    while True:
        xm = (a + b) / 2
        answer = formula.Solve(xm)
        pm = 0.0
        if answer == 0:
            print(f'Exact Zero point is {xm}')
            break
        elif b - a < epsilon:
            print(f'Approximate Zero point is {xm}')
            break
        else:
            pm = formula.Solve(xm) * formula.Solve(a)
            if pm > 0:
                a = xm
            elif pm < 0:
                b = xm


def jacobi_method() -> None:
    sympy.init_printing(pretty_print=True)
    n = int(input("Type n (Matrix is n x n)\t"))
    if n <= 1:
        print("Wrong Input, aborted")
        return
    raw_matrix = [[0.0 for x in range(n)] for y in range(n)]
    for j in range(n):
        for i in range(n):
            print("roiroi")
            raw_matrix[j][i] = float(input(f'Type A{j}{i}\t'))
    m = sympy.Matrix(raw_matrix)
    sympy.pretty_print(m)
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


def main():
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
