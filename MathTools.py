import math
from prettytable import PrettyTable

class Formula:
    def __init__(self,formula, xVar):
        self.formula = formula
        self.x = xVar
        self.ALLOWED_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        self.ALLOWED_NAMES.update({'x':self.x})
        self.code = self.Evaluate(formula)
        
    
    def Evaluate(self, expression):
        evalCode = compile(expression, "<string>", "eval")

        # Validate allowed names

        for name in evalCode.co_names:

            if name not in self.ALLOWED_NAMES:

                raise NameError(f"The use of '{name}' is not allowed")
        return evalCode
    
    def Solve(self, xVar)->float:
        self.x = xVar
        self.ALLOWED_NAMES.update({'x':self.x})
        return eval(self.code, {"__builtins__": {}}, self.ALLOWED_NAMES)

    def __repr__(self):
        return (f'{self.__class__.__name__}('f'{self.formula}, x = {self.x})')


def SimpsonSolver(formula, n, a,b,precision):
    if n < 1:
        print("Wrong Arguments, aborted!")
        return
    h = float((b - a) /  (2*n))
    x = a
    y0 = 0.0
    y1 = 0.0
    y2 = 0.0
    table = PrettyTable(['i', 'Xi','y0','y1','y2'])

    formula = Formula(formula, x)
    for i in range((2*n)+1):
        answer = formula.Solve(x)
        if i == 0 or i == 2 * n:
            y0 += answer
            table.add_row([i,x,answer,' ',' '])

        if i % 2 != 0 and i != 0 and i != 2*n:
            y1 += answer
            table.add_row([i,x,' ',answer,' '])

        if i % 2 == 0 and i != 2*n and i > 0:
            y2 += answer
            table.add_row([i,x,' ',' ',answer])
        
        x += h
    y0 = round(y0,precision)
    y1 = round(y1,precision)
    y2 = round(y2,precision)

    table.add_row(['--','----','----','----','----'])        
    table.add_row(['Σx','=',y0,y1,y2])
    print(table)
    print("Σ0 = ", y0," | Σ1 = ", y1," | Σ2 = ", y2 )
    aproxAnswer = (h/3)*(y0 + (4*y1)+(2*y2))
    print("S",(2*n)," = (",h,'/',3,")(Σ0 + 4 * Σ1 + 2 * Σ2) = ",round(aproxAnswer,precision))
    
    return aproxAnswer

def SimpsonCalculator()->None:
    answer =[]
    epsilon  = 0.0001
    n = 1
    a = 0.0
    b = 0.0
    formula = ""
    print("Solve Integrals with Simpson rule")
    formula = input("Type formula in python syntax:\t")
    n = int(input(" Type value for n:\t"))
    epsilon = float(input("Type epsilon Value like 0.001:\t"))
    a = float(input("Type value for a:\t"))
    b = float(input("Type value for b:\t"))
    
    iteration = 0
    while n <= 100:
        print("\nn = ",n)
        answer.append(SimpsonSolver(formula,n,a,b,4))
        if iteration >= 1:
            precision = round(abs(answer[iteration]-answer[iteration-1]),5)
            if precision < epsilon:
                print("Precision is good enough",precision)
                break
            else:
                print("Precision not good", precision)
        n *= 2
        iteration = iteration + 1 

def NewtonCalculator():
    fString = input("Type f in python syntax:\t")
    x = float(input("Type start value for x:\t"))
    n = int(input("Type value for n:\t"))
    
    f = Formula(fString, x)
    table = PrettyTable(['i', 'Xi'])
    for i in range(n):
        x = f.Solve(x)
        table.add_row(([i,x,]))
    print(table)
    return

def BestFitLine():
    n = int(input("Type n: "))
    Xi = []
    Yi = []
    XiSq = []
    XiYi = []
    #Data Input and generation
    for i in range(n):
        Xi.append(float(input(f'Type x{i}: ')))
    for i in range(n):
        Yi.append(float(input(f'Type y{i}: ')))
    for i in range(n):
        XiSq.append(round(Xi[i] ** 2))
    for i in range(n):
        XiYi.append(round(Xi[i] * Yi[i]))
    
    #Data Output with sums
    print(f'Σxi = {Xi} = {round(sum(Xi),4)}')
    print(f'Σyi = {Yi} = {round(sum(Yi),4)}')
    print(f'Σxi² = {XiSq} = {round(sum(XiSq),4)}')
    print(f'Σxiyi = {XiYi} = {round(sum(XiYi),4)}')
    print(f'\n {round(sum(XiSq),4)}a + {round(sum(Xi))}b = {round(sum(XiYi),4)}')
    print(f'\n {round(sum(Xi),4)}a + {n}b = {round(sum(Yi),4)}')
    return

def main():
    print("Simpson Calculator\t 1")
    print("Newtons Method\t 2")
    print("Best fit line\t 3")
    choice = int(input())

    if choice == 1:
        SimpsonCalculator()
    if choice == 2:
        NewtonCalculator()
    if choice == 3:
        BestFitLine() 

        
if __name__ == "__main__":
    main()