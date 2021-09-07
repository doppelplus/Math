import math
import numpy
import matplotlib.pyplot as plt
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


def SimpsonRule(formula, n, a,b, precision)->None:
    if n < 1:
        print("Wrong Arguments, aborted!")
        return
    h = float((b - a) /  (2*n))
    x = a
    y0 = []
    y1 = []
    y2 = []
    xList = []
    formula = Formula(formula, x)
    for i in range((2*n)+1):
        if i == 0 or i == 2 * n:
            y0.append(round(formula.Solve(x),precision))
        if i % 2 != 0 and i != 0 and i != 2*n:
            y1.append(round(formula.Solve(x),precision))
        if i % 2 == 0 and i != 2*n and i > 0:
            y2.append(round(formula.Solve(x),precision))
        
        xList.append(round(x,precision))
        x = x + h
    
    sum0 = round(sum(y0),precision)
    sum1 = round(sum(y1),precision)
    sum2 = round(sum(y2),precision)
    table = PrettyTable(['i', 'Xi','y0','y1','y2'])

    y0list = 0
    y1list = 0
    y2list = 0
    for i in range((2*n)+1):
        if i == 0 or i == 2 * n:
            table.add_row([i,xList[i],y0[y0list],' ',' '])
            y0list += 1

        if i % 2 != 0 and i != 0 and i != 2*n: 
            table.add_row([i,xList[i],' ',y1[y1list],' '])
            y1list += 1
        
        if i % 2 == 0 and i != 2*n and i > 0:
            table.add_row([i,xList[i],' ',' ',y2[y2list]])
            y2list += 1

    table.add_row(['--','----','----','----','----'])        
    table.add_row(['Σx','=',sum0,sum1,sum2])
    print(table)
    print("Σ0 = ", sum0," | Σ1 = ", sum1," | Σ2 = ", sum2 )
    aproxAnswer = (h/3)*(sum0 + (4*sum1)+(2*sum2))
    print("S",(2*n)," = (",h,'/',3,")(Σ0 + 4 * Σ1 + 2 * Σ2) = ",round(aproxAnswer,precision))
    
    return aproxAnswer


def main():
    answer =[]
    epsilon  = 0.0001
    n = 1
    a = 0.0
    b = 0.0
    formula = ""
    print("Solve Integrals with Simpson rule")
    formula = input("Type formula:\t")
    n = int(input(" Type value for n:\t"))
    epsilon = float(input("Type epsilon Value like 0.001:\t"))
    a = float(input("Type value for a:\t"))
    b = float(input("Type value for b:\t"))
    
    iteration = 0
    while n <= 100:
        print("\nn = ",n)
        answer.append(SimpsonRule(formula,n,a,b,6))
        if iteration >= 1:
            precision = round(abs(answer[iteration]-answer[iteration-1]),5)
            if precision < epsilon:
                print("Precision is good enough",precision)
                break
            else:
                print("Precision not good", precision)
        n *= 2
        iteration = iteration + 1 


        
if __name__ == "__main__":
    main()
 
