import sympy

sympy.init_printing()
x,y = sympy.symbols('x y') 
expr= sympy.sqrt(1/x)
erg = sympy.integrate(expr,(x) )
sympy.pretty_print(erg)