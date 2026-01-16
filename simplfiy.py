import sympy as sp

x, y = sp.symbols('x y')
a, b, m12, m13, m23 = sp.symbols('a b m12 m13 m23')

f4 = x**2 + (1 - a)*y**2 - 2*m12*x*y + 2*a*m23*y - a
f5 = x**2 - b*y**2 - 2*m13*x + 2*b*m23*y + 1 - b

g = sp.simplify(f5 - f4)  # = 2*m12*x*y - 2*m13*x - (1+b-a)*y**2 + 2*(b-a)*m23*y + (1-b+a)

x_of_y = sp.solve(sp.Eq(g, 0), x)[0]  

den = sp.simplify(sp.denom(sp.together(x_of_y)))   
num = sp.simplify(sp.numer(sp.together(x_of_y)))  
f5_sub = sp.simplify(f5.subs(x, x_of_y))

poly_y = sp.expand(sp.together(f5_sub) * den**2)
poly_y_collected = sp.collect(poly_y, y)
coeffs = [sp.expand(sp.poly(poly_y_collected, y).coeff_monomial(y**k)) for k in range(4, -1, -1)]

print("係数:");  [print(f"c{k} =", c) for k,c in zip(range(4, -1, -1), coeffs)]

