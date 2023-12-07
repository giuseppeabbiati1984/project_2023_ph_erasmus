#%%

#should use sympy, because numpy is for numerical purposes and we need to stick with symoblic for now
import sympy as sm

x1,y1,x2,y2,q1,q2,chi_x,chi_y = sm.symbols('x_1,y_1,x_2,y_2,q_1,q_2,chi_x,chi_y')

# eq 4.31 Taghirad
f = sm.Matrix([q1 - sm.sqrt((chi_x-x1)**2 + (chi_y-y1)**2) - sm.sqrt(x1**2 + y1**2),q2 - sm.sqrt((chi_x-x2)**2 + (chi_y-y2)**2) - sm.sqrt(x2**2 + y2**2)])
q = sm.Matrix([q1,q2])
chi = sm.Matrix([chi_x,chi_y])
Jq = -f.jacobian(q)
Jchi = f.jacobian(chi)
J = Jq.inv()*Jchi

# generation of the lambdified function
lambda_J = sm.lambdify((chi_x, chi_y, x1, x2, y1, y2),[J,Jq,Jchi])

#%% numerical test

x1_val = -0.6653
y1_val = 0.0

y2_val = -0.5840
x2_val = 0.0

chi_x_val = 0.10
chi_y_val = 0.05

J_val,Jq_val,Jchi_val = lambda_J(chi_x_val,chi_y_val,x1_val,x2_val,y1_val,y2_val)
#%%

import dill

with open('lambda_J_dill.pkl', 'wb') as file:
    dill.dump(lambda_J, file)

file.close()
# %%

with open('lambda_J_dill.pkl', 'rb') as file:
    lambda_J_chk = dill.load(file)

file.close()

# here I check that saving/loading with dill works properly
J_val1,Jq_val1,Jchi_val1 = lambda_J(chi_x_val,chi_y_val,x1_val,x2_val,y1_val,y2_val)
J_val2,Jq_val2,Jchi_val2 = lambda_J_chk(chi_x_val,chi_y_val,x1_val,x2_val,y1_val,y2_val)

print(J_val1)
print(J_val2)
# %%
