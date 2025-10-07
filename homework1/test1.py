from cmath import nan, sqrt
import pathlib
import copy

def extract(line:str, variable:str, mylist:list)->str:
    lhs,rhs=line.split(variable)
    lhs=lhs.replace('+','')
    if(not lhs):
        mylist.append(1)
    elif(lhs[0]=='-'):
        if(len(lhs)==1):
            mylist.append(-1)
        else:
            mylist.append(float(lhs[1:]))
    else:
        mylist.append(float(lhs))
    return rhs

# 1 
def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A=[]
    B=[]
    with path.open() as f:
        lines=f.readlines()
        for line in lines :
            line=line.strip()
            if not line:
                continue
            lhs,rhs=line.split("=")
            B.append(float(rhs))
            L=[]
            lhs=lhs.replace(' ', '')
            lhs=extract(lhs, "x", L)
            lhs=extract(lhs, "y", L)
            lhs=extract(lhs, "z", L)
            A.append(L)
    return A, B

# 2
#2.1
def determinant(matrix: list[list[float]]) -> float:
    return matrix[0][0]*((matrix[1][1]*matrix[2][2])-(matrix[1][2]*matrix[2][1]))-matrix[0][1]*((matrix[1][0]*matrix[2][2])-(matrix[1][2]*matrix[2][0]))+matrix[0][2]*((matrix[1][0]*matrix[2][1])-(matrix[1][1]*matrix[2][0]))

#2.2
def trace(matrix: list[list[float]]) -> float:
    x=0
    if (len(matrix)!=len(matrix[0])):
        return nan
    for i in range(len(matrix)):
        x+=matrix[i][i]
    return x
#2.3
def norm(vector: list[float]) -> float:
    res=0
    for i in vector:
        res+=i**2
    return sqrt (res).real

#2.4
def transpose(matrix: list[list[float]]) -> list[list[float]]:
    T=[]
    for i in range(len(matrix)):
        L=[]
        for j in range(len(matrix[0])):
            L.append(matrix[j][i])
        T.append(L)
    return T

#2.5
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
        result=[]
        if (len(matrix[0])!=len(vector)):
            return []
        for i in range(len(matrix)):
            x=0
            for j in range(len(matrix[0])):
                x+=matrix[i][j]*vector[j]
            result.append(x)
        return result


#3
def replace_column(matrix: list[list[float]], vector: list[float], index:int)->list[list[float]]:
    new_matrix=copy.deepcopy(matrix)
    for i in range(len(vector)):
        new_matrix[i][index]=vector[i]
    return new_matrix

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    L=[]
    L.append(determinant(replace_column(matrix, vector, 0))/determinant(matrix))
    L.append(determinant(replace_column(matrix, vector, 1))/determinant(matrix))
    L.append(determinant(replace_column(matrix, vector, 2))/determinant(matrix))
    return L

#4

def minor(matrix: list[list[float]], row:int, coloumn:int)->list[list[float]]:
    rows=list(range(len(matrix)))
    coloumns=list(rows.copy())
    rows.remove(row)
    coloumns.remove(coloumn)
    new_matrix=[]
    for i in rows:
        L=[]
        for j in coloumns:
            L.append(matrix[i][j])
        new_matrix.append(L)
    return new_matrix

def cofactor(matrix: list[list[float]]) -> list[list[float]]:

    T=[]
    for i in range(len(matrix)):
        L=[]
        for j in range(len(matrix[0])):
            temp_matrix=minor(matrix, i, j)
            L.append(((-1)**(i+j))*(temp_matrix[0][0]*temp_matrix[1][1]-temp_matrix[0][1]*temp_matrix[1][0]))
        T.append(L)
    return T

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(copy.deepcopy(matrix)))

def inverse(matrix:list[list[float]])->list[list[float]]:
    inverse_matrix=[]
    coeff=1/determinant(matrix)
    adjoint_matrix=adjoint(copy.deepcopy(matrix))
    for i in adjoint_matrix:
        tmp=[]
        for j in i:
            tmp.append(j/coeff)
        inverse_matrix.append(tmp)
    return inverse_matrix

def solve_cramer(matrix:list[list[float]], vector:list[float]) ->list[float]:
    return multiply(inverse(matrix),vector)

A, B = load_system(pathlib.Path("data.txt"))
print(A, B)
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")