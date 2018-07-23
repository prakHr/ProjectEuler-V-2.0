def PrimeTable(bound):
    primes=[]
    visited=[False]*(bound+1)
    visited[0]=True
    visited[1]=True
    for i in range(2,bound+1):
        if not visited[i]:
            primes.append(i)
        for j in range(i+i,bound+1,i):
            visited[j]=True
    return primes


def populate(prime_factors,upper_bound):
    rv=[1]
    for d in prime_factors:
        next=[d*x for x in rv if d*x<upper_bound]
        #elements in next list is getting added to rv list
        rv+=next
    rv.sort()
    return rv

def get_upper_bound(n):
    product=1
    for i in n:
        product=product*i
    return int(product**0.5)+1

def pseudo_square_root(prime_factors):
    upper_bound=get_upper_bound(prime_factors)
    half_len=len(prime_factors)//2
    lower_half_array=populate(prime_factors[:half_len],upper_bound)
    upper_half_array=populate(prime_factors[half_len:],upper_bound)
    best_so_far=0
    for n in lower_half_array:

        def binary_search(n,upper_bound):
            L=0
            U=len(upper_half_array)-1
            while L<=U:
                M=(L+U)//2
                #if element in --------
                #              L  M   U
                #if product is less than upperbound,then element is in b/w (M,U)
                if upper_half_array[M]*n < upper_bound:
                    L=M+1
                else:
                    U=M-1
            return upper_half_array[U]*n
        
        x=binary_search(n,upper_bound)
        if x>best_so_far:
            best_so_far=x
    return best_so_far%10**16
        
def Problem_266():
    #factorization=Factorization()
    primes=PrimeTable(190)
    print(pseudo_square_root(primes))

Problem_266()

from math import factorial
def C(m,n):
    return factorial(m)//factorial(n)//factorial(m-n)

