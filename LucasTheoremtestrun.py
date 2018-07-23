import math
def extendedGCD(a,b):
    (x,y)=(0,1)
    (last_x,last_y)=(1,0)
    while b!=0:
        (q,r)=divmod(a,b)
        (a,b)=(b,r)
        (x,last_x)=(last_x-q*x,x)
        (y,last_y)=(last_x-q*y,y)
    return (last_x,last_y)

def solve_equations(a,m,b,n):
    q=m*n
    (x,y)=extendedGCD(m,n)
    root=a+(b-a)*x*m
    return (root%q+q)%q

def Problem365():
    primes=[]
    binomial_coefficients={}
    #initPrimes
    SIEVE_RANGE=5000
    visited=[False]*SIEVE_RANGE
    visited[0]=True
    visited[1]=True
    for i in range(2,SIEVE_RANGE):
        if not visited[i]:
            if i>1000:
                primes.append(i)
            for j in range(i+i,SIEVE_RANGE,i):
                visited[j]=True

    #initialize Binomial Coefficients
    m=10**18
    n=10**9
    for p in primes:
        def combination_number(m,n,prime):
            def base_convert(number,base):
                d=number
                rv=[]
                while d>0:
                    #divmod return (a//b,a%b)
                    d,r=divmod(d,base)
                    rv.append(r)
                return rv
            m_rep=base_convert(m,prime)
            n_rep=base_convert(n,prime)
            len_diff=len(m_rep)-len(n_rep)
            n_rep+=[0]*len_diff
            rv=1
            #zip will return tuples of the form (x,y),where x and y are elements in m_rep and n_rep
            for i,j in zip(m_rep,n_rep):
                if i<j:
                    rv=0
                    break
                curr_c=math.factorial(i)//math.factorial(j)//math.factorial(i-j)
                rv=(rv*curr_c)%prime
            return rv
            
        binomial_coefficients[p]=combination_number(m,n,p)

    n=len(primes)
    rv=0
    for i in range(n):
        p=primes[i]
        a=binomial_coefficients[p]
        for j in range(i+1,n):
            q=primes[j]
            b=binomial_coefficients[q]                
            x=solve_equations(a,p,b,q)
            for k in range(j+1,n):
                r=primes[k]
                c=binomial_coefficients[r]
                y=solve_equations(x,p*q,c,r)
                rv+=y
    print(rv)

Problem365()    
                
