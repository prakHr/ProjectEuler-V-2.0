def sieve():
    visited=[False]*45
    primes=[]
    cubic_roots={}
    for i in range(2,45):
        if visited[i] is False:
            primes.append(i)
            
            def set_cubic_roots(prime):
                for i in range(1,prime):
                    if (i*i*i)%prime is 1:
                        if prime not in cubic_roots:
                            cubic_roots.update({prime:[i]})
                        else:
                            cubic_roots.update({prime:cubic_roots[prime]+[i]})
                            
            set_cubic_roots(i)
            for j in range(i+i,45,i):
                visited[j]=True
    return cubic_roots,primes

def extended_gcd(a,b):
    (x,y)=(0,1)
    (last_x,last_y)=(1,0)
    while b!=0:
        (q,r)=divmod(a,b)
        (a,b)=(b,r)
        (x,last_x)=(last_x-q*x,x)
        (y,last_y)=(last_y-q*y,y)
    return (last_x,last_y)

def solve_equations(a,m,b,n):
    q=m*n
    (x,y)=extended_gcd(m,n)
    root=a+(b-a)*x*m
    return (root%q+q)%q
   
def compute_271():
    cubic_roots,primes=sieve()
    print(cubic_roots)
    sol=cubic_roots[2]
    #print(sol)
    m=2
    for prime in primes:
        if prime is 2:
            continue
        temp=[]
        for i in sol:
            for j in cubic_roots[prime]:
                temp.append(solve_equations(i,m,j,prime))
        sol=temp
        m=m*prime
    print(sum(sol)-1)
        
compute_271()
