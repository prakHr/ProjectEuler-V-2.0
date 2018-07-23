import itertools
def Union_of_get_sieve_AND_get_primes_from_sieve():
    #We now have the ability to modify PRIME_COUNT and SIEVE_RANGE
    #Value of Primes are stored in an array
    curr_pos=0
    SIEVE_RANGE=1000200
    PRIME_COUNT=78514
    sieve_visited=[0]*SIEVE_RANGE
    primes=[0]*PRIME_COUNT
    prime_bound=[0]*PRIME_COUNT
    bound=10
    for i in range(2,SIEVE_RANGE):
        if not sieve_visited[i]:
            primes[curr_pos]=i
            #Here bound corresponding to prime_bound[curr_pos] is the smallest value that exceeds prime,i
            if i>bound:
                bound=bound*10
            prime_bound[curr_pos]=bound
            curr_pos+=1
            for j in range(i+i,SIEVE_RANGE,i):
                sieve_visited[j]=True
    return primes,prime_bound


def compute_134():
    primes,prime_bound=Union_of_get_sieve_AND_get_primes_from_sieve()
    sum=0
    for i in itertools.count(2):
        if primes[i]>1000000:
            break
        def Getconnection(p1,p2):
            #For example p1=19,p2=23,bound=100=>for n=1219 to be divisible by p2
            #n=m*(bound)+p1=0(mod p2)
            #this gives m=(-p1*(bound)^-1 (mod p2))
            #then m will satisfy the divisibility test as it comes inside the ring p2,st 0<=m<p2
            bound=prime_bound[p1]
            p1=primes[p1]
            p2=primes[p2]
            def ExtendedGCD(a,b):
                x,y,last_x,last_y=0,1,1,0
                while b:
                    quotient=a//b
                    t=a
                    a=b
                    b=t%b

                    t=x
                    x=last_x-quotient*x
                    last_x=t

                    t=y
                    y=last_y-quotient*y
                    last_y=t
                return last_x
            last_x=ExtendedGCD(bound,p2)
            rv=(-last_x*p1)%p2
            if rv<=0:
                rv+=p2
            return rv*bound+p1
        sum+=Getconnection(i,i+1)
    return sum
print(compute_134())


def get_norm(n,p):
    t=p
    result=0
    while t<=n:
        #no of factors of p divisible by n=(n//p)+(n//p^2)+..0 
        result+=n//t
        t=t*p
    return result

def get_free_factorial_naive(n,p,s):
    result=1
    mod=p**s
    for i in range(1,n+1):
        if i%p!=0:#all factors of 5 are removed and n! is computed AND n=n//1%5^5,n//5%5^5,n//25%5^5,...
            result=(result*i)%mod
            
    return result

def get_last_factorial(n,p,s):
    result=1
    t=1
    i=0
    while t<=n:
        #Sign (-1)^(n//(5^(5+i))) i=0,1,2,...
        result=result*( (-1)**(n//p**(s+i))*get_free_factorial_naive((n//t)%(p**s),p,s))
        #Due to negative sign result<0,so result%5^5>0
        result=result%(p**s)
        t=t*p
        i+=1
    return result

def extended_gcd(a,b):
    #ax+by=gcd(a,b)
    #here sequences of q,r,s,t are in play
    #r(i-2)=r(i)+r(i-1)*q(i),where r0=a,r1=b,also st r(i)=s(i)*a+t(i)*b
    #Finite Steps as seq of r(i) is decreasing, dut to euclid's division
    #r(i)=r(i-2)-r(i-1)*q(i)=(s(i-2)*a+t(i-2)*b)-(s(i-1)*a+t(i-1)*b)*q(i)
    #r(i)=(s(i-2)-s(i-1)*q(i))*a+(t(i-2)-t(i-1)*q(i))*b
    #Corresponding sequence for s(i) and t(i)
    #a=r0=s0*a+t0*b => s0=1,t0=0
    #b=r1=s1*a+t1*b => s1=0,t1=1
    #finally we stop at the iteration when r(i-1)=0
    #let r(n-1)=0
    #gcd(a,b)=gcd(r0,r1)=gcd(r1,r2)=...=gcd(r(n-2),r(n-1))=r(n-2)
    # => gcd(a,b)=r(n-2)=s(n-2)*a+t(n-2)*b
    
    (x,y)=(0,1)
    (last_x,last_y)=(1,0)
    while b!=0:
        (q,r)=divmod(a,b)
        (a,b)=(b,r)#simple gcd(a,b)=gcd(b,a%b)
        (x,last_x)=(last_x-q*x,x)
        (y,last_y)=(last_y-q*y,y)
    return (last_x,last_y)

def solve(a,m,b,n):
    #solve Chinese Remainder Theorem
    #case of 2 moduli
    #x=a1(mod n1)
    #x=a2(mod n2)
    #if m1,n1 are relatively prime, then m1*n1+m2*n2=1
    #n2 corresponding to m1 comes from egcd
    #a solution of form a2*m1*n1+a1*m2*n2 => x
    #x=a1*(1-m1*n1)+a2*(m1*n1)
    #x=a1+(a2-a1)*m1*n1
    #x=a(mod m) and y=b(mod n),where m and n are coprime.
    q=m*n
    (x,y)=extended_gcd(m,n)
    root=a+(b-a)*x*m
    #factorial will go very large so return root%(2^s*5^s)
    return (root%q+q)%q#+q to ensure positive value of mod


def _extended_gcd(a,b):
    #a*x+b*y=gcd(a,b)
    #=>(b%a)*x1+a*y1=gcd(b%a,a)
    #As b%a=b-b//a*a
    #(b-b//a*a)*x1+a*y1=gcd(b%a,a)
    #=>(b)*(x1)+(a)*(y1-(b//a)*x1)=gcd(b%a,a)
    if a==0:
        return (b,0,1)
    else:
        g,y,x=_extended_gcd(b%a,a)
        return (g,x-(b//a)*y,y)

def mod_inverse(a,m):
    #ax=1(mod m)
    #ax+my=egcd(a,m)=1
    g,x,y=_extended_gcd(a,m)
    #inverse exists only for g==1
    if g!=1:
        raise Exception('multiplicative inverse does not exist')
    return x%m


def get_last_nonzero_digits(n):
    s=5
    norm=get_norm(n,5)
    a=get_last_factorial(n,5,s)
    x=solve(a,5**s,0,2**s)#solving noOfFactors(mod 5^5) and 0(mod 2^5)
    y=pow(mod_inverse(2,5**s),norm,5**s)#((inverse(2,5^5))^noOfFactors)%5^5
    return (x*y)%10**s
    
    
print(get_last_nonzero_digits(10**12))

