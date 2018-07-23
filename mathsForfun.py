

sum=0
#n is of form 10*x+y for 10<n<Google no
#such that one right rotated number is of form 10^(length)*y+x
#now we want some digit d in range(1,10) such that number and right rotated number has same length
#st (10*x+y)*d=10^(length)*y+x
#solving x=(10^length-d)*y/(10*d-1)
#also length !=0 for non zero leading digit
#here length=k

for k in range(1,100):
    #k=1,2,3...99
    for d in range(1,10):
        #d=1,2,3...9
        for y in range(1,10):
           
            x,r=divmod((10**k-d)*y,10*d-1)
            if r!=0 or x<10**(k-1):
                continue
            sum+=10*x+y
print(sum%10**5)

from itertools import count


def compute_square_free_193():
    M=2**50-1
    N=2**25
    div=[0]*N
    for i in count(2):
        if(i*i>=M):
            break
        if div[i]!=0:
            continue
        if div[i]==-1:
            continue
        if i*i<N:
            for j in range(i*i,N,i*i):
                div[j]=-1

        for j in range(i,N,i):
            if div[j]==-1:
                continue
            div[j]+=1
    total=0
    for i in range(2,N):
        if div[i]==-1:
            continue
        test=M//(i*i)
        if div[i]%2==1:
            total+=test
        else:
            total-=test
    return str(M-total)
print(compute_square_free_193())



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


    
def GetDigitalRoot(n):
    sum=0
    while(n):
        sum+=n%10
        n=n//10
    if sum>=10:
        sum=GetDigitalRoot(sum)
    return sum

mdrs=[0]*1000000
for i in range(2,1000000):
    mdrs[i]=GetDigitalRoot(i)
    d=2
    while(d*d<=i):
        if i%d==0:
            mdrs[i]=max(mdrs[i],mdrs[d]+mdrs[i//d])

        d+=1
print(sum(mdrs[i] for i in range(2,1000000)))




#It is possible to show that if p is prime, choose(m, n) is not divisible by p if and only if the addition n + (m-n) when written in base p has no carries.  This means that the number of entries in the mth row of Pascal's triangle that are not divisible by p is equal to the product over all digits d of m written in base p of 1+d.
#For example...
#10 base 2 is 1001, which means that the number of odd entries in the 10th row of PT is 2*1*1*2=4.
#10 base 5 is 20 --> number of entries in 10th row not divisible by 5 is 3*1=3.
#This can be extended as follows.  The number of entries in rows 0 through p^n-1 that are not divisible by p is ((p(p+1))/2)^n. 
#This leads to a recursive formulation...
#Let n have k digits when written in base p, and let d be the most significant digit.  Let f(n) denote the number of entries in rows 0-n of PT not divisible by p.  Then...
#f(n) = (d(d+1)/2)(p(p+1)/2)^(k-1) + (d+1)f(n/p)

def sumupto(digits,p):
    if not digits:
        return 1
    n=digits[0]
    l=len(digits)-1
    return (n*(n+1)//2)*(p*(p+1)//2)**l+(n+1)*sumupto(digits[1:],p)
N=10**9-1
p=7
digits=[]
while N>0:
    digits.append(N%p)
    N=N//p
digits.reverse()

print(sumupto(digits,7))


def get_sieve(until):
    sieve=[1]*until
    sieve[1]=0
    sieve[0]=0
    prime=2
    while prime<until:
        if sieve[prime]:
            temp=prime+prime
            while temp<until:
                sieve[temp]=0
                temp=temp+prime
        prime+=1
    return sieve


def get_primes_from_sieve(sieve):
    primes=[]
    for i in range(0,len(sieve)):
        if sieve[i]==1:
            primes.append(i)
    return primes

def solve_146():
        INCREMENTS=[1,3,7,9,13,27]
        LIMIT=150000000
        #print(INCREMENTS[-1])
        #print(set(range(INCREMENTS[-1])))
        #print(set(INCREMENTS))
        #print( set(range(INCREMENTS[-1]))-set(INCREMENTS)   )
        NON_INCREMENTS=set(range(INCREMENTS[-1]))-set(INCREMENTS)
        max=LIMIT**2+INCREMENTS[-1]#limit^2+27
        until=int(max**0.5)
        sieve=get_sieve(until)
        primes=get_primes_from_sieve(sieve)

        def has_consecutive_primes(n):
                n2=n**2
                temp=[(n2+k) for k in INCREMENTS]
                if any((x!=p and x%p==0)for p in primes for x in temp):
                        return False
                #to check the prime numbers we obtained is consecutive or if
                #there is any composite of form n^2+k 0<k<27 not 1,3,7,9,13,27
                return all((not isPrime(n2+k) ) for k in NON_INCREMENTS)
        def isPrime(n):
                end=int(n**0.5)
                for p in primes:
                        if p>end:
                                break
                        if n%p==0:
                                return False
                return True
               
        #10 is written in range as n=0(mod5) and n=0(mod2)
        ans=sum(n for n in range(0,LIMIT,10) if has_consecutive_primes(n))
        return str(ans)

print(solve_146())


def solve_calfin_wilf_tree():
        path=[]
        den=17
        num=13
        m,n=den,num
        while m!=1 or n!=1:
                if m>n:
                        m,n=m-n,n
                        path.append(1)
                elif m<n:
                        m,n=m,n-m
                        path.append(0)
        path.append(1)
        
        return path
        
print(solve_calfin_wilf_tree())
def floyd_cycle_detection_for_fuction_that_maps_any_finite_set_to_itself():
        def f(x):
                return int(2**(30.403243784-x*x))*(10**-9)
        ITERATIONS=10**12
        hare=-1
        tortoise=-1
        i=0
        while i<ITERATIONS:
                if i>0 and tortoise==hare:
                        break
                tortoise=f(tortoise)
                hare=f(f(hare))
                i+=1
        remain=(ITERATIONS-i)%i
        for i in range (remain):
                tortoise=f(tortoise)
        ans=tortoise+f(tortoise)
        ans=int(ans*10**9)/10**9
        return ans
print(floyd_cycle_detection_for_fuction_that_maps_any_finite_set_to_itself())
                        


from math import *
import fractions

def A(n):
        if(fractions.gcd(n,10)!=1):
                return 0
        x=1
        k=1
        while x!=0:
                x=(x*10+1)%n
                k+=1
        return k
def isPrime(n):
    cnt=2
    if n<=1 :
        return 0
    if n==2 or n==3:
        return 1
    if n%2==0 or n%3==0:
        return 0
    if n<9:
        return 1
    counter=5
    while(counter*counter<=n):
        if n%counter==0:
            return 0
        if n%(counter+2)==0:
            return 0
        counter+=6
    
    return 1

def solve_130():
        found=0
        limit=25
        n=1
        sum=0
        while(found<limit):
                n+=1
                if(isPrime(n)):
                        continue
                a=A(n)
                if(a!=0 and ((n-1)%a==0)):
                        sum+=n
                        found+=1
        return sum
print(solve_130())
def solve_129():
        L=10**6+1
        n=L # via pigeohole principle A(n) is congruent to 0 mod n while each number
        #is in the range(0 to n-1) .So one more number with index exists which satisfy
        #A(n) until n<=L
        while A(n)<L:
                n+=2#n has to odd
                #A(n)<n so we start search at L
        return n

print(solve_129())
                


def is_square(n) :
    return int(sqrt(n)) == sqrt(n)
# r < d < q
# a > b , a/b = c
# because d^2 = q*r
# cb^2 < cab < ca^2
# n = cabca^2 + cb^2 = a^3*c*2*b + b^2*c = a*a*a*c*c*b + b*b*c
L = 10**12
progressiveSquares = set()
sum_progressive_squares = 0
for a in range(2,int(L**(1/3.)+1)) :
    for b in range(1,a) :
        if a*a*a*b*b + b*b >= L : break
        c = 1
        n = a*a*a*c*c*b + b*b*c
        while n < L :
            n = a*a*a*c*c*b + b*b*c
            if is_square(n) and n not in progressiveSquares :
                progressiveSquares.add(n)
                sum_progressive_squares += n
                #print n,b*b*c,a*b*c,a*a*c
            c +=1
      
print (sorted(progressiveSquares))
print ("answer",sum(progressiveSquares))
print ("sum_progressive_squares", sum_progressive_squares)
                                        
                                        

def get_Maximized_Product(m):
        n=2**(m*((m+1)/2))
        d=(m+1)**(m*((m+1)/2))
        x=1
        for i in range(1,m+1):
                for j in range(1,i+1):
                        x=x*i
                #print(x)
                        
      
        return int((x*n)/d)
def solve_190():
        s=0
        for i in range(2,16):
                
                s+=get_Maximized_Product(i)
                
        return s
#print(get_Maximized_Product(10))
print(solve_190())
def solve_199():
        k=1+2/(3**0.5)#1+(2/sqrt(3))
        uncovered_area=1-3*(1/k)**2
        stack=[(-1,k,k),(k,-1,k),(k,k,-1),(k,k,k)]
        for i in range(10):
                next_stack=[]
                for k1,k2,k3 in stack:
                        k4=k1+k2+k3+2*((k1*k2+k2*k3+k1*k3)**0.5)
                        next_stack+=[(k1,k2,k4),(k2,k3,k4),(k3,k1,k4)]
                        uncovered_area=uncovered_area-(1/k4)**2
                stack=next_stack
        return(uncovered_area)
        
print(solve_199())      

                        
def compute():
	divisors = [2] * (10**7 + 1)  # Invalid for indexes 0 and 1
	for i in range(2, (len(divisors) + 1) // 2):
		for j in range(i * 2, len(divisors), i):# example take 3 then all multiples of 3 less than 10**7 will contain 3 like 6,9,12,18..so their count is incrementated by 1
			divisors[j] += 1
	
	ans = sum((1 if divisors[i] == divisors[i + 1] else 0) for i in range(2, len(divisors) - 1))
	return str(ans)


if __name__ == "__main__":
	print(compute())
    
def isPandigital(n):
    a=list(sorted(str(n)))
    if(a==['1', '2', '3', '4', '5', '6', '7', '8', '9']):
        return True
    return False

def digits(n):
        cnt=0
        while(n!=0):
                cnt+=1
                n=n//10
        return cnt
                
fn1=1
fn2=1
tailcut=10**9
n=2
found=False
while found!=True:
    n+=1
    fn=fn1+fn2
    tail=fn%tailcut
    if(isPandigital(tail)):
        x=digits(fn)
        if(x>9):
            head=fn//10**(x-9)
            if(isPandigital(head)):
                found=True
    fn2=fn1
    fn1=fn
print(n)
import fractions 

limit=10**6
result=0
for m in range(2,10**6):
    for n in range(1,m):
        if (n+m)%2==1 and fractions.gcd(n,m)==1:
            a=2*m*n
            b=m*m-n*n
            c=m*m+n*n
            p=a+b+c
            if c%(b-a)==0:
                result+=limit/p



def compute_155():
    SIZE=18
    all=set()
    possible=[]
    possible.append(set())#
    possible.append({(30,1)})#
    all.update(possible[1])#doubt
    #n0/d0+n1/d1=(n0*d1+n1*d0)/(d0*d1)
#1/(d0/n0+d1/n1)=1/(d0*n1+n0*d1/(n0*n1))=(n0*n1)/(d0*n1+n0*d1)
    for i in range(2,SIZE+1):
        poss=set()
        for j in range(1,i//2+1):
            for (n0,d0) in possible[j]:
                for (n1,d1) in possible[i-j]:
                    pseudosum=d0*n1+n0*d1
                    num_product=n0*n1
                    den_product=d0*d1
                    np=fractions.gcd(pseudosum,num_product)
                    dp=fractions.gcd(pseudosum,den_product)
                    poss.add((pseudosum//dp,den_product//dp))
                    poss.add((num_product//np,pseudosum//np))
        possible.append(poss)
        all.update(poss)
    return str(len(all))
print(compute_155())

n=2
d=1
result=0
for i in range(2,101):
    temp=d
    if i%3==0:
        c=2*(i//3)
    else:
        c=1
    d=n
    n=c*d+temp
l=[]
while n!=0:
    l.append(n%10)
    n=n//10

print(sum(l))




def is_bouncy(number):
    n=str(number)
    sorted_n=sorted(n)
    return "".join(sorted_n)!=n and "".join(reversed(sorted_n))!=n

def least_number_with_bouncy_percentage(percentage):
    current=1
    bouncy_numbers=0.0
    while True:
        if is_bouncy(current):
            bouncy_numbers+=1
        if(bouncy_numbers/current)*100==percentage:
            return current
        current+=1
print (least_number_with_bouncy_percentage(99))
            





def is_palindrome(n):
    str_n=str(n)
    return str_n==str_n[::-1]

def is_number_plus_reverse_palindromic(number):
    str_number=str(number)
    str_reverse=str_number[::-1]
    return is_palindrome(number+int(str_reverse))


def find_lychrel_numbers(until,max_iterations):
    total_lychrel_numbers=until
    for number in range(1,until+1):
        tests=[]
        curr_number=number
        curr_iteration=0
        while curr_iteration<max_iterations:
            if is_number_plus_reverse_palindromic(curr_number):
                total_lychrel_numbers-=1
                break
            else:
                curr_number+=int(str(curr_number)[::-1])
            curr_iteration+=1
    return total_lychrel_numbers

print(find_lychrel_numbers(10000,50))

def reversible(num):
    n1=str(num)
    n2=n1[::-1]
    sumx=int(n1)+int(n2)
   # print(sumx)
    y=sumx
    #print(y)
    if y%2==0:
        return 0
    while y>0:
        x=y%10
        if x%2==0:
            return 0
        y=(y//10)
    return 1
    


cnt=0
for i in range(1,1000000000):
    if (i%10!=0 and i%2!=0 and reversible(i)):
        cnt+=1
print(2*cnt)


def sumx(n):
    l=[]
    c=[]
    for i in range(1,n+1):
        l.append(i**2)
        c.append(i)
    x=(sum(c)*sum(c))
    return(x-sum(l))
print (sumx(100))

def prime(n):
    primes=[2]
    a=3
    while(len(primes)<n):
        if(all(a%prime!=0 for prime in primes)):
            primes.append(a)
        a=a+2
    return primes[-1]

print(prime(10))
    

def mean(numbers):
    s=sum(numbers)
    n=len(numbers)
    mean=s/n
    return mean

def find_diff(numbers):
    x=mean(numbers)
    d=[]
    for n in numbers:
        a=n-x
        d.append(a)
    return d

def variance(numbers):
    diff=find_diff(numbers)
    squared=[]
    for d in diff:
        squared.append(d**2)
    s=sum(squared)
    variance=s/len(numbers)
    return variance

def correlation(x,y):
    d=[]
    d1=[]
    d2=[]
    X1=len(x)
    X2=len(y)
    X=sum(x)
    Y=sum(y)
    for n1,n2 in zip(x,y):
        d.append(n1*n2)

    for n1 in x:
        d1.append(n1**2)
    for n2 in y:
        d2.append(n2**2)
    s=sum(d)
    s1=sum(d1)
    s2=sum(d2)
    a=((X1*s)-(X*Y))
    b=( (   (X1*s1)-(X**2) )*( (X2*s2)  -  (Y**2) ) )**0.5
    ans=a/b
    return ans

x=[90,92,95,96,87,87,90,95,98,96]
y=[85,87,86,97,96,88,89,98,98,87]
print('{0:.2f}'.format(correlation(x,y)))
    
    
        

def median(numbers):
    a=len(numbers)
    x=sorted(numbers)
    if(a%2==0):
        n1=a/2
        n2=a/2+1
        return (x[int(n1-1)]+x[int(n2-1)])/2
    else:
        n1=a/2
        return(x[int(n1)])
scores=[1,2,3,4444,555,666,4444]
print(median(scores))

def rangexx(numbers):
    lowest=min(numbers)
    highest=max(numbers)
    r=highest-lowest
    return r,highest,lowest
scores = [7, 8, 9, 2, 10, 9, 9, 9, 9, 4, 5, 6, 1, 5, 6, 7, 8, 6, 1, 10]
r,highest,lowest=rangexx(scores)


from collections import Counter
def frequency_table(numbers):
    table=Counter(numbers)
    freq=table.most_common()
    freq.sort()
    for n in freq:
        print('{0}\t{1}'.format(n[0],n[1]))
scores = [7, 8, 9, 2, 10, 9, 9, 9, 9, 4, 5, 6, 1, 5, 6, 7, 8, 6, 1, 10]
frequency_table(scores)
        
def multiplemode(numbers):
    c=Counter(numbers)
    freq=c.most_common()
    max_count=freq[0][1]
    modes=[]
    for n in freq:
        if n[1]==max_count:
            modes.append(n[0])
    return modes   
        
def mode(numbers):
    c=Counter(numbers)
    mode=c.most_common(1)
    return mode[0][0]
scores=[1,2,3,4444,555,666,4444,55]
print(mode(scores))
    


def fib(n):
    if (n==1):
        return([1])
    elif(n==2):
        return([1,1])
    else:
        a=1
        b=1
        x=[a,b]
        for i in range(n-2):
            c=a+b
            x.append(c)
            a=b
            b=c
        return(x)

y=fib(int(input())
     )
print(y)

n=int(input())
upper=int(input())
for i in range(1,upper+1):
    print('{0}*{1}={2}'.format(n,i,n*i)) 

n=int(input())
if n%2==0:
    print('even')
else:
    print('odd')
for i in range (0,20,2):
    print(n+i)


def roots(a,b,c):
    D=(b**2-4*a*c)**0.5
    x1=(-b+D)/(2*a)
    x2=(-b-D)/(2*a)
    print('{0:.2f},{1:.3f}'.format(x1,x2))



a=float(input())
b=float(input())
c=float(input())
roots(a,b,c)
