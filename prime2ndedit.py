import itertools
def Sieve_127():#18407904
    SIEVE_RANGE=120000
    sieve_visited=[0]*SIEVE_RANGE
    rad=[1]*SIEVE_RANGE
    sieve_visited[0]=True
    sieve_visited[1]=True
    for i in range(2,SIEVE_RANGE):
        if not sieve_visited[i]:
            rad[i]=rad[i]*i
            for j in range(i+i,SIEVE_RANGE,i):
                sieve_visited[j]=True
                rad[j]=rad[j]*i
    return rad
def IsCoPrime(a,b):
    while b:
        t=b
        b=a%t
        a=t
    return a==1
def compute_127():
    rad=Sieve_127()
    sum=0
    for c in range(3,120000):
        if 2*rad[c]>=c:
            continue
        for a in itertools.count(1):
            if a+a>=c:
                break
            b=c-a
            if rad[a]*rad[b]*rad[c]<c and IsCoPrime(rad[a],rad[b]):
                sum+=c
    print(sum)
compute_127()
    
    
def compute_214():
    visited=[0]*40000000
    prime=[0]*40000000
    phi=[0]*40000000
    def InitPrimeAndPhiTable():
        for i in range(0, 40000000):
            phi[i]=i
        curr_pos=0
        for i in range(2, 40000000):
            if not visited[i]:
                #if i is prime phi[prime]=prime-1
                phi[i]=i-1
                prime[curr_pos]=i
                curr_pos+=1
                for j in range(i+i, 40000000,i):
                    #as j is not prime it is factor of prime i
                    #so apply,phi[composite_number]=composite_number*{where i comes as a product of all prime factors of composite_number}(1-1//i)
                    phi[j]=(phi[j]/i)*(i-1)
                    visited[j]=True
    InitPrimeAndPhiTable()
    
    length=0
    sum=0
    for i in range(0,40000000):
        def GetChainLength(n,limit):
            length=1
            d=n
            while d!=1 and length<=limit:
                d=phi[int(d)]
                length+=1
            if length<=limit:
                return length
            else:
                return -1
        length=GetChainLength(prime[i],25)
        if length==25:
            sum+=prime[i]
    
    return sum
print(compute_214())
    
    


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

def is_prime(n):
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

def solve_131():
    cnt=0
    for i in range(1,577):#to check for the range p<=10**6,p after solving comes
        #out to be (i+1)^3 - ^3,put i=577
        if(is_prime( (i+1)**3-i**3  )):
            cnt+=1
    return cnt
print(solve_131())


def count_distinct_prime_factors(n):
    count=0
    while n>1:
        count+=1
        for i in range(2,int(n**0.5)):
            if n%i==0:
                while True:
                    n=n//i
                    if n%i!=0:
                        break
                break
        else:
            break
    return count
flag=0
for i in range(647,10**9):
    if (count_distinct_prime_factors(i)>=4 and count_distinct_prime_factors(i+3)>=4 and count_distinct_prime_factors(i+1)>=4 and count_distinct_prime_factors(i+2)>=4):
        x=i
        flag=1
    if(flag):
        break
print(i)


def compute_122():
    until=5*(10**6)
    sieve=get_sieve(until)
    primes=get_primes_from_sieve(sieve)
    r=0#((a-1)^n)+((a+1)^n)%(a^2)=2*a*n when n is odd
    n=7037
    while(r<10000000000):
        n+=2
        p=primes[n-1]
        r=2*p*n
    return n
print(compute_122())


def compute_204():
    LIMIT=10**9
    sieve=get_sieve(100)
    primes=get_primes_from_sieve(sieve)
    #print(primes)

    def count(primeindex,product):
        if primeindex==len(primes):
            #type n consists of all primes<n
            return 1 if product<=LIMIT else 0
        else:
            result=0
            while product<=LIMIT:
                result+=count(primeindex+1,product)
                #hamming number is formed from product of primes
                #So,what we have to see is this product goes upto type n of hamming number
                product*=primes[primeindex]
            
            return result
    return str(count(0,1))
print(compute_204())





print((28433*2**7830457+1)%10**10)#Mercennes prime



def is_prime(n):
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

def primes_factors_of_num(n):
    x=[]
    x.append(1)
    for i in range(2,n):
        if(n%i==0 and is_prime(i)):
            x.append(i)
    if is_prime(n):
        x.append(n)
    return x



            
            

        


def compute_50():
    ans=0
    isprime=get_sieve(999999)
    primes=get_primes_from_sieve(isprime)
    consecutive=0
    for i in range(len(primes)):
        sum=primes[i]
        consec=1
        for j in range(i+1,len(primes)):
            sum+=primes[j]
            consec+=1
            if sum>=len(isprime):
                break
            if isprime[sum] and consec>consecutive:
                ans=sum
                consecutive=consec
    return str(ans)

print(compute_50())

def compute_87():
    LIMIT= 50000000
    sieve=get_sieve(int(50000000**0.5))
    primes=get_primes_from_sieve(sieve)

    sums={0}
    for i in range(2,5):
        new_sums=set()
        for p in primes:
            q=p**i
            if q>LIMIT:
                break
            for x in sums:
                if x+q <= LIMIT:
                    new_sums.add(x+q)
        sums=new_sums
    return(str(len(sums)))
print(compute_87())
        
