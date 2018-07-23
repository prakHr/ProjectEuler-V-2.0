import itertools

    

def InitPrimeTable():
    SIEVE_RANGE=100000001
    #PRIME_COUNT=664579 for 347
    PRIME_COUNT=5761455
    visited=[0]*SIEVE_RANGE
    prime=[0]*PRIME_COUNT
    visited[0]=True
    visited[1]=True
    curr=0
    for j in range(2,SIEVE_RANGE):
        if not visited[j]:
            prime[curr]=j
            curr+=1
            for i in range(j+j,SIEVE_RANGE,j):
                visited[i]=True
    return visited,prime

def solve357():
    sieve,prime=InitPrimeTable()
    
    sum=0
    PRIME_COUNT=5761455
    #(prime[PRIME_COUNT-1])=99999989<100000001
    for i in range(0,PRIME_COUNT):

        def isPrimeGeneratingInteger(n):
            d=2
            while d*d<=n:
                if n%d==0 and sieve[d+n//d]:
                    return False
                d+=1
            return True
        #We Want f(1)=1+n/1=1+n to be prime
        #n=prime[i]-1
        if isPrimeGeneratingInteger(prime[i]-1):
            sum+=prime[i]-1
    return sum
print(solve357())
    

def M(p,q,N):
    MAX=0
    for i in itertools.count(1):
        p_pow=pow(p,i)
        if p_pow>N:
            break
        for j in itertools.count(1):
            q_pow=pow(q,i)
            curr=p_pow*q_pow
            if curr>N:
                break
            if curr>MAX:
                MAX=curr
    return MAX

def S(N):#347
    PRIME_COUNT=664579
    x,prime=InitPrimeTable()
    sum=0
    for i in range(0,PRIME_COUNT):
        if prime[i]*prime[i]>N:
            break
        for j in range(i+1,PRIME_COUNT):
            if prime[i]*prime[j]>N:
                break
            sum+=M(prime[i],prime[j],N)
    return sum

def compute_347():
    x=S(10000000)
    print(x)

compute_347()

def compute_187():
    MAX=100000000#10^8
    prime=[0]*MAX
    for i in range(2,MAX):
        if prime[i]==0:
            prime[i]=1
            for j in range(i+i,MAX,i):
                prime[j]+=1
            if i<10000:#10^(8/2)
                for j in range(i*i,MAX,i*i):
                    prime[j]+=1
                    
            if i<465:#10^(8/3)
                for j in range(i*i*i,MAX,i*i*i):
                    prime[j]+=1
    count=0
    for i in range(4,MAX):
        if prime[i]==2:
            count+=1
    return(count)
print(compute_187())


def compute_231():
    MAX=20000001
    visited=[0]*(MAX)
    query=[0]*(MAX)
    d=[0]*(MAX)
    d=[i for i in range(0,MAX)]
    for i in range(2,MAX):
        if not visited[i]:
            #initially query[i] becomes smallest prime i
            query[i]=i
            for j in range(i+i,MAX,i):
                prime_power=0
                while d[j]%i==0:
                    d[j]=d[j]//i
                    prime_power+=1
                    #then sum of primefactors for a particular query is calculated
                    #repeatedly for different primes i by incrementing prime_power
                    #prime_power is computed using d[j]=j where j is iterated over
                    #the multiples of prime until all prime factors of a particular i
                    #vanishes in d[j]
                query[j]=query[j]+i*prime_power
                visited[j]=True
    rv=0
    for i in range(20000000-5000000+1,20000001):
        rv+=query[i]
    for i in range(1,5000001):
        rv-=query[i]
    return rv
print(compute_231())
