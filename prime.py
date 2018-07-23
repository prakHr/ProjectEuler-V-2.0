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


def solve_216():
    count=0
    for i in range(2,50,000,000):
        if is_prime(2*i*i-1):
            count+=1
    return count
print(solve_216())

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


     

def compute_35():
    until=999999
    sieve=get_sieve(until)
    prime=get_primes_from_sieve(sieve)
    
    def is_circular_prime(n):
        x=str(n)
        return all( sieve[int( x[i:]+x[:i] ) ] for i in range(len(x)) ) 

    ans=sum(1 for i in range(len(sieve)) if is_circular_prime(i))
    return str(ans)
print(compute_35())
    


def solve_133():
    until=100000
    sieve=get_sieve(until)
    prime=get_primes_from_sieve(sieve)
    k=10**24
    result=0
    for i in range(0,len(prime)):
        if ( pow(10,k,9*prime[i])!=1):
            result+=prime[i]
    return result
print(solve_133())
    
def solve_132():
    until=200000
    prime=get_sieve(until)
    
    factorsum=0
    factornum=0
    i=4#factors are not 2,3... as product then can't be of form 111...
    while factornum<40:
        if prime[i] and pow(10,10**9,9*i)==1:
            factorsum+=i
            factornum+=1
        i+=1
    return factorsum
print(solve_132())


def isPerm(n,m):
    x=sorted(str(n))
    y=sorted(str(m))
    return x==y

def solve_totient_permutation_70():
    best=1
    phiBest=1
    bestRatio=float('inf')

    limit=10**7
    
    upper=5000
    sieve=get_sieve(upper)
    primes=get_primes_from_sieve(sieve)

    for i in range(0,len(primes)):
        for j in range(i+1,len(primes)):
            n=primes[i]*primes[j]
            if n>limit:
                break
            #first ratio will be min when we consider n as prime,moreover
            #product of distint primes
            #just search in the direction where we are most likely to find a solution
            #2 prime factors which are close to sqrt(limit)=3162 so, include upto 5000
            phi=(primes[i]-1)*(primes[j]-1)
            ratio=n/phi

            if(isPerm(n,phi) and bestRatio>ratio):
                best=n
                phiBest=phi
                bestRatio=ratio
    return str(best)
print(solve_totient_permutation_70())



sieve=get_sieve(10000)
primes=get_primes_from_sieve(sieve)

def get_twice_square(n):
    return 2*n**2

def does_comply_with_goldbach(number):
    n=1
    current_twice_square=get_twice_square(n)
    while current_twice_square < number:
        for prime in primes:
            if current_twice_square+prime>number:
                break
            if sieve[number]==0 and number==current_twice_square+prime:
                return True
        n+=1
        current_twice_square=get_twice_square(n)
    return False

def first_odd_composite_that_doesnt_comply():
    i=9
    while sieve[i]==1 or does_comply_with_goldbach(i):
        i+=2
    return i
print(first_odd_composite_that_doesnt_comply())
            
        
    
#print(get_sieve(10))

def is_left_truncable_prime(num,sieve):
    while sieve[num] and len(str(num))>1:
        num=int(''.join(list(str(num))[1:]))
        ##print(num)
    return (sieve[num]==1 and len(str(num))==1)

def is_right_truncable_prime(num,sieve):
    while sieve[num] and len(str(num))>1:
        num=int(''.join(list(str(num))[:-1]))
    return(sieve[num] and len(str(num))==1)

def truncable_prime(num,prime):
    return is_left_truncable_prime(num,sieve) and is_right_truncable_prime(num,sieve)
        
sieve = get_sieve(1000000)
total=0
#print(is_left_truncable_prime(3797,sieve))
for i in range(13,1000000):
    if truncable_prime(i,sieve):
        total+=i
    
print(total)



l=[]
x=pow(2,1000)
while(x!=0):
    a=x%10
    l.append(a)
    x=x//10

print(sum(l))







def collatz_sequence(n):
    x=[]
    while(n>1):
        x.append(n)
        if(n%2==0):
            n=n/2    
        else:
            n=3*n+1
    x.append(1)  
    return(x)
    

def lcs(until):
    longest_sequence_size=1
    number=2
    for i in range(2,until+1):
        seq=collatz_sequence(i)
        seq_len=len(seq)
        if(longest_sequence_size < seq_len):
            longest_sequence_size=seq_len
            number=i
    return number


print(lcs(1000000))



def primes_sum(until):
    sieve=[1]*until
    total=2
    curr=3
    while(curr<until):
        
        #hence the answer is the index of sieve from 3 onwards for which sieve is 1
        if(sieve[curr]):
            total=total+curr
            temp=curr
            #removes the factors of curr from sieve 
            while(temp<until):
                sieve[temp]=0
                temp=temp+curr
        #taking advantage of the fact that 2 is only even prime
        # and rest of the nos follow even odd sequence
            
        curr=curr+2
    return total
                
print(primes_sum(2000000))


def smallest_mutiple(n):
    step=1
    test=step

    for i in range(2,n+1):
        while(True):
            is_multiple=True
            
            for j in range(1,i+1):
                if(test%j!=0):
                    is_multiple=False
                    break

            if(is_multiple):
                step=test
                break
            test=test+step
    return(test)
print(smallest_mutiple(20))

def lpf(n):
  test=2
  while(n>1):
      if(n%test==0):
          n=n/test
      else:
          test=test+1
  print(test)

lpf(600851475143)    


