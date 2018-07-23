import itertools

def Union_of_get_sieve_AND_get_primes_from_sieve():
    #We now have the ability to modify PRIME_COUNT and SIEVE_RANGE
    #Value of Primes are stored in an array
    curr_pos=0
    SIEVE_RANGE=1000035
    PRIME_COUNT=78500
    sieve_visited=[0]*SIEVE_RANGE
    primes=[0]*PRIME_COUNT
    for i in range(2,SIEVE_RANGE):
        if not sieve_visited[i]:
            primes[curr_pos]=i
            curr_pos+=1
            for j in range(i+i,SIEVE_RANGE,i):
                sieve_visited[j]=True
    return primes

def compute_lps_and_ups_234():
    primes=Union_of_get_sieve_AND_get_primes_from_sieve()
    total_sum=0
    upper_bound=999966663333
    for i in itertools.count():
        #primes[i]<=sqrt(n)<=primes[i+1]
        #primes[i]*primes[i]+1<n<primes[i+1]*primes[i+1]-1
        #start<n<end
        if primes[i]*primes[i]>upper_bound:
            break
        start=primes[i]*primes[i]+1
        end=min(primes[i+1]*primes[i+1]-1,upper_bound)
#dividend=quotient*divisor+remainder, remainder=dividend%divisor,quotient=dividend//divisor
#dividend-remainder=quotient*divisor
        def SumMultiple(divisor,start,end):
            #(divisor=4,start=17,end=15)
            #SUM OF AP OF FORM a,a+d,a+2d,...a+(n-1)d=>Sn=n*[2a+(n-1)d]/2
            #Also a+=divisor-(a%divisor)...until a%divisor=a%primes[i]==0 such that a is not divisible by prime[i]
            #d=divisor=primes[i],n={primes[i+1]*primes[i+1]-1//divisor}  - {primes[i]*primes[i]//divisor}
            a=start
            if a%divisor:
                a+=(divisor-(a%divisor))
            n=end//divisor-(start-1)//divisor
            return n*(2*a+(n-1)*divisor)//2
        #So look at start and end consecutive primes
        #Semidivisible numbers are numbers that are multiple of start and next but not both
        total_sum+=SumMultiple(primes[i],start,end)+SumMultiple(primes[i+1],start,end)-2*SumMultiple(primes[i]*primes[i+1],start,end)

    return total_sum
print(compute_lps_and_ups_234())

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

def compute_278():
    until=5000
    sieve=get_sieve(until)
    primes=get_primes_from_sieve(sieve)
    sum=0
    x=len(primes)
    for i in range(0,x):
        for j in range(i+1,x):
            for k in range(j+1,x):
                #for any linear combination of 2 primes m=a*p+b*q
                #=>pq-p-q is impossible value for m via pigeonhole principle
                
                def GetLargestImpossibleValue(p,q,r):
                    return 2*p*q*r-p*q-p*r-q*r
                sum+=GetLargestImpossibleValue(primes[i],primes[j],primes[k])
    print(sum)
compute_278()
