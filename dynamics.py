import itertools
def compute():
	LENGTH = 20
	BASE = 10
	MODULUS = 10**9
	
	# Maximum possible squared digit sum (for 99...99)
	MAX_SQR_DIGIT_SUM = (BASE - 1)**2 * LENGTH
	
	# sqsum[n][s] is the sum of all length-n numbers with a square digit sum of s, modulo MODULUS
	# count[n][s] is the count of all length-n numbers with a square digit sum of s, modulo MODULUS
	sqsum = []
	count = []
	
	for i in range(LENGTH + 1):
		sqsum.append([0] * (MAX_SQR_DIGIT_SUM + 1))
		count.append([0] * (MAX_SQR_DIGIT_SUM + 1))
		if i == 0:
			count[0][0] = 1
		else:
			for j in range(BASE):
                                #k starts from 0
				for k in itertools.count():
					index = k + j**2
					if index > MAX_SQR_DIGIT_SUM:
						break
					sqsum[i][index] = (sqsum[i][index] + sqsum[i - 1][k] + pow(BASE, i - 1, MODULUS) * j * count[i - 1][k]) % MODULUS
					count[i][index] = (count[i][index] + count[i - 1][k]) % MODULUS
	print(count[1][36])
	ans = sum(sqsum[LENGTH][i**2] for i in range(1, int(MAX_SQR_DIGIT_SUM**0.5)))
	return "{:09d}".format(ans % MODULUS)


if __name__ == "__main__":
	print(compute())

def compute_173():
        SIZE=10**6
        ans=0
        #number of tiles in square of length n containing a-1's hole like a frame is a=4*n-4,so n=a/4+4
        for n in range(3,SIZE//4+2):
                for m in range(n-2,0,-2):#we cant do the reverse as we have to start by putting a hole in centre of size n-2(1 square removed frm each side) till 2 
                        tile=n*n-m*m
                        if(tile>SIZE):
                                break
                        ans+=1
        return str(ans)
print(compute_173())
                        

def compute():
	LENGTH = 50
	print(count_ways(LENGTH, 2)+count_ways(LENGTH, 3)+count_ways(LENGTH, 4)) 


# How many ways can a row n units long be filled with black squares 1 unit long
# and colored tiles m units long? Denote this quantity as ways[n].
# Compute n = 0 manually as a base case.
# 
# Now assume n >= 1. Look at the leftmost item and sum up the possibilities.
# - If the item is a black square, then the rest of the row
#   is allowed to be anything of length n-1. Add ways[n-1].
# - If the item is a colored tile of length m where m <= n, then the
#   rest of the row can be anything of length n-m. Add ways[n-m].
# 
# At the end, return ways[length]-1 to exclude the case where the row is all black squares.
def count_ways(length, m):
	# Dynamic programming
	ways = [1] + [0] * length
	for n in range(1, len(ways)):
		ways[n] += ways[n - 1]
		if n >= m:
			ways[n] += ways[n - m]
	return ways[-1] - 1



if __name__ == "__main__":
	print(compute())


def partition():
    partitions=[1]
    for i in itertools.count(len(partitions)):
        item=0
        for j in itertools.count(1):
            sign=-1 if j%2==0 else +1
            index=(j*j*3-j)//2
            if index>i:
                break
            item+=partitions[i-index]*sign
            
            index+=j
            if index>i:
                break
            item+=partitions[i-index]*sign
            item=item%(10**6)
        if item==0:
            return str(i)
        partitions.append(item)


print(partition())


 #partitions(i) =     partitions(i - pentagonal(1)) + partitions(i - pentagonal(-1))
#                   - partitions(i - pentagonal(2)) - partitions(i - pentagonal(-2))
#                   + partitions(i - pentagonal(3)) + partitions(i - pentagonal(-3))
#                   - partitions(i - pentagonal(4)) - partitions(i - pentagonal(-4))
#                   + ...,




def binomial(n,k):
    
    C=[[0 for x in range(k+1)] for x in range(n+1)]
    for i in range(n+1):
        for j in range(min(i,k)+1):
            if j==0 or j==i:
                C[i][j]=1
            else:
                C[i][j]=C[i-1][j]+C[i-1][j-1]
    return C[n][k]
    
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



def compute_Pascal_203():
    
    numbers = set(binomial(n, k) for n in range(51) for k in range(n + 1))
   
    maximum=max(numbers)
    
    
   
    def is_squarefree(n,maximum):
        sieve=get_sieve(100)
        primes=get_primes_from_sieve(sieve)
        
        squared_primes=[p*p for p in primes]
        
        for p2 in squared_primes:
            if p2>n:
                break
            if n%p2==0:
                return False
        return True
    

    ans=sum(n for n in numbers if is_squarefree(n,maximum))
    return str(ans)
    
print(compute_Pascal_203())    
    
    






arr=[1,2,5,10,20,50,100,200]
x=len(arr)
print(x)
ways=[0]*201
ways[0]=1
target=200
for i in range(0,x):
    for j in range(arr[i],target+1):
        ways[j]+=ways[ j-arr[i] ]#basically ways to arrange 3 is
        #to calculate ways[j] using ways[j-1],ways[j-2]
print(ways)
