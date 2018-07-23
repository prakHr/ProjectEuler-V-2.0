
def f(n):
    s=0
    while(n!=0):
        s+=(n%10)**2
        n=n//10
    return s

def is_perfect_square(x):
    a=x**0.5
    return a==int(a)
    
def compute_171():
    sumx=0
    for n in range(1,10**20):
        x=f(n)
        if(is_perfect_square(x)):
            sumx+=x
    return (sumx%9)
print(compute_171())


import math,fractions,itertools

def compute_105():
    ans=sum(sum(s) for s in SETS if is_special_sum_set(s))
    return str(ans)

def is_special_sum_set(s):
    sums_seen=set()
    duplicate_seen=[False]
    min_sum=[None]*(len(s)+1)
    max_sum=list(min_sum)
    #many things to notice here
    #max_sum[i] is taking max sum of elements taken 1,2,3... at a time
    #Similarly,min_sum[i] is taking min sum of elements in a subsets
    #2nd condition can be interpreted as min_sum[i+1]>max_sum[i]

    def explore_subsets(i,count,sum):
        if i==len(s):
            if sum in sums_seen:
                duplicate_seen[0]=True
            else:
                #IN RECURSION TREE ALL ELEMENTS OF A SUBSET WILL BE DISJOINT
                #THIS IS TAKING CARE OF BY sums_seen AS 
                sums_seen.add(sum)
                if min_sum[count] is None or sum < min_sum[count]:
                    min_sum[count]=sum
                if max_sum[count] is None or sum > max_sum[count]:
                    max_sum[count]=sum
        else:
            #RECURSION IS HAPPENING HERE,DISJOINT SUBSETS ARE TRAVERSED THROUGH RECURSION
            #FIRST RECURSION TO BUILD FULL HEIGHT OF TREE
            explore_subsets(i+1,count,sum)
            #SECOND RECURSION TO CREATE SUBSETS WITH SUM TAKEN 1,2,3... AT A TIME FOR COUNT 1,2,3...
            explore_subsets(i+1,count+1,sum+s[i])

    explore_subsets(0,0,0)
    return not duplicate_seen[0] and all(max_sum[i] < min_sum[i+1] for i in range(len(s)))

SETS=[  [81,88,75,42,87,84,86,65],
	[157,150,164,119,79,159,161,139,158],
	[673,465,569,603,629,592,584,300,601,599,600],
	[90,85,83,84,65,87,76,46],
	[165,168,169,190,162,85,176,167,127],
	[224,275,278,249,277,279,289,295,139],
	[354,370,362,384,359,324,360,180,350,270],
	[599,595,557,298,448,596,577,667,597,588,602],
	[175,199,137,88,187,173,168,171,174],
	[93,187,196,144,185,178,186,202,182],
	[157,155,81,158,119,176,152,167,159],
	[184,165,159,166,163,167,174,124,83],
	[1211,1212,1287,605,1208,1189,1060,1216,1243,1200,908,1210],
	[339,299,153,305,282,304,313,306,302,228],
	[94,104,63,112,80,84,93,96],
	[41,88,82,85,61,74,83,81],
	[90,67,84,83,82,97,86,41],
	[299,303,151,301,291,302,307,377,333,280],
	[55,40,48,44,25,42,41],
	[1038,1188,1255,1184,594,890,1173,1151,1186,1203,1187,1195],
	[76,132,133,144,135,99,128,154],
	[77,46,108,81,85,84,93,83],
	[624,596,391,605,529,610,607,568,604,603,453],
	[83,167,166,189,163,174,160,165,133],
	[308,281,389,292,346,303,302,304,300,173],
	[593,1151,1187,1184,890,1040,1173,1186,1195,1255,1188,1203],
	[68,46,64,33,60,58,65],
	[65,43,88,87,86,99,93,90],
	[83,78,107,48,84,87,96,85],
	[1188,1173,1256,1038,1187,1151,890,1186,1184,1203,594,1195],
	[302,324,280,296,294,160,367,298,264,299],
	[521,760,682,687,646,664,342,698,692,686,672],
	[56,95,86,97,96,89,108,120],
	[344,356,262,343,340,382,337,175,361,330],
	[47,44,42,27,41,40,37],
	[139,155,161,158,118,166,154,156,78],
	[118,157,164,158,161,79,139,150,159],
	[299,292,371,150,300,301,281,303,306,262],
	[85,77,86,84,44,88,91,67],
	[88,85,84,44,65,91,76,86],
	[138,141,127,96,136,154,135,76],
	[292,308,302,346,300,324,304,305,238,166],
	[354,342,341,257,348,343,345,321,170,301],
	[84,178,168,167,131,170,193,166,162],
	[686,701,706,673,694,687,652,343,683,606,518],
	[295,293,301,367,296,279,297,263,323,159],
	[1038,1184,593,890,1188,1173,1187,1186,1195,1150,1203,1255],
	[343,364,388,402,191,383,382,385,288,374],
	[1187,1036,1183,591,1184,1175,888,1197,1182,1219,1115,1167],
	[151,291,307,303,345,238,299,323,301,302],
	[140,151,143,138,99,69,131,137],
	[29,44,42,59,41,36,40],
	[348,329,343,344,338,315,169,359,375,271],
	[48,39,34,37,50,40,41],
	[593,445,595,558,662,602,591,297,610,580,594],
	[686,651,681,342,541,687,691,707,604,675,699],
	[180,99,189,166,194,188,144,187,199],
	[321,349,335,343,377,176,265,356,344,332],
	[1151,1255,1195,1173,1184,1186,1188,1187,1203,593,1038,891],
	[90,88,100,83,62,113,80,89],
	[308,303,238,300,151,304,324,293,346,302],
	[59,38,50,41,42,35,40],
	[352,366,174,355,344,265,343,310,338,331],
	[91,89,93,90,117,85,60,106],
	[146,186,166,175,202,92,184,183,189],
	[82,67,96,44,80,79,88,76],
	[54,50,58,66,31,61,64],
	[343,266,344,172,308,336,364,350,359,333],
	[88,49,87,82,90,98,86,115],
	[20,47,49,51,54,48,40],
	[159,79,177,158,157,152,155,167,118],
	[1219,1183,1182,1115,1035,1186,591,1197,1167,887,1184,1175],
	[611,518,693,343,704,667,686,682,677,687,725],
	[607,599,634,305,677,604,603,580,452,605,591],
	[682,686,635,675,692,730,687,342,517,658,695],
	[662,296,573,598,592,584,553,593,595,443,591],
	[180,185,186,199,187,210,93,177,149],
	[197,136,179,185,156,182,180,178,99],
	[271,298,218,279,285,282,280,238,140],
	[1187,1151,890,593,1194,1188,1184,1173,1038,1186,1255,1203],
	[169,161,177,192,130,165,84,167,168],
	[50,42,43,41,66,39,36],
	[590,669,604,579,448,599,560,299,601,597,598],
	[174,191,206,179,184,142,177,180,90],
	[298,299,297,306,164,285,374,269,329,295],
	[181,172,162,138,170,195,86,169,168],
	[1184,1197,591,1182,1186,889,1167,1219,1183,1033,1115,1175],
	[644,695,691,679,667,687,340,681,770,686,517],
	[606,524,592,576,628,593,591,584,296,444,595],
	[94,127,154,138,135,74,136,141],
	[179,168,172,178,177,89,198,186,137],
	[302,299,291,300,298,149,260,305,280,370],
	[678,517,670,686,682,768,687,648,342,692,702],
	[302,290,304,376,333,303,306,298,279,153],
	[95,102,109,54,96,75,85,97],
	[150,154,146,78,152,151,162,173,119],
	[150,143,157,152,184,112,154,151,132],
	[36,41,54,40,25,44,42],
	[37,48,34,59,39,41,40],
	[681,603,638,611,584,303,454,607,606,605,596]]
print(compute_105())
                
            

def fractions_compute_26():
    ans=max(range(1,1000),key=reciprocal_cycle_len_algo)
    return str(ans)

def reciprocal_cycle_len_algo(n):
    seen={}
    x=1
    for i in itertools.count():
        if x in seen:
            return i-seen[x]
        else:
            seen[x]=i
            x=x*10%n
            
print(fractions_compute_26())



def compute_24():
    arr=list(range(10))
    temp=itertools.islice(itertools.permutations(arr),10**6-1,None)
    print(str(temp))#temp=<itertools.islice object at 0x034371E0>
    #it will return a pointer temp so we will use next to iterate to the value at that pointer
    return "".join(str(x)for x in next(temp))
print(compute_24())

def compute_108():
    for n in itertools.count(1):
        #n^2 will contain odd no of divisors say m so n^2=i*j and i will contain (m+1)/2 divisors as i<=j
        #and 1<i<n for unique solutions of j
        if( (count_divisors_squared(n)+1)//2 >1000 ):
            return str(n)
        
def count_divisors_squared(n):#n=(p1^a1)*(p2^a2)....
    #n^2=(p1^2a1)....
    # divisors=(2a1+1)*(2a2+1).....
    count=1
    end=int(n**0.5)
    #we can factor n as half of the factors will be < n and half > n for n^2 
    for i in itertools.count(2): 
        if i>end:
            break
        if n%i==0:
            j=0
            while True:
                n=n//i
                j+=1
                if n%i!=0:#checking all products of distinct primes p1,p2... keeping a1,a2... as j
                    break
            count=count*(2*j+1)
            
    if n!=1:
        count=count*3
    return count
print(compute_108())
    


def compute_120():
    ans=0
    for i in range(3,1001):
        if i%2==0:
            ans+=i*(i-2)
        else:
            ans+=i*(i-1)
    return ans
print(compute_120())


def compute_135():
    LIMIT=50*10**6
    solutions=[0]*(LIMIT)
    for m in range(1,2*LIMIT):
        for k in range(m//5+1,(m+1)//2):
            temp=(m-k)*(5*k-m)
            if temp>=LIMIT:
                break
            solutions[temp]+=1
    ans=solutions.count(1)
    return str(ans)
    
    
print(compute_135())

def compute_77():
    cond=lambda n:num_of_prime_sum_ways(n)>5000
    
    ans=next(filter(cond,itertools.count(2)))#how to print the first number of function,fiters all the values starting from 2,3,4,..inf and then mapped their index
    #with cond such that first no is obtained as we cant use print(list(map(cond))) as it requires 2 arguments 
    return str(ans)


def is_prime2(n):
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
    

primes=[2]

def num_of_prime_sum_ways(n):
    for i in range(primes[-1]+1,n+1):
        if is_prime2(i):
            primes.append(i)
    #primes below n=7 is in the list
    ways=[1]+[0]*n#ways[0]=1
    #using dynamic programming interesting take example n=7  then ways[1/2/3/4/5] exists 
    for p in primes:
        for i in range(n+1-p):
            ways[i+p]+=ways[i]
    return ways[n]

print( compute_77())



def digit_sum(n):
    x=0
    while(n!=0):
        x+=(n%10)**2
        n=n//10
    return x
    
def compute_92():
    LIMIT=10000000
    cnt=0
    for i in range(1,LIMIT):
        curr=i
        
        x=True
        while(x==True):
            next=digit_sum(curr)
            
            if(next==89 or next==1):
                if(next==89):
                    cnt+=1
                x=False
                
            prev=next
            curr=prev
    return cnt
print(compute_92())
            


def compute_85():
    TARGET= 2000000
    end=int(TARGET**0.5)+1
    #end=1415
    gen=((w,h) for w in range(1,end) for h in range(1,end))
    #3 things to notice here
    ##1. 2 functions from lambda can be passed as *xy for the purpose of single return value
    ##2. lambda is used as a key-value pair
    ##3. x+1(as x points,so x+1 lines) choices out of any 2 for forming horizontal lines of rectangle,(x+1)C(2)
    func=lambda wh: abs( num_rectangles(*wh)-TARGET  )
    ans=min(gen,key=func)
    print(ans)
    return str(ans[0]*ans[1])

def num_rectangles(m,n):
    return (m+1)*m*(n+1)*n//4

print(compute_85())

def compute_75():
    LIMIT=1500000
    triples=set()
    # Pythagorean triples theorem:
	#   Every primitive Pythagorean triple with a odd and b even can be expressed as
	#   a = st, b = (s^2-t^2)/2, c = (s^2+t^2)/2, where s > t > 0 are coprime odd integers.
	# 
    for s in range(3,int(LIMIT**0.5)+1,2):
        for t in range(s-2,0,-2):
            if fractions.gcd(s,t)==1:
                a=s*t
                b=(s*s-t*t)//2
                c=(s*s+t*t)//2
                if a+b+c <= LIMIT:
                    triples.add((a,b,c))
                    
    ways=[0]*(LIMIT+1)
    for triple in triples:
        sigma=sum(triple)
        for i in range(sigma,len(ways),sigma):
            ways[i]+=1
            
    ans=ways.count(1)
    return str(ans)

print(compute_75())
            






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
    
print(is_prime(13))
    
    

def compute_58():
	TARGET = fractions.Fraction(1, 10)
	numprimes = 0
	for i in itertools.count(1, 2):
		for j in range(4):
			if is_prime(i * i - j * (i - 1)):
				numprimes += 1
		if i > 1 and fractions.Fraction(numprimes, i * 2 - 1) < TARGET:
			return str(i)


if __name__ == "__main__":
	print(compute_58())










def compute_longest_chain():
    LIMIT=10**6

    divisorsum=[0]*(LIMIT+1)
    for i in range(1,LIMIT+1):
        for j in range(i*2,LIMIT+1,i):
            divisorsum[j]+=i
            
    max_chain_len=0
    ans=-1
    for i in range(LIMIT+1):
        visited=set()
        curr=i
        for count in itertools.count(1):
            visited.add(curr)
            #retrieve the next item from container(can be used as variable as nextx)
            next=divisorsum[curr]
            if next==i:
                if count>max_chain_len:
                    ans=i
                    max_chain_len=count
                break
            elif next>LIMIT or next in visited:
                break
            else:
                curr=next
    return str(ans)
print(compute_longest_chain())




















def isPentagonal(n):
    x=(1+((1+24*n)**0.5))/6
    if(x==int(x)):
        return 1
    else:
        return 0

result=0
i=143
while(True):
    i+=1
    result=i*(2*i-1)
    if(isPentagonal(result)):
        break
print(result)


def form_pandigital(number):
    n=2
    str_number=str(n)
    while len(str_number)<9:
        str_number+=str(n*number)
        n+=1

    if '0' not in str_number and len(str_number)==9 and len(set(list(str_number)))==9:
        return int(str_number)
    else:
        return None

max_pandigital=0
number=1
while len(str(number)+str(number*2))<=9:
    pandigital=form_pandigital(number)
    if pandigital:
        max_pandigital=max(max_pandigital,pandigital)
    number+=1
print (max_pandigital)

def check(x,y,z):
    numbers=10*[0]
    while(x>0):
        if(numbers[x%10]==1):
            return 0
        numbers[x%10]=1
        x=x//10
    while(y>0):
        if(numbers[y%10]==1):
            return 0
        numbers[y%10]=1
        y=y//10
    while(z>0):
        if(numbers[z%10]==1):
            return 0
        numbers[z%10]=1
        z=z//10
    if(numbers[0]==1):
        return 0
    i=1
    while(i<=9):
        if(numbers[i]==0):
            return 0
        i+=1
    return 1
        

product=set()
i=2

while(i<=999):
    j=i+1
    while(j<=9999):
        if(check(i,j,i*j)):
            product.add(i*j)
        j+=1
    i+=1
print(sum(product)) 



from math import sqrt
def divisors(n):
    divisors=[1]
    for i in range(2,int(sqrt(n))+1):
        if(n%i==0):
            divisors.append(i)
            if(n/i!=i):
                divisors.append(n/i)
    divisors.sort()
    return divisors

def is_abundant(n):
    return (sum(divisors(n))>n)

def can_be_written_as_sum(num,numbers):
    for num1 in numbers:
        if num-num1 in numbers:
            return True
    return False

def non_abundant_sum(limit):
    abundant_numbers=set()
    total=0
    for i in range(1,limit):
        if is_abundant(i):
            abundant_numbers.add(i)

        if not can_be_written_as_sum(i,abundant_numbers):
            total+=i
            
    return total

print(non_abundant_sum(28123))

def fibonacci(until):
    prev=0
    current=1
    fib=[]
    i=0
    while(i<until):
        fib.append(current)
        current=prev+current
        prev=current-prev
        i=i+1
    return fib
print(fibonacci(12))

def first_fibonacci_term_with(digits):
    term=1
    prev=0
    current=1
    while(len(str(current))<digits):
        current=current+prev
        prev=current-prev
        term=term+1
    return term

print(first_fibonacci_term_with(1000))














def divisors(n):
    divs=[1]
    for i in range(2,int(n**0.5)+1):
        if(n%i==0):
            divs.append(i)
            if(n/i!=i):
                divs.append(n/i)
    return divs

def sum_of_divisors(until):
    divisors_sum={}
    for i in range(1,until+1):
        divisors_sum[i]=sum(divisors(i))
    return divisors_sum

def amicable_numbers(until):
    divisors_sum=sum_of_divisors(until)
    amicables=[]
    for i in range(1,until+1):
        if (divisors_sum[i] in divisors_sum and i!=divisors_sum[i] and i==divisors_sum[divisors_sum[i]]):
            amicables.append(i)
    return amicables

print(amicable_numbers(10000))

def amicable_sum(until):
    return(sum(amicable_numbers(until)))


print(amicable_sum(10000))

count=0
den=2
num=3
for i in range(1,1000):
    num=num+2*den
    den=num-den
    if(len(str(num))>len(str(den))):
        count=count+1
print(count)
    



count=0
for power in range(1,25):
    for i in range(1,100):
        n=i**power
        if(len(str(n))==power):
           count=count+1
print(count)



import time
 
# read file
rows = []
FILE = open("triangle.txt", "r")
for blob in FILE: rows.append([int(i) for i in blob.split(" ")])
 
start = time.time()
 
for i,j in [(i,j) for i in range(len(rows)-2,-1,-1) for j in range(i+1)]:
    rows[i][j] +=  max([rows[i+1][j],rows[i+1][j+1]])
 
elapsed = time.time() - start
 
print( "%s found in %s seconds" % (rows[0][0],elapsed))


list=[]
for a in range (2,101):
    for b in range(2,101):
        list.append(pow(a,b))
print(len(set(list)))
        
        




from math import sqrt
score=(lambda word:sum(ord(c)-64 for c in word) )
istn=(lambda t:(((sqrt(1+8*t)-1)/2)%1)==0  )
print  ( sum( istn(score(x[1:-1])) for x in open('words.txt').read().split(',')))  
