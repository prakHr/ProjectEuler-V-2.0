n=int(input())
a=[]
for i in range(0,n):
    ele=int(input())
    a.append(ele)
print(list(set(a)))


lower=int(input())
upper=int(input())
b=[x for x in range(lower,upper+1) if ( int(x**0.5))**2==x and sum(list(map(int,str(x))))<10 ]
a=[(x,x**2)for x in range(lower,upper+1)]
print(b)

n=int(input())
m=int(input())
a=[]
b=[]
for c in range(0,n):
    ele=int(input())
    a.append(ele)
for c in range(0,m):
    ele=int(input())
    b.append(ele)
print(list(set(a)&set(b)))


n=int(input())
a=[]
b=[]
c=[]
for i in range(1,n+1):
    x=int(input())
    a.append(x)
    if(x%2==0):
        b.append(x)
    else:
        c.append(x)   
a.sort()
print(a[n-1])
for i in range(0,len(b)):
    print(b[i],sep=" ",end=" ")



n= int (input())
sieve=set(range(2,n+1))
while sieve:
    prime=min(sieve)
    print(prime,end="\t")
    sieve=sieve-set(range(prime,n+1,prime))
print()

n=int(input())
for i in range(n,0,-1):
    print((n-i)*''+i*'*')
    
n=int(input())
for i in range(1,n+1):
    for j in range(1,n+1):
        if(i==j):
            print(1,sep=" ",end=" ")
        else:
            print(0,sep=" ",end=" ")
    print()


n=int(input())
for j in range(1,n+1):
    a=[]
    for i in range(1,j+1):
        print(i,sep=" ",end=" ")
        if(i<j):
            print("+",sep=" ",end=" ")
        a.append(i)
    print("=",sum(a))
        
n=int(input())
count=0
while(n>0):
    count=count+1
    n=n//10
print(count)

n=int(input())
temp=n
rev=0
while(n>0):
    dig=n%10
    rev=rev*10+dig
    n=n//10
if(temp==rev):
    print("YES")
    
n=int(input())
temp=str(n)
a=temp+temp
b=temp+temp+temp
print(n+int(a)+int(b))

n=int(input())
a=[]
for i in range(1,n+1):
    a.append(i)


print(sum(a))

A=int(input())
B=int(input())
C=int(input())
d=[]
d.append(A)
d.append(B)
d.append(C)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            if(i!=j & j!=k & k!=i):
                print(d[i],d[j],d[k])
   
