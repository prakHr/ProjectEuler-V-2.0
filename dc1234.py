import math
def sin(x,n):
    sine=0
    for i in range(n):
        sign=(-1)**i
        pi=22/7
        y=x*(pi/180)
        sine=sine+((y**(2*i+1))/math.factorial(2*i+1))**sign
        return sine
x=int(input())
n=int(input())
print(round(sin(x,n),2))

import fractions
a=int(input())
b=int(input())
min1=min(a,b)
while(1):
    if(min1%a==0 and min1%b==0):
        print("LCM,GCD is:",min1,fractions.gcd(a,b))
        break
    min1=min1+1
    

n=int(input())
sum1=0
temp=n
while(n):
    i=1
    f=1
    r=n%10
    while(i<=r):
        f=f*i
        i=i+1
    sum1=sum1+f
    n=n//10
if(sum1==temp):
    print('strong no')


n=int(input())
a=[]
for i in range(n):
    a.append([])
    a[i].append(1)
    for j in range(1,i):
        a[i].append(a[i-1][j-1]+a[i-1][j])
    if(n!=0):
        a[i].append(1)
for i in range(n):
    print(" "*(n-i),end=" ",sep=" " )
    for j in range(0,i+1):
        print('{0:6}'.format(a[i][j]),end=" ",sep=" ")
    print()

n=int(input())
a=list(map(int,str(n)))
b=list(map(lambda x:x**3,a))
if(sum(b)==n):
    print('armstrong')

a=[]
n=int(input())
for i in range(1,n+1):
    a.append(int(input()))
sum1=0
sum2=0
sum3=0
for x in a:
    if(x<0):
        sum1=sum1+x
    elif(x>0 and x%2==0):
        sum2=sum2+x
    else:
        sum3+=x
print(sum1,sum2,sum3)
    
    

n=int(input())
for i in range(1,11):
    print(n,"x",i,"=",n*i)

n=int(input())
i=1
while(i<=n):
    k=0
    if(n%i==0):
        j=1
        while(j<=i):
            if(i%j==0):
                k=k+1
            j=j+1
        if(k==2):
            print(i)
    i=i+1


cm=int(input())
inches=0.394*cm
feet=0.0328*cm
print(round(inches,2))

celsius=int(input())
f=(celsius*1.8)+32


date=input()
dd,mm,yy=date.split('/')
dd=int(dd)
mm=int(mm)
yy=int(yy)
if(mm==1 or mm==3 or mm==5 or mm==7 or mm==8 or mm==10 or mm==12):
    max1=31
elif(mm==4 or mm==6 or mm==9 or mm==11):
    max1=30
elif(yy%4==0 and yy%100!=0 or yy%400==0):
    max1=29
else:
    max1=28
if(mm<1 or mm>12 or dd<1 or dd>max1):
    print(-1)
elif(dd==max1 and mm!=12):
    dd=1
    mm=mm+1
    print(dd,mm,yy)
elif(dd==max1 and mm==12):
    dd=1
    mm=1
    yy=yy+1
    print(dd,mm,yy)
else:
    dd=dd+1
    print(dd,mm,yy)
    

