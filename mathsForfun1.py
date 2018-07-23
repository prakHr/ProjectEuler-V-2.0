def fibo(n):
    if n==1:
        return [1]
    elif n==2:
        return [1,1]
    else:
        a=1
        b=1
        for i in range(n):
             series=[a,b]
             c=a+b
             series.append(c)
             a=b
             b=c
        return series

print(fibo(input()))

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
