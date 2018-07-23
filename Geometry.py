from itertools import count
from itertools import permutations


def solve_222():
    
    ball_arrangement=[49,47,45,43,41,39,37,35,33,31,30,32,34,36,38,40,42,44,46,48,50]

    def get(ball_arrangement):
        distance=ball_arrangement[0]+ball_arrangement[-1]
        ball_count=len(ball_arrangement)
        for i in range(0,ball_count-1):
            def get_distance(a,b):
                #convert 3D objects into 2D by seeing 2 balls a and b stacked on
                #top of each other and reducing them to circles and lines
                #formula is hyp^2-x^2=(a+b)^2-(100-(a+b)^2)=100*(2(a+b)-100)
                return ((200*(a+b-50))**0.5)
            distance+=get_distance(ball_arrangement[i],ball_arrangement[i+1])
        return distance
    
    def benchmark():
        best_arrangement_so_far=None
        shortest_so_far=300000
        all_balls=[i for i in range(30,51)]
        for ball_arrangement in permutations(all_balls):
            total_distance=get(ball_arrangement)
            if total_distance<shortest_so_far:
                shortest_so_far=total_distance
                best_arrangement_so_far=ball_arrangement
        print(best_arrangement_so_far,shortest_so_far)
    
    
    #benchmark()
    print(get(ball_arrangement))

solve_222()
P=10**8

def gcd(x,y):
    while y!=0:
        x,y=y,x%y
    return x

total=0
for n in count(1):
    if 2*(n+1)*(2*n+1) >= P: break
    for m in count(n+1):
        if (m-n)%2==0 or gcd(m,n) != 1: continue
        p = 2*m*(m+n)
        if p >= P: break
        hyp = m**2+n**2
        diff = 2*m*n-m**2+n**2
        if hyp%diff==0:
            total += P//p
            #print p, m**2-n**2, 2*m*n, m**2+n**2
print (total)

def compute_174():
    MAX=1000000
    nb_squares=[0]*(MAX+1)
    #solution of diophantine equation x*x-y*y=n
    #considering x=a+b and y=a-b from symmetry purpose
    #so (a+b)^2-(a-b)^2=4*a*b ans should be a mutiple of 4
    #OR another way (a+2*k)^2-(a)^2=4*(k)*(a+k) accounts for +2
    for outer in range(3,int(2+MAX/4)+1):
        width=1
        while 1:
            inner=outer-2*width
            nb=outer*outer-inner*inner
            if nb>MAX or inner<=0:
                break
            nb_squares[nb]+=1
            width+=1
    out=0
    for i in range(1,MAX+1):
        if nb_squares[i]>=1 and nb_squares[i]<=10:
            out+=1
    return out
print(compute_174())

def functionForEuler_151():
    def f(a5,a4,a3,a2):
        result=0
        nb=a5+a4+a3+a2
        #if a4=0 and a2=0 and a3=0 and only a5 remains then expected prob.=0
        if(not a4 and not a3 and not a2):
            return 0
        if(1==nb):
            result+=1
        if(a5):
            #so calculate the probababilty of a5/nb and move onto smaller cuts a4,a3...
            result+=a5*f(a5-1,a4,a3,a2)/nb
        if(a4):
            #if u move onto a4 that means u are coming from a5 so a5-1
            result+=a4*f(a5+1,a4-1,a3,a2)/nb
        if(a3):
            #similarly a3 and a2
            result+=a3*f(a5+1,a4+1,a3-1,a2)/nb
        if(a2):
            result+=a2*f(a5+1,a4+1,a3+1,a2-1)/nb
        return result
    return str(f(1,1,1,1))
print(functionForEuler_151())
            
def euler_126():
    def CubesCovered(x,y,z,n):
        #consider only seeing from top side for a*b
        #In ith layer 2*(a+b)+4*(i-1) {for corners}
        #If we include a*c,then by induction
        #add c*(2*(a+b)+4*(i-1))+{also,2*a*b contribution}
        #Hence add new top and bottom levels each of dimension a*b
        #Expand all of the other levels in 2D
        #polynomial becomes c*(2*(a+b)+4*(i-1))+2(a*b+Sigma(j=1 to i-1)(2*(a+b)+4*(j-1))
        #simplification will give (2*(a*b+b*c+c*a)-4*(a+b+c)+8)+(4*(a+b+c)-12)*i+4*i*i
        return 2*(x*y+y*z+z*x)+4*(x+y+z+n-2)*(n-1)
    Limit=20000
    #assume z<=y<=x
    #Cubes(x,y,z,1)=2*(x*y+y*z+x*z),....
    #Now add layers z=>y=>x=>n assuming solution<20,000
    count=[0]*(Limit+1)
    z=1
    while CubesCovered(z,z,z,1)<=Limit:
        y=z
        while CubesCovered(z,y,z,1)<=Limit:
            x=y
            while CubesCovered(z,y,x,1)<=Limit:
                n=1
                while CubesCovered(z,y,x,n)<=Limit:
                    count[CubesCovered(z,y,x,n)]+=1
                    n+=1
                x+=1

            y+=1
            
        z+=1
    i=0
    for count[i] in count:
        if count[i]==1000:
            x=i
            break
        i+=1
    return x
print(euler_126())

def gcd(a,b):
  
    if a==0:
        return b
    else:
        return gcd(b%a,a)

def RSA_Encryption_min_unconcealed_messages():
    p=1009
    q=3643
    n=1009*3643
    phi=1008*3642
    min_unconcealed=n
    sum=0
    nb=0
    for e in range(2,phi):
        if 1==gcd(e,phi):
            #we have to find the number of unconcealed messages
            unconcealed=(gcd(e-1,p-1)+1)*(gcd(e-1,q-1)+1)
            if unconcealed<min_unconcealed:
                sum=0
                min_unconcealed=unconcealed
            if(unconcealed==min_unconcealed):
                nb+=1
                sum+=e
    return sum
print(RSA_Encryption_min_unconcealed_messages())
def inscribed_circle(r):
    count=0
    bound=int(2*r*3**0.5)
    for m in range(1,bound+1):
        for n in range(1,m):
            if m*n>bound:
                break
            if gcd(m,n)!=1:
                continue
            a=(m+2*n)*m
            b=(n+2*m)*n
            c=m*m+m*n+n*n
            g=3 if (m-n)%3==0 else 1
            count+=int(r*g*2/(3**0.5*m*n))
    return str(count)
print (inscribed_circle(100))
