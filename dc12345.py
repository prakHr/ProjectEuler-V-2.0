string=input()
substring=input()
if(string.find(substring)==-1):
    print('not found')
else:
    print('fwd')

string=input()
word=input()
a=[]
count=0
a=string.split(" ")
for i in range(len(a)):
    if(word==a[i]):
        count=count+1
print (count)

lst=[n for n in input().split('-')]
lst.sort()
print('-'.join(lst))

string=input()
char=0
word=1
numbers=0
for i in string:
    if(i.isdigit()):
        numbers=numbers+1
    char=char+1
    if(i==' '):
        word=word+1
print(word,char)

from string import ascii_lowercase as asc_lower
def check(s):
    return set(asc_lower)-set(s.lower())==set([])
string=input()
if(check(string)==True):
    print('pangram')

string=input()
for i in string:
    if(i.islower()):
        count=count+1
print(count)

def modify(string):
    final=""
    for i in range(len(string)):
        if i%2 ==0:
            final=final+string[i]
    return final
string=input()
print(modify(string))

string=input()
print(string.replace(' ','~'))


def change(string):
    return string[-1:]+string[1:-1]+string[:1]
string=input()
print(change(string))

def remove(string,n):
    first=string[:n]
    last=string[n+1:]
    return first+last
string=input()
n=int(input())
print(remove(string,n))

strin1=input()
strin2=input()
if(sorted(strin1)==sorted(strin2)):
    print("anagrams")

    
string=input()
string=string.replace('a','$')
print(string)
