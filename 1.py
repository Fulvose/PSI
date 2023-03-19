x1 = int(input())
y1 = int(input())
r1 = int(input())

x2 = int(input())
y2 = int(input())
r2 = int(input())

if (x1-x2)**2 + (y1-y2)**2 > (r1+r2)**2:
    print("Brak")
else:
    print("Istnieje")

user_input: str = ""
while user_input != "stop":
    user_input = input("Wpisuj slowa az do 'stop': ")
    print(user_input)


m = int(input("m= "))
n = int(input("n= "))

for i in range(m):
    for j in range(n):
        print( (i + j + 1 ) % 2, end="")
    print()



dl = int(input("Dlugosc miarki: "))
print(''.join(["|...." if i < dl else "|" for i in range(dl+1)]))
ss=''
k=0
for i in range(dl*5+1):
    if i%5 == 0:
        ss = ss + str(k)
        k += 1
    else:
        ss = ss+ ' '
print (ss)



def operations_on_list(lst):
    total_sum = sum(lst)
    
    difference = lst[0] - sum(lst[1:])
    
    product = 1
    for num in lst:
        product *= num
    
    return (total_sum, difference, product)




def funkcja(napis, czym, ilosc):
    separator=''
    for i in range(ilosc):
        separator = separator + czym
    napis = separator.join(litera for litera in napis)
    print(napis)
   
napis = input()
funkcja(napis, 'x', 5)



def reverse(napis):
    return napis[::-1]


import random

def estimate_pi(num_points):
    points_in_circle = 0
    for i in range(num_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            points_in_circle += 1
    pi_estimate = 4 * points_in_circle / num_points
    return pi_estimate