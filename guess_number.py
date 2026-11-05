import random

num=random.randint(1,10)
guess_number=int(input("Enter an number:"))
 
if guess_number == num:
    print("Congratualation! You have guessed the correct number which is ",num)
else:
    print("Try Again! The correct number is",num,)
